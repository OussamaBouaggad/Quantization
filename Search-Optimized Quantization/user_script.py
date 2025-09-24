# -------------------------------------------------------------------------
# End-to-End Optimization Pipeline
# -------------------------------------------------------------------------

import copy
import numpy as np
import torch
import torch.quantization
import torchmetrics
from torchmetrics import MeanSquaredError
import transformers
from datasets import load_dataset, load_metric
from datasets.utils import logging as datasets_logging
from neural_compressor.data import DefaultDataLoader
import intel_extension_for_pytorch as ipex
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

from olive.constants import Framework
from olive.data.registry import Registry
from olive.model import OliveModelHandler

datasets_logging.disable_progress_bar()
datasets_logging.set_verbosity_error()

# pylint: disable=attribute-defined-outside-init, protected-access


# -------------------------------------------------------------------------
# Model Loader
# -------------------------------------------------------------------------

import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-KRISSBERT-PubMed-UMLS-EL')

model = AutoModel.from_pretrained('microsoft/BiomedNLP-KRISSBERT-PubMed-UMLS-EL')

def load_origin_model(model_path=None):
    if model_path is None:
        model_path = "microsoft/BiomedNLP-KRISSBERT-PubMed-UMLS-EL"  # Default model

    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    def model_forward(*args, **kwargs):
        with torch.no_grad():
            return model(*args, **kwargs)

    model.forward = model_forward 
    return model

def load_optimized_model(model_path=None):
    if model_path is None:
        model_path = "models/optimized_model"

    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    def model_forward(*args, **kwargs):
        with torch.no_grad():
            return model(*args, **kwargs)

    model.forward = model_forward 
    return model


# -------------------------------------------------------------------------
# Dummy Input for ONNX Export
# -------------------------------------------------------------------------

def create_input_tensors(model):
    seq_length = min(model.config.max_position_embeddings, model.config.model_max_length)
    return {
        "input_ids": torch.ones(1, seq_length, dtype=torch.long),
        "attention_mask": torch.ones(1, seq_length, dtype=torch.long),
        "token_type_ids": torch.ones(1, seq_length, dtype=torch.long),
    }


# -------------------------------------------------------------------------
# Common Dataset
# -------------------------------------------------------------------------

default_data_collator = None 

class BertDataset:
    def __init__(self, model_name_or_path):
        self.model_name_or_path = model_name_or_path
        
        self.task_name = "medsts"
        self.max_seq_length = 128
        self.data_collator = default_data_collator
        self.padding = "max_length"

        self.config = AutoConfig.from_pretrained(
            self.model_name_or_path,
            num_labels=1,
            finetuning_task=self.task_name,
            cache_dir=None,
            revision="main",
            use_auth_token=None,
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            cache_dir=None,
            use_fast=True,
            revision="main",
            use_auth_token=None,
        )
        
        self.setup_dataset()

    def setup_dataset(self):
        self.raw_datasets = load_dataset("csv", data_files={
            "train": "medsts_train.csv",
            "validation": "medsts_dev.csv"
        })

        self.raw_datasets = self.raw_datasets.map(
            self.preprocess_function,
            batched=True,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )

    def preprocess_function(self, examples):
        sentence1_key, sentence2_key = "sentence1", "sentence2"
        args = (examples[sentence1_key], examples[sentence2_key])
        
        tokenized_inputs = self.tokenizer(
            *args, padding=self.padding, max_length=self.max_seq_length, truncation=True
        )
        
        if examples.get("score") is not None:
            tokenized_inputs["label"] = [float(label) for label in examples["score"]]
        
        return tokenized_inputs

    def get_train_dataset(self):
        return self.raw_datasets["train"]

    def get_eval_dataset(self):
        return self.raw_datasets["validation"]

class BertDatasetWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        input_dict = {
            "input_ids": torch.tensor(sample["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(sample["attention_mask"], dtype=torch.long),
        }
        
        if "token_type_ids" in sample:
            input_dict["token_type_ids"] = torch.tensor(sample["token_type_ids"], dtype=torch.long)
        
        label = torch.tensor(sample["label"], dtype=torch.float)
        return input_dict, label


# -------------------------------------------------------------------------
# Post Processing Function for Accuracy Calculation
# -------------------------------------------------------------------------

def post_process(output):
    if hasattr(output, "logits"):
        preds = output.logits.squeeze()
    else:
        preds = output.squeeze()
    
    return preds


# -------------------------------------------------------------------------
# Dataloader for Evaluation and Performance Tuning
# -------------------------------------------------------------------------

@Registry.register_dataset()
def create_bert_dataset(model_name=None, *args, **kwargs):
    if model_name is None:
        model_name = "microsoft/BiomedNLP-KRISSBERT-PubMed-UMLS-EL"
    return BertDataset(model_name)

@Registry.register_dataloader()
def create_bert_dataloader(dataset, batch_size=8, *args, **kwargs):
    if "collate_fn" not in kwargs:
        kwargs["collate_fn"] = default_data_collator or None
    
    return DataLoader(
        BertDatasetWrapper(dataset.get_eval_dataset()),
        batch_size=batch_size,
        drop_last=True,
        shuffle=False,
        collate_fn=kwargs["collate_fn"]
    )

def create_dataloader(model_name=None, batch_size=8, *args, **kwargs):
    dataset = create_bert_dataset(model_name)
    return create_bert_dataloader(dataset, batch_size, *args, **kwargs)


# -------------------------------------------------------------------------
# Calibration Data Reader for Intel® Neural Compressor Quantization
# -------------------------------------------------------------------------

class IncBertDataset:
    """Dataset for Intel® Neural Compressor must implement __iter__ or __getitem__ magic method."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        input_dict = {
            "input_ids": torch.tensor(sample["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(sample["attention_mask"], dtype=torch.long),
        }
        
        if "token_type_ids" in sample:
            input_dict["token_type_ids"] = torch.tensor(sample["token_type_ids"], dtype=torch.long)
        
        label = torch.tensor(sample["label"], dtype=torch.float)
        return input_dict, label

def inc_medsts_calibration_reader(model_name=None, batch_size=8, *args, **kwargs):
    if model_name is None:
        model_name = "microsoft/BiomedNLP-KRISSBERT-PubMed-UMLS-EL"
    
    bert_dataset = BertDataset(model_name)
    bert_dataset = IncBertDataset(bert_dataset.get_eval_dataset())
    
    return DataLoader(dataset=bert_dataset, batch_size=batch_size, drop_last=False, shuffle=False)


# -------------------------------------------------------------------------
# Accuracy Calculation Function
# -------------------------------------------------------------------------

def eval_accuracy(model: OliveModelHandler, data_dir, batch_size, device, execution_providers):
    dataloader = create_dataloader(data_dir, batch_size)
    preds = []
    target = []
    
    sess = model.prepare_session(inference_settings=None, device=device, execution_providers=execution_providers)
    
    if model.framework == Framework.ONNX:
        input_names = [i.name for i in sess.get_inputs()]
        output_names = [o.name for o in sess.get_outputs()]
        
        for inputs_i, labels in dataloader:
            input_dict = {k: inputs_i[k].tolist() for k in inputs_i if k in input_names}
            
            res = model.run_session(sess, input_dict)
            result = torch.tensor(res[0] if len(output_names) == 1 else res)
            
            outputs = post_process(result)
            preds.extend(outputs.tolist())
            target.extend(labels.gpu().numpy().tolist())

    elif model.framework == Framework.PYTORCH:
        with torch.no_grad():
            for inputs, labels in dataloader:
                result = model.run_session(sess, inputs)
                outputs = post_process(result)
                preds.extend(outputs.tolist())
                target.extend(labels.gpu().numpy().tolist())
    
    preds_tensor = torch.tensor(preds, dtype=torch.float)
    target_tensor = torch.tensor(target, dtype=torch.float)
    
    mse = MeanSquaredError()
    result = mse(preds_tensor, target_tensor)
    
    return result.item()


# -------------------------------------------------------------------------
# Trainer for Quantization Post Training
# -------------------------------------------------------------------------

def training_loop_func(model):
    set_seed(42)

    training_args = TrainingArguments("bert_qpt")
    training_args._n_gpu = 0
    training_args.learning_rate = 2e-5
    training_args.do_eval = True
    training_args.do_train = True
    training_args.per_device_train_batch_size = 8
    training_args.per_device_eval_batch_size = 8
    training_args.num_train_epochs = 2
    training_args.output_dir = "bert_qpt"
    training_args.seed = 42
    training_args.overwrite_output_dir = True
    training_args.eval_steps = 100
    training_args.save_steps = 100
    training_args.greater_is_better = False 
    training_args.load_best_model_at_end = True
    training_args.evaluation_strategy = "steps"
    training_args.save_strategy = "steps"
    training_args.save_total_limit = 1
    training_args.metric_for_best_model = "mse" 

    bert_dataset = BertDataset("microsoft/BiomedNLP-KRISSBERT-PubMed-UMLS-EL")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=bert_dataset.get_train_dataset(),
        eval_dataset=bert_dataset.get_eval_dataset(),
        compute_metrics=compute_metrics,
        tokenizer=bert_dataset.tokenizer,
        data_collator=bert_dataset.data_collator,
    )

    trainer.train(resume_from_checkpoint=None)
    
    model.eval()
    
    model_quantized = torch.quantization.quantize_dynamic(
        model, 
        {torch.nn.Linear}, 
        dtype=torch.qint8 
    )

    torch.save(model_quantized.state_dict(), "bert_qpt/model_quantized.bin")

    trainer.save_state()

    model_trained = copy.deepcopy(model)
    model_trained.load_state_dict(torch.load("bert_qpt/model_quantized.bin"), strict=False)
    return model_trained

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = preds.squeeze()
    metric = MeanSquaredError()
    result = metric(torch.tensor(preds, dtype=torch.float), torch.tensor(p.label_ids, dtype=torch.float))
    return {"mse": result.item()}

def qpt_post_process(output):
    if isinstance(output, (transformers.modeling_outputs.SequenceClassifierOutput, dict)):
        preds = output["logits"].squeeze().float()
    else:
        try:
            preds = output[0].squeeze().float()
        except Exception:
            preds = output.squeeze().float()
    return preds


# -------------------------------------------------------------------------
# Optimization with Intel® Extension for PyTorch
# -------------------------------------------------------------------------

if __name__ == "__main__":

    model = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/BiomedNLP-KRISSBERT-PubMed-UMLS-EL"
    )
    model.load_state_dict(torch.load("bert_qpt/model_quantized.bin"), strict=False)
    model.eval()

    model_ipex = ipex.optimize(model, dtype=torch.qint8)
    model_ipex.eval()

    torch.save(model_ipex.state_dict(), "bert_qpt/final_model.bin")
    print("IPEX optimized int8 model saved to 'bert_qpt/final_model.bin'")

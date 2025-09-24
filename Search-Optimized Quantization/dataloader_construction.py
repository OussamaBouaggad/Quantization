# -------------------------------------------------------------------------
# Calibration Dataloader Construction
# --------------------------------------------------------------------------

import torch
from datasets import load_dataset  # type: ignore[import]
from datasets.utils import logging as datasets_logging  # type: ignore[import]
from transformers import AutoTokenizer, AutoModelForSequenceClassification  # type: ignore[import]

datasets_logging.disable_progress_bar()
datasets_logging.set_verbosity_error()

def load_origin_model(model_path=None):
    if model_path is None:
        model_path = "microsoft/BiomedNLP-KRISSBERT-PubMed-UMLS-EL"
    
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return model

def create_calibration_dataloader(batch_size, calib_size, *args, **kwargs):
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-KRISSBERT-PubMed-UMLS-EL")

    def tokenization(example):
        tokenized_inputs = tokenizer(
            example["sentence1"], example["sentence2"], 
            padding="max_length", max_length=128, truncation=True
        )
        if "score" in example:
            tokenized_inputs["label"] = float(example["score"])
        return tokenized_inputs

    dataset = load_dataset("csv", data_files={"validation": "medsts_dev.csv"})["validation"]
    dataset = dataset.map(tokenization, batched=True)
    
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"] + 
                       (["token_type_ids"] if "token_type_ids" in dataset.column_names else []))

    calib_size = min(calib_size, len(dataset))
    return torch.utils.data.DataLoader(dataset.select(range(calib_size)), batch_size=batch_size, drop_last=True)



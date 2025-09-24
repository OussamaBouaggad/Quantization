# Search-Optimized Quantization in Biomedical Ontology Alignment

[![PyPI release](https://img.shields.io/pypi/v/olive-ai)](https://pypi.org/project/olive-ai/)
[![Documentation](https://img.shields.io/website/https/microsoft.github.io/Olive?down_color=red&down_message=offline&up_message=online)](https://microsoft.github.io/Olive/)

We introduce an orchestrated optimization pipeline that algorithmically engineers hardware-aware quantized models through Microsoft Olive and its ecosystem. The framework preserves accuracy while unlocking substantial gains in efficiency, scalability, and real-world deployment for biomedical ontology alignment. Given a model and target hardware, our approach selects and integrates the most effective optimization strategies, ensuring efficient inference across both cloud and edge environments.


## News

**\[23 Sep 2025]** The proposed optimization pipeline has been integrated into the [Microsoft Olive deep learning toolkit](https://github.com/microsoft/Olive), enabling streamlined quantization workflows for real-world applications.



## Abstract

In the fast-moving world of AI, as organizations and researchers develop more advanced models, they face challenges due to their sheer size and computational demands. Deploying such models on edge devices or in resource-constrained environments adds further challenges related to energy consumption, memory usage and latency. To address these challenges, emerging trends are shaping the future of efficient model optimization techniques. From this premise, by employing supervised state-of-the-art transformer-based models, this research introduces a systematic method for ontology alignment, grounded in cosine-based semantic similarity between a biomedical layman vocabulary and the Unified Medical Language System (UMLS) Metathesaurus. It leverages Microsoft Olive to search for target optimizations among different Execution Providers (EPs) using the ONNX Runtime backend, followed by an assembled process of dynamic quantization employing Intel Neural Compressor and IPEX (Intel Extension for PyTorch). Through our optimization process, we conduct extensive assessments on the two tasks from the DEFT 2020 Evaluation Campaign, achieving a new state-of-the-art in both. We retain performance metrics intact, while attaining an average inference speed-up of 20x and reducing memory usage by approximately 70%.


## Launcher

If you prefer using the command line directly instead of Jupyter notebooks, we've outlined the quickstart commands here.

### 1. Install Olive CLI
We recommend installing Olive in a [virtual environment](https://docs.python.org/3/library/venv.html) or a [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

```
pip install olive-ai[auto-opt]
pip install transformers onnxruntime-genai
```
> [!NOTE]
> Olive has optional dependencies that can be installed to enable additional features. Please refer to [Olive package config](./olive/olive_config.json) for the list of extras and their dependencies.

The components required to fully reproduce our optimization pipeline are listed in the provided `requirements.txt`:

```
pip install -r requirements.txt
```

### 2. Automatic Optimizer

You can optimize a Hugging Face model that provides multiple precision files, while Olive relies only on the one explicitly specified.

Run the automatic optimization:

```bash
olive optimize \
    --model_name_or_path BERT/model_name \
    --precision int8 \
    --output_path models/BERT
```

>[!TIP]
><details>
><summary>PowerShell Users</summary>
>Line continuation between Bash and PowerShell are not interchangeable. If you are using PowerShell, then you can copy-and-paste the following command that uses compatible line continuation.
>
>```powershell
>olive optimize `
>    --model_name_or_path BERT/model_name `
>    --output_path models/BERT `
>    --precision int8
>```
</details>

The automatic optimizer will:

1. Acquire the model from the Hugging Face model repository.
1. Quantize the model to `int8` using GPTQ.
1. Capture the ONNX Graph and store the weights in an ONNX data file.
1. Optimize the ONNX Graph.

Olive can even automatically optimize popular LLM architectures like Llama, Phi, Qwen, Gemma, etc., out of the box - [see detailed list here](https://huggingface.co/docs/optimum/en/exporters/onnx/overview). Other model architectures can also be optimized by providing details on the input and output definitions of the model (`io_config`).

### 3. Inference on the ONNX Runtime

The ONNX Runtime (ORT) is a fast and lightweight cross-platform inference engine with bindings for popular programming languages such as Python, C/C++, C#, Java, JavaScript, etc. ORT enables you to infuse AI models into your applications so that inference is handled on-device.

The sample chat app can be found in [model-chat.py](https://github.com/microsoft/onnxruntime-genai/blob/main/examples/python/model-chat.py) in the [onnxruntime-genai](https://github.com/microsoft/onnxruntime-genai/) GitHub repository.


## Citation

```bibtex
@article{Bouaggad2025,
author = {Bouaggad, Oussama and Grabar, Natalia},
year = {2025},
month = {07},
pages = {},
title = {{S}earch-{O}ptimized {Q}uantization in {B}iomedical {O}ntology {A}lignment},
doi = {10.48550/arXiv.2507.13742}
}
```


## Learn more

- [Documentation](https://microsoft.github.io/Olive)
- [Recipes](https://github.com/microsoft/olive-recipes)


## Contributions and Feedback

We welcome contributions! Please read the [contribution guidelines](./CONTRIBUTING.md) for more details on how to contribute to the Olive project.


## License

The project is licensed under the terms of the [MIT License](./LICENSE).

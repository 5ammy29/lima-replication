# LIMA Small-Scale Replication

A small-scale replication of the ideas presented in the **LIMA (Less Is More for Alignment)** paper using **Qwen2.5-0.5B-Instruct** and **LoRA-based Supervised Fine-Tuning (SFT)**.

The objective of this project is to understand the complete supervised fine-tuning pipeline by implementing, training, evaluating, and analysing every stage instead of treating the training process as a black box.

---

## Paper

This project is based on:

**LIMA: Less Is More for Alignment**

https://arxiv.org/abs/2305.11206

---

## Project Goal

The original LIMA paper investigates whether a pretrained language model can become more helpful through supervised fine-tuning on a small number of high-quality instruction-following examples.

This repository is **not** a reproduction of the original LIMA experiments. Instead, it is a learning-focused implementation that follows the same workflow on a much smaller scale while carefully inspecting each stage of the pipeline.

The project covers:

* Dataset preparation
* Chat template formatting
* Tokenization
* Supervised fine-tuning with LoRA
* Training analysis
* Response generation
* Manual pairwise evaluation
* Failure analysis

---

## Project Pipeline

* Inspect the original LIMA dataset
* Convert conversations into chat-style JSONL
* Prepare train and evaluation datasets
* Format conversations using the Qwen chat template
* Inspect tokenization and batching
* Understand the transformer forward pass
* Fine-tune Qwen2.5-0.5B-Instruct using LoRA
* Evaluate the base and fine-tuned models
* Analyse strengths and failure cases

---

## Repository Structure

```text
data/
├── train_data_900.jsonl
├── eval_data_130.jsonl

notebooks/
├── inspect_qwen_dataset.ipynb
├── inspect_tokenizer.ipynb
├── inspect_forward_pass.ipynb

scripts/
├── split_data.py
├── train.py
├── lora_qwen.py
├── evaluate_base.py
└── evaluate_sft_qwen.py

outputs/
├── evaluation/      # Final project report and evaluation results
├── graphs/          # Training and evaluation plots
└── sft_qwen/        # Training logs and trainer outputs
```

---

## Project Outputs

The complete outputs generated during this project are available under the **`outputs/`** directory.

### Project Report

The complete project report, including the evaluation methodology, response comparisons, manual human preference analysis, metrics, observations, and conclusions, can be found in:

```text
outputs/evaluation/
```

If you are visiting this repository to understand the results of the project, **please begin by reading the report available in `outputs/evaluation/`**.

This directory contains:

* Project evaluation report
* Manual human preference evaluation
* Base model responses
* Fine-tuned model responses
* Pairwise comparison results
* Evaluation metrics
* Supporting evaluation files

---

### Training Outputs

Training logs and trainer outputs are available in:

```text
outputs/sft_qwen/
```

This directory contains:

* Training logs
* Trainer state
* Run summary
* Final training statistics

---

### Training Graphs

Training visualizations are available in:

```text
outputs/graphs/
```

These include:

* Training loss
* Evaluation loss
* Gradient norm
* Learning rate schedule

---

## Dataset Format

Training data is stored in JSONL format.

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Question here"
    },
    {
      "role": "assistant",
      "content": "Answer here"
    }
  ]
}
```

Each conversation is converted into the Qwen chat template before tokenization and supervised fine-tuning.

---

## Project Status

* [x] Dataset inspection
* [x] Dataset preparation
* [x] Chat template inspection
* [x] Tokenizer inspection
* [x] Forward pass inspection
* [x] LoRA-based supervised fine-tuning
* [x] Training analysis
* [x] Base model evaluation
* [x] Fine-tuned model evaluation
* [x] Manual response comparison
* [x] Failure analysis

---

## Summary

This project demonstrates an end-to-end implementation of a supervised fine-tuning pipeline for a small language model using LoRA.

The evaluation compares the fine-tuned model against the original instruction-tuned base model using held-out prompts and manual pairwise preference analysis. The results show that the fine-tuned model improves some creative and conversational responses, while the base model remains stronger on several factual, technical, and safety-sensitive tasks.

The primary objective of this project is to understand and analyse the complete supervised fine-tuning workflow rather than to reproduce the exact results reported in the original LIMA paper.

---

## Future Work

Potential extensions of this project include:

* Comparing LoRA, QLoRA, and DoRA on larger language models.
* Expanding evaluation using automated LLM-as-a-Judge and benchmark-based metrics.
* Investigating training configurations that improve response quality while preserving factual accuracy and instruction following.
* Evaluating larger instruction-tuned models using the same experimental pipeline.

---

## References

* LIMA Paper — https://arxiv.org/abs/2305.11206
* Hugging Face Transformers — https://github.com/huggingface/transformers
* PEFT — https://github.com/huggingface/peft
* Qwen2.5 Models — https://huggingface.co/Qwen

# LIMA Small-Scale Replication

A small-scale replication of the ideas presented in the LIMA paper using a lightweight open-source language model.

The goal of this project is to understand the complete supervised fine-tuning pipeline by implementing and inspecting each stage instead of treating the training process as a black box.

---

## Paper

This project is based on:

**LIMA: Less Is More for Alignment**

https://arxiv.org/abs/2305.11206

---

## Goal

The original LIMA paper studies whether a pretrained language model can become more helpful through supervised fine-tuning on a small number of high-quality instruction-following examples.

This repository is not a reproduction of the original experiments. Instead, it is a learning-focused implementation that follows the same pipeline on a much smaller scale while examining each stage in detail.

---

## Current Scope

- Inspect the original LIMA dataset
- Convert conversations into chat-style JSONL
- Prepare train and test datasets
- Format conversations using the Qwen chat template
- Inspect tokenization and batching
- Understand the transformer forward pass
- Build the supervised fine-tuning pipeline

---

## Repository Structure

```
data/
├── train_data.jsonl
└── test_data.jsonl

notebooks/
├── inspect_qwen_dataset.ipynb
├── inspect_tokenizer.ipynb
├── inspect_forward_pass.ipynb

scripts/
└── prepare_data.py
```

---

## Dataset Format

The processed dataset is stored as JSONL.

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

After formatting with the Qwen chat template, each training example becomes a single text sequence used for supervised fine-tuning.

---

## Progress

- [x] Dataset inspection
- [x] Data preparation
- [x] Chat template inspection
- [x] Tokenizer inspection
- [ ] Forward pass inspection
- [x] Loss inspection
- [ ] Supervised fine-tuning
- [ ] Evaluation

---

## Planned Work

- Complete loss inspection
- Train a Qwen 0.5B model using supervised fine-tuning
- Compare responses before and after fine-tuning
- Evaluate instruction-following behaviour

---

## References

- LIMA Paper — https://arxiv.org/abs/2305.11206
- Hugging Face Transformers — https://github.com/huggingface/transformers
- Qwen2.5 Models — https://huggingface.co/Qwen
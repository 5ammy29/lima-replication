# LIMA Small-Scale Replication

A small experiment to understand the idea behind LIMA using a tiny model and supervised fine-tuning.

## Paper

This project is inspired by:

**LIMA: Less Is More for Alignment**
Paper: https://arxiv.org/abs/2305.11206

## Goal

The original LIMA paper studies whether a pretrained language model can become more helpful through supervised fine-tuning on a small number of high-quality examples.

This repo is not a full reproduction of the paper. It is a small learning-focused replication that prepares the LIMA data and builds toward training a tiny model with SFT.

## Current Scope

* Inspect the LIMA dataset
* Convert raw conversations into chat-style JSONL format
* Prepare train and test data for supervised fine-tuning
* Later: train a small SFT model and compare it with the base model

## Dataset Format

The cleaned data is stored as JSONL files with chat-style messages:

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
  ],
  "source": "stackexchange",
  "num_turns": 2
}
```

## Status

Currently completed:

- Dataset inspection
- Data preparation script
- Clean train/test JSONL files
- Qwen tokenizer inspection
- Chat template analysis
- Qwen-formatted SFT training dataset

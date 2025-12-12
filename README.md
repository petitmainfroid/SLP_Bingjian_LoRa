# LoRA Reproduction and Analysis on Small Language Models

## Overview

This repository contains the code and experimental results for a **critical reproduction study of LoRA (Low-Rank Adaptation)** on a small causal language model (~20M parameters).

The goal of this project is **not** to maximize benchmark performance, but to investigate the **limitations and assumptions of LoRA** when applied outside the large, over-parameterized regime considered in the original paper.

In particular, we study whether LoRA remains effective for **instruction-following behavior** under severe model capacity and compute constraints, and compare it against **full fine-tuning (FFT)**.

---

## Motivation

LoRA has been widely adopted as a parameter-efficient fine-tuning method for large language models. However, its effectiveness relies on an implicit assumption:

> **Task-specific parameter updates lie in a low-dimensional subspace of an already over-parameterized model.**

This project aims to empirically test this assumption by applying LoRA to a much smaller model, where over-parameterization no longer holds.

---

## Experimental Setup

### Base Model
- Causal language model (~0.02B parameters)

### Training Stages
- Pretraining from scratch (public codebase)
- LoRA-based fine-tuning
- Full fine-tuning (FFT) as a baseline

### Hardware
- Single GPU with limited compute budget

### Datasets
- General-domain text data (pretraining and fine-tuning)
- Medical-domain QA dataset (domain-specific LoRA experiment)

---

## Evaluation Strategy

Due to the extremely limited capability of the base model, standard automatic benchmarks were not informative in early stages.

Instead, we rely on **qualitative probing via manual prompting**, which is sufficient to clearly distinguish:

- Instruction understanding vs. failure
- Coherent responses vs. degenerate behaviors (repetition, blank outputs, off-topic answers)

Example prompts and model outputs (with English translations) are provided in the repository for transparency.

---

## Key Experiments

### 1. Initial LoRA Fine-tuning

- Batch size = 32  
- Rank = 8  
- Epoch = 1  

After fixing data loading bottlenecks (`num_workers`), LoRA reduced blank outputs compared to pretraining, but instruction-following remained poor.

---

### 2. Increasing Training Epochs

- Epoch = 10, 50  

**Result:** no meaningful improvement. The model continued to produce off-topic or incoherent responses.

---

### 3. Effect of LoRA Rank

- Tested higher rank: `r = 16`

**Hypothesis:** the small base model may require a higher-dimensional adaptation space.  
**Result:** hypothesis not supported. Increasing rank did not resolve instruction-following failures.

---

### 4. Effect of Batch Size

Motivated by prior work suggesting smaller batch sizes may lead to flatter minima:

- Reduced batch size during LoRA fine-tuning

**Result:** no performance improvement, while GPU utilization dropped significantly.

---

### 5. Domain-Specific Fine-tuning

**Hypothesis:** restricting the fine-tuning data to a single domain might reduce task complexity.

- LoRA fine-tuning on a medical QA dataset

**Result:** model behavior further deteriorated, producing less coherent outputs.

---

### 6. Full Fine-Tuning (FFT) Comparison

Using the same general-domain dataset:

- FFT converged in **2 epochs (~3 hours)**
- LoRA required **~8 hours for 10 epochs**

**Unexpected result:**

> **FFT consistently outperformed LoRA in instruction-following behavior**, producing coherent and relevant responses without repetition.

---

## Main Findings

- LoRA did not demonstrate clear advantages in a severely under-parameterized setting.
- Increasing rank, epochs, or changing batch size did not resolve LoRA failures.
- Full fine-tuning was more effective and more stable for instruction alignment.
- Parameter-efficient fine-tuning does not necessarily imply better performance or faster convergence.

---

## Discussion

These results suggest that **LoRA critically depends on the over-parameterization regime of the base model**.

When the model capacity is severely limited, restricting updates to a low-rank subspace may overly constrain learning, whereas full parameter updates remain necessary to align behavior.

---

## Repository Structure
├── trainer/
│ ├── train_pretrain.py # Pretraining from scratch
│ ├── train_lora.py # LoRA fine-tuning
│ └── train_full_sft.py # Full fine-tuning (FFT)
├── evaluation/
│ ├── qualitative_prompts.md # Manual qualitative evaluation prompts
│ └── model_outputs/ # Saved model outputs for inspection
├── model/
│ ├── init.py # Module initialization
│ ├── model_minimind.py # Base causal language model definition
│ ├── model_lora.py # LoRA adapter implementation
│ ├── tokenizer.json # Tokenizer vocabulary
│ ├── tokenizer_config.json # Tokenizer configuration
│ └── pycache/ # Python cache files
├── eval_llm.py # Script for qualitative LLM evaluation
└── README.md

---

## Notes on Reproducibility

- All hyperparameters and configurations used in the experiments are documented.
- Qualitative evaluation examples are provided with translations.
- Due to compute limitations, large-scale sweeps were not feasible.

---

## Reference

Hu et al.,  
**LoRA: Low-Rank Adaptation of Large Language Models**,  
arXiv:2212.10560, 2022.

---

## Final Remark

This repository represents an **honest and critical reproduction study**, emphasizing **failure analysis and methodological limitations** rather than selective success cases.


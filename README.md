# Phase 2 — Node-Level Field Detection

This repository contains **Phase 2 (implementation)** of the project:  
high-accuracy event field extraction from heterogeneous websites using **node-level machine learning models**.

The goal of this phase is to validate a **robust, reproducible ML pipeline** that generalizes across unseen websites.

---

## Scope (fixed design decisions)

The following decisions are **intentionally fixed and validated**:

- Node-level field detection only (each DOM node is classified independently)
- One binary classifier per field:
  - Date
  - Time
  - Location
  - Name (optional, depending on data sufficiency)
- LightGBM as the baseline model
- Leave-One-Site-Out (LOSO) cross-validation
- Strong class imbalance handled via:
  - Negative undersampling
  - `scale_pos_weight`
- No event reconstruction
- No ranking loss
- No NER
- No deep learning
- Evaluation via **Precision@K** and **Recall@K**

---

## Current Status

### ✅ Implemented
- **Date classifier (end-to-end)**:
  - Data loading and merging (16 sites)
  - Label normalization (composite → atomic `is_Date`)
  - Leakage-safe feature matrix construction
  - LOSO cross-validation (15 folds)
  - LightGBM training with class balancing
  - Evaluation with Precision@K / Recall@K
  - Feature importance analysis across folds

### 🔄 In Progress / Next
- Clone the validated pipeline for:
  - Time
  - Location
- Comparative analysis across fields
- Final Phase 2 write-up

---

## Dataset Overview

- 16 CSV files, each corresponding to a different website
- Each row represents a single DOM node
- ~3,030 nodes total
- ~35 engineered features per node:
  - DOM structure (depth, parent_tag, sibling index, etc.)
  - Text statistics (length, digit ratio, whitespace ratio, etc.)
  - Pattern-based indicators (e.g., `contains_date`)
- Labels include atomic and composite types (e.g., `Date`, `DateTime`, `StartEndTime`)
  - Composite labels are normalized into atomic binary targets

---

## Evaluation Notes (Important)

- A true page identifier is not available in the dataset.
- Evaluation currently uses a **coarse page proxy** (`text_context`), which can group many nodes into large “pages”.
- As a result:
  - Recall@K is often high (many pages contain a single true instance)
  - Precision@K is structurally low due to large candidate sets
- This is a **known evaluation limitation**, not a model failure, and is documented as future work.

---

## How to Run (Date Classifier)

```bash
# Set PYTHONPATH
export PYTHONPATH=src

# Sanity check data
python -m dateclf.cli data-check --config config.yaml

# Feature matrix validation
python -m dateclf.cli feature-check --config config.yaml

# LOSO folds check
python -m dateclf.cli fold-check --config config.yaml

# Train and evaluate Date classifier
python -m dateclf.cli train-date --config config.yaml


## How to Run (V2) 

Training Runs (Field-by-Field)
Train one field
Time :
python -m dateclf.cli --config config.yaml train-field --target is_Time
Location :
python -m dateclf.cli --config config.yaml train-field --target is_Location
Date :
python -m dateclf.cli --config config.yaml train-field --target is_Date
Train all fields sequentially :
python -m dateclf.cli --config config.yaml train-all

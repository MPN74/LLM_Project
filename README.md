# Interpretable and Topic Classification of Persian and English Poetry Using LLMs

This repository contains the implementation, datasets, and experiments for the project:

**Interpretable and Topic Classification of Persian and English Poetry Using Large Language Models**  
*Authors: Mahdi Nasiri, Mohammad Amin Nosrati*

---

## 📌 Project Overview
Large Language Models (LLMs) have shown impressive progress in NLP, but interpreting their behavior in **low-resource languages** like Persian remains challenging—especially in poetry, where metaphor, cultural context, and emotional richness complicate sentiment recognition.

This project investigates:
- **Bilingual dataset (Persian + English poetry)**
- **Layer-wise analysis** of emotions using the Logit Lens
- **Activation Patching** to identify critical layers
- **Steering Vectors** for representation-level interventions
- **Comparison of fine-tuning strategies** (full vs. selective layer tuning)

---

## 🗂️ Dataset
- **Persian poetry**: ~400 verses from 14 classical poets (Hafez, Saadi, Khayyam, Parvin, etc.)  
  Labeled into: *Love, Philosophy, Spirituality, Death*.  
  Preprocessing with [Hazm](https://github.com/sobhe/hazm) and Parsivar.

- **English poetry**: [Poetry Foundation Dataset](https://www.kaggle.com/datasets/tgdivy/poetry-foundation-poems) (~13k poems).  
  Tags remapped to the same 4 categories.

- Length capped at **≤ 340 words** to avoid GPU memory issues.  
- Final dataset size: **4,771 poems** (balanced across labels).

---

## 🔍 Methods
1. **Logit Lens Analysis**  
   - Layer-wise tracking of logits per label.  
   - Cross-lingual comparison between Persian and English.

2. **Activation Patching**  
   - Injecting English activations into Persian representations.  
   - Identifies **critical layers (15–17)** for cultural/emotional signals.

3. **Steering Vectors**  
   - Constructed from cross-lingual activation differences.  
   - Injected into model to improve classification accuracy.

4. **Fine-tuning**  
   - **Full model LoRA fine-tuning** (4 epochs).  
   - **Selective tuning** (layers 15–17).  
   - Comparison with and without steering vectors.

---

## 📊 Results
- **English baseline**: ~51% accuracy (4-class).  
- **Persian baseline (zero-shot)**: ~63% accuracy.  
- **Full fine-tuning (LoRA)**: ~85% (train), ~67% (val).  
- **Selective tuning (layers 15–17)**: ~70% (val).  
- **Fine-tuning + Steering Vectors**: boosted logits & accuracy.  

Consistent improvements were seen in *Love* and *Death*, while *Philosophy* and *Spirituality* remained challenging.

---

## ⚙️ Requirements
- Python 3.10+
- [PyTorch](https://pytorch.org/)
- [Transformers](https://github.com/huggingface/transformers)
- [TRL](https://github.com/huggingface/trl)
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes)
- [Hazm](https://github.com/sobhe/hazm), Parsivar
- GPU with ≥ 16GB VRAM (recommended)

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage
### Preprocessing
```bash
python preprocess.py --dataset persian
python preprocess.py --dataset english
```

### Logit Lens & Activation Patching
```bash
python analysis/logit_lens.py
python analysis/activation_patching.py
```

### Fine-tuning
```bash
python train.py --mode full     # Full fine-tuning
python train.py --mode partial  # Layers 15–17 only
```

### Steering Vectors
```bash
python steering/apply_vectors.py
```

---

## 📈 Visualizations
- Layer-wise logit curves (Persian vs. English).  
- Delta logits after activation patching.  
- Steering vector effect plots.  

Plots are available in the `figures/` directory.

---

## 👥 Contributions
- **Mohammad Amin Nosrati**: dataset collection, labeling, model selection, fine-tuning, report writing.  
- **Mahdi Nasiri**: logit lens, activation patching, steering vectors, evaluation, report writing.

---

## 📌 Citation
If you use this work, please cite:
```
Nasiri, M., Nosrati, M.A. (2025). Interpretable and Topic Classification of Persian and English Poetry Using LLMs.
```

---

## 🔮 Future Work
- Apply to **larger models** (GPT, LLaMA 4, Qwen 3).  
- Expand **Persian dataset** with better coverage.  
- Investigate **longer context windows** & metaphor handling.  
- Refine steering vectors for cultural alignment.

---

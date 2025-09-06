# Interpretable and Topic Classification of Persian and English Poetry Using LLMs

This repository presents a project on **interpretable classification of Persian and English poetry** using Large Language Models (LLMs).  
The focus is on analyzing how multilingual models capture **cultural and emotional cues** in Persian poetry and comparing them with English poems.

---

##  Problem Statement
Understanding poetry in low-resource languages like Persian is challenging for LLMs due to:
- Complex metaphors and cultural references
- Emotional richness
- Limited labeled datasets

The project aims to:
- Identify which layers of an LLM capture Persian cultural/emotional signals
- Improve performance with **lightweight interpretability methods** such as **activation patching** and **steering vectors**

---

##  Dataset
- **Persian poetry**: ~400 verses from 14 classical poets (e.g., Hafez, Saadi, Khayyam, Parvin).  
  Manually labeled into 4 themes: *Love, Philosophy, Spirituality, Death*.  
  Preprocessed with Hazm & Parsivar libraries.

- **English poetry**: ~13k poems from the [Poetry Foundation dataset](https://www.kaggle.com/datasets/tgdivy/poetry-foundation-poems).  
  Tags remapped to the same 4 categories.  
  Poem length capped at **≤ 340 words** to fit GPU limits.

- Final balanced dataset: **4,771 poems**.

---

##  Methods
1. **Logit Lens Analysis**  
   - Layer-wise extraction of logits per label  
   - Comparison between Persian and English

2. **Activation Patching**  
   - Injecting English activations into Persian poems  
   - Identified **critical layers (15–17)** for cultural/linguistic representation

3. **Steering Vectors**  
   - Constructed from cross-lingual activation differences  
   - Added to hidden states of critical layers to guide classification

4. **Fine-tuning**  
   - **Full model LoRA fine-tuning** (4 epochs)  
   - **Selective tuning** (layers 15–17 only)  
   - Comparison of results with/without steering vectors

---

##  Results
- **English baseline**: ~51% accuracy  
- **Persian zero-shot baseline**: ~63% accuracy  
- **Full LoRA fine-tuning**: 85% (train) / 67% (validation)  
- **Selective fine-tuning (layers 15–17)**: ~70% validation accuracy  
- **Steering vectors**: Improved logits & classification, especially for *Love* and *Death*  

Challenges: *Philosophy* and *Spirituality* remain harder to classify.

---


---



---

##  Key Insights
- Critical mid-layers (15–17) encode Persian-specific cultural/emotional features.  
- Steering vectors enhance interpretability and efficiency.  
- Parameter-efficient fine-tuning is competitive with full fine-tuning.  

---

##  Contributions
- **Mohammad Amin Nosrati**: dataset collection & labeling, model selection, fine-tuning, reporting  
- **Mahdi Nasiri**: logit lens, activation patching, steering vectors, evaluation, reporting  

---

##  Future Work
- Scale to larger LLMs (GPT, LLaMA 4, Qwen 3)  
- Expand Persian dataset with richer themes  
- Explore longer context windows for poetry  
- Improve steering vector design for cultural alignment  

---

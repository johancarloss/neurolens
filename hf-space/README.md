---
title: NeuroLens
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.14.0
app_file: app.py
pinned: false
license: mit
---

# 🧠 NeuroLens — Brain Tumor MRI Classifier with Explainability

Interactive companion to a study comparing **VGG16** and **ResNet50** on brain-tumor
MRI classification, each explained by **three XAI techniques** (Grad-CAM · LIME ·
SHAP) shown side by side against the original scan.

Pick a **curated example** to see a finding — e.g. a glioma both models miss and
where they (wrongly) look — or **upload your own** MRI.

- **Code, methodology & write-ups:** https://github.com/johancarloss/neurolens
- Curated examples load **instantly** (pre-computed). Free uploads run **live on
  CPU** (**~4 minutes**; LIME and SHAP dominate).

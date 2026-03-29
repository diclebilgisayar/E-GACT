<h1 align="center">
  🚀 Resource-Efficient Graph-Aware Contrastive Transformer <br>
  (E-GACT)
</h1>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Paper](https://img.shields.io/badge/Paper-IEEE_JBHI_Submission-green.svg)](#)

> **Official Code Repository for the paper:**  
> *"Resource-Efficient Graph-Aware Contrastive Transformer (E-GACT) for Early Diabetes Risk Prediction: Bridging Algorithmic Topology and Clinical Explainability"* (Submitted to IEEE Journal of Biomedical and Health Informatics - JBHI).

---

## 📖 Overview

Predicting Type 2 Diabetes Mellitus (T2DM) from tabular Electronic Health Records (EHR) is critical for early clinical intervention. However, current Deep Tabular Models (e.g., TabNet, FT-Transformer) either overfit on small clinical cohorts or fail to scale on massive population-level datasets due to $\mathcal{O}(N^2)$ attention complexities. Furthermore, they evaluate patients as Independent and Identically Distributed (I.I.D.) instances, completely ignoring the fundamental clinical practice of **Case-Based Reasoning**.

**E-GACT** solves these bottlenecks mathematically by integrating:
1. **Lightweight Tabular Transformer:** For non-linear, intra-patient feature projection.
2. **FAISS $k$-NN Graph Neural Network (GNN):** For inter-patient topological similarities (Case-Based Reasoning) dynamically constructed in $\mathcal{O}(N \log N)$ time.
3. **Supervised Contrastive Learning (SCL):** For robust representation regularization against severe class imbalances.

🔥 **Edge AI Ready:** With only **0.45M learnable parameters**, E-GACT operates with $<45$ ms inference latency on standard clinical microprocessors, making it highly suitable for zero-latency, privacy-preserving local Edge AI deployments.

---
## 🏗️ Architecture

The proposed architecture.

<p align="center">
  <img src="E-GACT Architecture.jpg" width="95%">
  <br><em>Fig 1: Overall workflow of the E-GACT architecture.</em>
</p>

---

## 📊 Benchmarked Datasets
To prove algorithmic robustness and scalability across varying modalities, E-GACT is evaluated on three globally validated, open-access cohorts:

| Dataset | Modality | Size (Patients) | Focus Area | Target Prediction |
| :--- | :--- | :--- | :--- | :--- |
| **NHANES (2017-2018)** | Clinical Lab + Demographics | ~10,000 | Physiological Signals | T2DM (HbA1c $\geq$ 6.5) |
| **130-US Hospitals** | Electronic Health Records (EHR) | ~101,000 | Case-Based Reasoning | Readmission Risk |
| **CDC BRFSS (2015)** | Population Survey | ~50,000* | Edge AI Scalability | T2DM (Imbalanced) |

*\*Note: BRFSS is sub-sampled to 50k to ensure stable execution within standard free-tier Cloud environments (e.g., Google Colab 12GB RAM) without Out-Of-Memory crashes.*

---

## ⚡ Zero-Click Reproducibility (For Peer-Reviewers)

We deeply respect the time of academic reviewers. We have designed a **"Zero-Click" Universal Data Pipeline**. 
- ❌ No Google Drive mounting required.
- ❌ No Kaggle API keys or passwords required.
- ❌ No manual file downloads required.

**How to Test the Code:**
1. Open our official interactive Notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/E-GACT/blob/main/E_GACT_Reproducibility_Demo.ipynb)
2. Click **`Runtime -> Run All`**.
3. The code will automatically fetch the raw `.XPT` and `.csv` files from public academic mirrors, strictly perform inductive leakage-free graph constructions, train the E-GACT architecture, and output the ROC-AUC benchmarks shown in **Table 1** of our paper.

*(Note to Reviewers: Please ensure Hardware Accelerator is set to **T4 GPU** in Colab).*

---

## 🛠️ Local Installation & Usage

If you wish to clone and run this repository on your local workstation or Edge device:

### 1. Requirements
```bash
pip install -r requirements.txt

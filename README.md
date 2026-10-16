# 💳 Credit Risk Modeling & Explainable AI (XAI)

An end-to-end Machine Learning pipeline designed to evaluate borrower creditworthiness and predict loan default risk using LendingClub-style underwriting features. 

The pipeline specifically addresses **severe class imbalance** using **SMOTE** (Synthetic Minority Over-sampling Technique) strictly isolated to the training split, trains a tree-based ensemble classifier, and utilizes **SHAP (SHapley Additive exPlanations)** for macro-level risk driver attribution and individual applicant-level adverse action explanations.

---

## 📌 Problem Overview & Objectives

In retail and peer-to-peer lending, default events typically represent a small fraction of overall loans (often 10%–20%). Standard classifiers trained on uncalibrated datasets develop a severe bias toward the majority class (fully paid loans), resulting in high baseline accuracy but dangerously low sensitivity (recall) on delinquent accounts.

**Core Goals:**
1. **Accurate Default Detection:** Maximize out-of-sample ROC-AUC and recall for high-risk borrowers.
2. **Leakage-Free Resampling:** Balance minority default samples using SMOTE without allowing synthetic observations to bleed into evaluation data.
3. **Regulatory Explainability (XAI):** Ensure full compliance with Equal Credit Opportunity Act (ECOA) and Fair Credit Reporting Act (FCRA) requirements by providing transparent, mathematical justifications for adverse credit decisions via SHAP.

---

## 🏗️ Architecture & Pipeline Flow

```text
Raw Lending Data (5,000 Applicants)
       │
       ▼
Feature Preprocessing (One-Hot Encoding, Stratified Train/Test Split: 80/20)
       │
       ├──► Test Partition (Untouched, Real-World 85:15 Ratio)
       │
       ▼
SMOTE Resampling (Fitted strictly on Training Partition -> 50:50 Balance)
       │
       ▼
Random Forest Classifier (Ensemble Training with balanced_subsample weights)
       │
       ├──► Evaluation: ROC-AUC, Classification Report, Confusion Matrix
       │
       ▼
SHAP TreeExplainer
       ├──► Global Feature Impact (Beeswarm Plot)
       └──► Single Applicant Decision Breakdown (Waterfall Plot)

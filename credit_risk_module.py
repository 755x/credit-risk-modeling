"""
Credit Risk Modeling Module
Evaluates applicant creditworthiness, addresses class imbalance via SMOTE,
and explains predictions using SHAP (SHapley Additive exPlanations).
"""

from typing import Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/script rendering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class LendingDataGenerator:
    """Generates synthetic lending data mimicking LendingClub applicant profiles."""

    @staticmethod
    def generate(
        n_samples: int = 5000,
        default_rate: float = 0.15,
        seed: int = 42,
    ) -> pd.DataFrame:
        np.random.seed(seed)

        # Baseline features
        loan_amnt = np.random.choice([5000, 10000, 15000, 20000, 25000, 35000], size=n_samples)
        annual_inc = np.random.lognormal(mean=11.0, sigma=0.45, size=n_samples).round(-2)
        fico_score = np.random.normal(loc=700, scale=45, size=n_samples).clip(580, 850).astype(int)
        dti = np.random.uniform(5.0, 40.0, size=n_samples).round(2)
        int_rate = (28.0 - (fico_score - 580) * 0.06 + np.random.normal(0, 1.5, n_samples)).clip(5.0, 29.9).round(2)
        revol_util = np.random.uniform(5.0, 95.0, size=n_samples).round(1)
        inq_last_6mths = np.random.poisson(lam=0.8, size=n_samples).clip(0, 6)
        home_ownership = np.random.choice(["MORTGAGE", "RENT", "OWN"], size=n_samples, p=[0.50, 0.40, 0.10])

        # Latent log-odds of default based on underwriting risk factors
        z = (
            - 0.015 * (fico_score - 600)
            + 0.08 * dti
            + 0.12 * int_rate
            + 0.25 * inq_last_6mths
            + 0.015 * revol_util
            - 0.00002 * annual_inc
            + (home_ownership == "RENT") * 0.35
            - 2.8
        )
        prob_default = 1 / (1 + np.exp(-z))

        # Calibrate default rate to requested imbalance ratio
        threshold = np.quantile(prob_default, 1.0 - default_rate)
        loan_status = (prob_default >= threshold).astype(int)  # 1 = Default (High Risk), 0 = Fully Paid

        df = pd.DataFrame(
            {
                "loan_amnt": loan_amnt,
                "annual_inc": annual_inc,
                "fico_score": fico_score,
                "dti": dti,
                "int_rate": int_rate,
                "revol_util": revol_util,
                "inq_last_6mths": inq_last_6mths,
                "home_ownership": home_ownership,
                "loan_status": loan_status,
            }
        )
        return df


class CreditRiskPreprocessor:
    """Preprocesses categorical variables, handles splits, and balances training data via SMOTE."""

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []

    def prepare(
        self, df: pd.DataFrame, target_col: str = "loan_status"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        # Encode categorical variables
        df_encoded = pd.get_dummies(df, columns=["home_ownership"], drop_first=True)
        X = df_encoded.drop(columns=[target_col])
        y = df_encoded[target_col]
        self.feature_names = list(X.columns)

        # Stratified train/test split to preserve natural default distribution
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, stratify=y, random_state=self.random_state
        )

        return X_train, X_test, y_train, y_test

    def balance_with_smote(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Applies SMOTE strictly to the training set to prevent synthetic data leakage."""
        smote = SMOTE(random_state=self.random_state)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        return pd.DataFrame(X_resampled, columns=self.feature_names), y_resampled


class CreditRiskModel:
    """Ensemble model for credit scoring."""

    def __init__(self, random_state: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=150,
            max_depth=7,
            min_samples_split=10,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "CreditRiskModel":
        self.model.fit(X_train, y_train)
        return self

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series, output_img: str = "confusion_matrix.png") -> Dict[str, float]:
        preds = self.model.predict(X_test)
        probs = self.model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, probs)
        print("\n--- Model Evaluation (Out-of-Sample Test) ---")
        print(f"ROC-AUC Score: {auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, preds, target_names=["Non-Default (0)", "Default (1)"]))

        # Confusion Matrix
        cm = confusion_matrix(y_test, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fully Paid", "Default"])
        fig, ax = plt.subplots(figsize=(6, 5), dpi=100)
        disp.plot(cmap="Blues", ax=ax, values_format="d")
        plt.title("Credit Risk Confusion Matrix", weight="bold")
        plt.tight_layout()
        plt.savefig(output_img, dpi=300)
        plt.close(fig)
        print(f"Confusion Matrix saved to: {output_img}")

        return {"ROC_AUC": round(auc, 4)}


class CreditExplainability:
    """Explains model decisions globally and locally using Tree SHAP."""

    def __init__(self, model: RandomForestClassifier, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(model)

    def explain(
        self,
        X_test: pd.DataFrame,
        summary_plot_path: str = "shap_summary.png",
        applicant_plot_path: str = "shap_single_applicant.png",
    ) -> None:
        print("\nCalculating SHAP values...")
        shap_values = self.explainer(X_test)

        # For binary classification with RandomForest, extract SHAP values for class 1 (Default risk)
        if len(shap_values.shape) == 3 and shap_values.shape[2] == 2:
            shap_vals_default = shap_values[:, :, 1]
        else:
            shap_vals_default = shap_values

        # 1. Global Feature Importance (Beeswarm Plot)
        plt.figure(figsize=(10, 6))
        shap.plots.beeswarm(shap_vals_default, max_display=10, show=False)
        plt.title("SHAP Global Feature Importance (Impact on Default Risk)", weight="bold", pad=12)
        plt.tight_layout()
        plt.savefig(summary_plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Global SHAP summary saved to: {summary_plot_path}")

        # 2. Local Applicant Decision Explanation (Waterfall Plot for Applicant #0)
        plt.figure(figsize=(9, 5))
        shap.plots.waterfall(shap_vals_default[0], max_display=8, show=False)
        plt.title("Individual Risk Factor Breakdown (Applicant #0)", weight="bold", pad=12)
        plt.tight_layout()
        plt.savefig(applicant_plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Individual applicant SHAP breakdown saved to: {applicant_plot_path}")


# ==========================================
# Execution Pipeline
# ==========================================
if __name__ == "__main__":
    print("1. Generating synthetic LendingClub dataset (5,000 applicants, 15% default rate)...")
    data_gen = LendingDataGenerator()
    df = data_gen.generate(n_samples=5000, default_rate=0.15)
    print(f"   Class balance: {df['loan_status'].value_counts(normalize=True).to_dict()}")

    print("\n2. Preprocessing & Splitting data...")
    preprocessor = CreditRiskPreprocessor(test_size=0.2)
    X_train, X_test, y_train, y_test = preprocessor.prepare(df)
    print(f"   Original Training Set: {len(y_train)} samples (Defaults: {sum(y_train)})")

    print("\n3. Balancing training data using SMOTE...")
    X_train_res, y_train_res = preprocessor.balance_with_smote(X_train, y_train)
    print(f"   Resampled Training Set: {len(y_train_res)} samples (Defaults: {sum(y_train_res)})")

    print("\n4. Training Ensemble Credit Risk Model (Random Forest)...")
    risk_model = CreditRiskModel()
    risk_model.fit(X_train_res, y_train_res)

    print("\n5. Evaluating model performance on untouched test partition...")
    risk_model.evaluate(X_test, y_test, output_img="credit_risk_confusion_matrix.png")

    print("\n6. Generating SHAP Explainability Reports...")
    explainer = CreditExplainability(risk_model.model, preprocessor.feature_names)
    explainer.explain(
        X_test,
        summary_plot_path="shap_global_summary.png",
        applicant_plot_path="shap_single_applicant.png",
    )

    print("\nPipeline finished successfully.")
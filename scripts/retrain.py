"""
scripts/retrain.py - Model retraining pipeline.
Implements time-series validation, feature engineering integration,
and production-ready model persistence for the betting engine.
"""
import os
import sys
import joblib
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, classification_report
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

# Add project root to path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from prediction.features import FeatureEngineer
from config import BASE_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(BASE_DIR / "logs" / "retrain.log")
    ]
)
logger = logging.getLogger(__name__)

# Paths
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ==================== DATA LOADER (DUMMY + REAL SUPPORT) ====================
def load_training_data(use_dummy: bool = True) -> pd.DataFrame:
    """
    Load historical match data for retraining.
    Replace with your CSV/API loader in production.
    """
    if use_dummy:
        logger.info("⚠️ Using DUMMY DATA for pipeline testing. Replace with real CSV/API in production.")
        dates = pd.date_range(end=datetime.now(), periods=800, freq='3D')
        teams = ["Team A", "Team B", "Team C", "Team D", "Team E", "Team F"]
        data = []
        for i in range(800):
            h, a = np.random.choice(teams, 2, replace=False)
            fthg = np.random.poisson(1.2)
            ftag = np.random.poisson(1.1)
            ftr = "H" if fthg > ftag else ("A" if ftag > fthg else "D")
            data.append({
                "Date": dates[i], "HomeTeam": h, "AwayTeam": a,
                "FTHG": fthg, "FTAG": ftag, "FTR": ftr,
                "HS": np.random.poisson(10), "HST": np.random.poisson(4),
                "AS": np.random.poisson(10), "AST": np.random.poisson(4),
                "HC": np.random.poisson(5), "AC": np.random.poisnum(5),
                "HF": np.random.poisson(10), "AF": np.random.poisson(10),
                "B365H": np.round(1.5 + np.random.rand(), 2),
                "B365D": np.round(3.0 + np.random.rand(), 2),
                "B365A": np.round(4.0 + np.random.rand(), 2)
            })
        return pd.DataFrame(data)
    else:
        # Example: df = pd.read_csv("data/epl_history.csv", parse_dates=["Date"])
        # return df
        raise NotImplementedError("Real data loader not implemented. Update load_training_data().")

# ==================== FEATURE & TARGET PREP ====================
def prepare_features_and_target(df: pd.DataFrame):
    """
    Apply feature engineering pipeline, map targets, and remove leakage columns.
    """
    logger.info("🔧 Applying feature engineering pipeline...")
    engine = FeatureEngineer(elo_k=32, home_adv=65)
    featured_df = engine.process(df)

    # Map result to numeric
    result_map = {"A": 0, "D": 1, "H": 2}
    featured_df["Result"] = featured_df["FTR"].map(result_map)
    featured_df = featured_df.dropna(subset=["Result"])
    featured_df["Result"] = featured_df["Result"].astype(int)

    # Remove leakage columns (data only available AFTER match)
    leaky_cols = [
        "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "Result", "League", "Season",
        "home_xG_overperf", "away_xG_overperf", "home_fatigued", "away_fatigued"
    ]
    feature_cols = [c for c in featured_df.columns if c not in leaky_cols]
    
    # Filter out any remaining non-numeric or constant columns
    featured_df = featured_df[feature_cols + ["Result"]].dropna()
    featured_df = featured_df.loc[:, featured_df.nunique() > 1]
    feature_cols = [c for c in feature_cols if c in featured_df.columns and c != "Result"]

    X = featured_df[feature_cols].copy()
    y = featured_df["Result"].copy()
    
    logger.info(f"✅ Prepared {len(X)} samples with {len(feature_cols)} features.")
    return X, y, feature_cols

# ==================== TRAINING PIPELINE ====================
def train_models(X: pd.DataFrame, y: pd.Series):
    """
    Time-series validation, scaling, and ensemble training.
    Returns best model, scaler, and evaluation metrics.
    """
    logger.info("🚀 Starting time-series training pipeline...")
    tscv = TimeSeriesSplit(n_splits=5)
    scaler = StandardScaler()

    models = {
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            eval_metric="mlogloss", use_label_encoder=False
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=10, random_state=42
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, multi_class="multinomial", C=0.5, random_state=42
        )
    }

    best_acc = -1.0
    best_model_name = ""
    results = {}

    for name, model in models.items():
        fold_accs, fold_ll = [], []
        for train_idx, test_idx in tscv.split(X):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

            X_tr_sc = scaler.fit_transform(X_tr)
            X_te_sc = scaler.transform(X_te)

            model.fit(X_tr_sc, y_tr)
            preds = model.predict(X_te_sc)
            proba = model.predict_proba(X_te_sc)

            fold_accs.append(accuracy_score(y_te, preds))
            fold_ll.append(log_loss(y_te, proba))

        mean_acc = np.mean(fold_accs)
        mean_ll = np.mean(fold_ll)
        results[name] = {"acc": mean_acc, "ll": mean_ll}
        logger.info(f"📊 {name}: Accuracy={mean_acc:.4f}, LogLoss={mean_ll:.4f}")

        if mean_acc > best_acc:
            best_acc = mean_acc
            best_model_name = name

    # Train final ensemble with best performers
    logger.info("🤖 Building soft-voting ensemble...")
    ensemble = VotingClassifier(
        estimators=[
            ("xgb", XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, eval_metric="mlogloss")),
            ("rf", RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)),
            ("lr", LogisticRegression(max_iter=1000, multi_class="multinomial", C=0.5, random_state=42))
        ],
        voting="soft",
        weights=[2, 1, 1]
    )

    # Final fit on full dataset (walk-forward style assumes no future leak in production retrain)
    X_scaled = scaler.fit_transform(X)
    ensemble.fit(X_scaled, y)

    logger.info(f"🏆 Best single model: {best_model_name} (Acc: {best_acc:.4f})")
    return ensemble, scaler

# ==================== PERSISTENCE ====================
def save_artifacts(model, scaler, feature_cols: list):
    """Save trained models and metadata to disk."""
    model_path = MODELS_DIR / "ensemble_model.pkl"
    scaler_path = MODELS_DIR / "scaler.pkl"
    meta_path = MODELS_DIR / "feature_metadata.pkl"

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump({"feature_cols": feature_cols, "trained_at": datetime.now().isoformat()}, meta_path)

    logger.info(f"💾 Artifacts saved to: {MODELS_DIR}")
    logger.info(f"   📦 Model: {model_path}")
    logger.info(f"   📏 Scaler: {scaler_path}")
    logger.info(f"   📋 Features: {meta_path}")

# ==================== MAIN ====================
if __name__ == "__main__":
    try:
        logger.info("="*60)
        logger.info("🔄 STARTING MODEL RETRAINING PIPELINE")
        logger.info("="*60)

        df = load_training_data(use_dummy=True)
        X, y, feature_cols = prepare_features_and_target(df)
        ensemble, scaler = train_models(X, y)
        save_artifacts(ensemble, scaler, feature_cols)

        logger.info("✅ RETRAINING COMPLETE. Model ready for production inference.")
    except Exception as e:
        logger.error(f"❌ Retraining failed: {e}", exc_info=True)
        sys.exit(1)


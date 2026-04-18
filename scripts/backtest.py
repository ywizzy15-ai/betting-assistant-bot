"""
scripts/backtest.py - Walk-Forward Validation & Backtesting Module.
Simulates real-time trading to evaluate model performance and ROI over time.
Implements strict time-series splitting to prevent look-ahead bias.
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add project root to path to allow imports if running this script directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# Import project modules
from prediction.features import FeatureEngineer

# ==========================================================
# DATA SIMULATION (Remove this when using real CSV data)
# ==========================================================
def generate_dummy_data(n_matches=1000):
    """Generates synthetic historical data for testing the backtest pipeline."""
    dates = pd.date_range(end=datetime.now(), periods=n_matches, freq='3D')
    teams = ["Team A", "Team B", "Team C", "Team D", "Team E"]
    
    data = []
    for i in range(n_matches):
        h_team, a_team = np.random.choice(teams, 2, replace=False)
        # Random scores weighted towards low scores
        fthg = np.random.poisson(1.2)
        ftag = np.random.poisson(1.1)
        ftr = "H" if fthg > ftag else ("A" if ftag > fthg else "D")
        
        data.append({
            "Date": dates[i],
            "HomeTeam": h_team,
            "AwayTeam": a_team,
            "FTHG": fthg,
            "FTAG": ftag,
            "FTR": ftr,
            # Simulate stats needed for features
            "HS": np.random.poisson(10), "HST": np.random.poisson(4),
            "AS": np.random.poisson(10), "AST": np.random.poisson(4),
            "B365H": np.round(1.5 + np.random.rand(), 2),
            "B365D": np.round(3.0 + np.random.rand(), 2),
            "B365A": np.round(4.0 + np.random.rand(), 2),
        })
    return pd.DataFrame(data)

# ==========================================================
# WALK-FORWARD BACKTEST ENGINE
# ==========================================================
class WalkForwardBacktest:
    """
    Executes a walk-forward backtest strategy:
    1. Train on past data (Window).
    2. Predict next N matches (Test).
    3. Roll window forward.
    4. Calculate Accuracy, LogLoss, and Simulated ROI.
    """
    def __init__(self, model=None, initial_train_size=100, step_size=20):
        self.model = model or XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42
        )
        self.scaler = StandardScaler()
        self.initial_train_size = initial_train_size
        self.step_size = step_size
        
        # Storage for results
        self.results = {
            "y_true": [], "y_pred": [], "y_proba": [],
            "odds": [], "profits": [], "roi_history": []
        }

    def run(self, df: pd.DataFrame, feature_cols: list, target_col: str = "Result"):
        """
        Run the backtest loop.
        
        Args:
            df: Dataframe with features and target.
            feature_cols: List of column names to use as inputs.
            target_col: Column name for the result (0=A, 1=D, 2=H).
        """
        print(f"🚀 Starting Walk-Forward Backtest...")
        print(f"   Total Matches: {len(df)} | Initial Window: {self.initial_train_size} | Step: {self.step_size}")

        current_profit = 0.0
        stake = 100.0  # Fixed stake per bet

        # Time Series Splits
        # We manually implement the loop to capture ROI/odds which sklearn's cross_val_score doesn't do
        start_idx = self.initial_train_size
        total_len = len(df)
        
        while start_idx < total_len:
            end_idx = min(start_idx + self.step_size, total_len)
            
            # 1. Split Data
            train_df = df.iloc[:start_idx]
            test_df = df.iloc[start_idx:end_idx]
            
            if len(train_df) < 10 or len(test_df) == 0:
                start_idx += 1
                continue

            X_train = train_df[feature_cols].fillna(0)
            y_train = train_df[target_col]
            X_test = test_df[feature_cols].fillna(0)
            y_test = test_df[target_col]
            odds_test = test_df.get("odds_target", pd.Series([2.0] * len(test_df))) # Mock odds if missing

            # 2. Scale & Train
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            self.model.fit(X_train_scaled, y_train)
            
            # 3. Predict
            preds = self.model.predict(X_test_scaled)
            proba = self.model.predict_proba(X_test_scaled)
            
            # 4. Simulate Betting (Simple Strategy)
            # Bet if model confidence > 60% (e.g., max proba > 0.60)
            max_probs = np.max(proba, axis=1)
            for i in range(len(test_df)):
                self.results["y_true"].append(y_test.iloc[i])
                self.results["y_pred"].append(preds[i])
                self.results["y_proba"].append(proba[i])
                
                conf = max_probs[i]
                # Check if we would have bet (Mock threshold)
                if conf > 0.55: 
                    # Simplified ROI logic: if win, profit = stake * (odds - 1), else lose stake
                    actual_result = y_test.iloc[i]
                    predicted_result = preds[i]
                    bet_odds = float(odds_test.iloc[i]) if isinstance(odds_test, pd.Series) else 2.0
                    
                    if actual_result == predicted_result:
                        current_profit += stake * (bet_odds - 1)
                        self.results["profits"].append(stake * (bet_odds - 1))
                    else:
                        current_profit -= stake
                        self.results["profits"].append(-stake)
                else:
                    self.results["profits"].append(0)
            
            self.results["roi_history"].append(current_profit)

            # Roll window
            start_idx += self.step_size

        self.results["roi_history"] = np.array(self.results["roi_history"])
        self.report()

    def report(self):
        """Prints and plots the backtest report."""
        y_true = np.array(self.results["y_true"])
        y_pred = np.array(self.results["y_pred"])
        y_proba = np.array(self.results["y_proba"])
        profits = np.array(self.results["profits"])
        roi = self.results["roi_history"]

        print("\n" + "="*60)
        print("📊 BACKTEST REPORT")
        print("="*60)
        print(f"✅ Total Predictions: {len(y_true)}")
        print(f"🎯 Accuracy:         {accuracy_score(y_true, y_pred):.4f}")
        print(f"📉 Log Loss:         {log_loss(y_true, y_proba):.4f}")
        
        # Filter non-zero bets for ROI calc
        bets_placed = profits[profits != 0]
        if len(bets_placed) > 0:
            total_roi = np.sum(profits)
            roi_pct = (total_roi / (abs(np.sum(bets_placed[bets_placed < 0]) + len(bets_placed[bets_placed > 0])*100)) * 100) # Rough ROI%
            print(f"💰 Net Profit:       ${total_roi:.2f}")
            print(f"📈 ROI:              {roi_pct:.2f}%")
            print(f"📝 Bets Placed:      {len(bets_placed)}")
        
        print("="*60)
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=['Away Win', 'Draw', 'Home Win']))

        # Plot Cumulative Profit
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(10, 6))
        plt.plot(np.cumsum(profits), label='Cumulative Profit ($)', color='#2ecc71', linewidth=2)
        plt.axhline(0, color='r', linestyle='--', alpha=0.5)
        plt.title("Backtest Performance: Cumulative Profit")
        plt.xlabel("Prediction Step")
        plt.ylabel("Profit ($)")
        plt.legend()
        
        # Save plot to scripts folder
        plot_path = os.path.join(os.path.dirname(__file__), "backtest_results.png")
        plt.savefig(plot_path)
        print(f"\n📈 Chart saved to: {plot_path}")
        plt.show()

# ==========================================================
# MAIN EXECUTION
# ==========================================================
if __name__ == "__main__":
    # 1. Load Data (Use real CSV or Dummy)
    # df = pd.read_csv("data/premier_league.csv") # Example for real data
    df = generate_dummy_data(n_matches=600)
    
    # 2. Feature Engineering
    # Requires 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR'
    # And stats columns 'HS', 'HST', 'AS', 'AST' for xG proxy
    print("🔧 Engineering features...")
    engine = FeatureEngineer(elo_k=32, home_adv=65)
    
    # Note: generate_dummy_data creates necessary columns, real data must match schema
    featured_df = engine.process(df)

    # 3. Prepare Target
    # Map FTR to numeric: A=0, D=1, H=2
    result_map = {"A": 0, "D": 1, "H": 2}
    featured_df["Result"] = featured_df["FTR"].map(result_map)
    featured_df = featured_df.dropna(subset=["Result"])
    featured_df["Result"] = featured_df["Result"].astype(int)

    # 4. Select Features (Only those available BEFORE match)
    # Exclude leaky columns (FTR, FTHG, etc.)
    feature_cols = [c for c in featured_df.columns 
                    if c not in ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "Result"]
                    and not c.startswith("home_xG_overperf")] # Overperf is calculated post-match usually
    
    # Add mock odds column if missing (for ROI sim)
    if "odds_target" not in featured_df.columns:
        featured_df["odds_target"] = 1.90 # Fixed odds for dummy run

    # 5. Run Backtest
    backtester = WalkForwardBacktest(
        model=XGBClassifier(n_estimators=50, max_depth=3, random_state=42),
        initial_train_size=100,
        step_size=20
    )
    backtester.run(featured_df, feature_cols=feature_cols, target_col="Result")


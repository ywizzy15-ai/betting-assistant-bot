"""
features.py - Feature engineering pipeline for football prediction models.
Implements ELO ratings, xG proxy, fatigue tracking, H2H history, and rolling stats.
Based on the technical guide's data processing layer.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from config import (
    ELO_K_FACTOR, ELO_HOME_ADVANTAGE,
    XG_SOT_CONVERSION, XG_OFF_TARGET_CONVERSION,
    FATIGUE_THRESHOLD_DAYS, DEFAULT_INITIAL_REST
)

# ==================== ELO RATING SYSTEM ====================
class FootballELO:
    """
    ELO ratings for football teams using FIFA/FiveThirtyEight hybrid logic.
    Formula: R_new = R_old + K * M * (S - E)
    """
    def __init__(self, k: int = ELO_K_FACTOR, home_advantage: int = ELO_HOME_ADVANTAGE):
        self.k = k
        self.home_advantage = home_advantage
        self.ratings: Dict[str, float] = {}

    def get_rating(self, team: str) -> float:
        return self.ratings.setdefault(team, 1500.0)

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Probability of A beating B using the standard ELO formula."""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def margin_multiplier(self, goal_diff: int) -> float:
        """FiveThirtyEight style goal difference multiplier."""
        return np.log(abs(goal_diff) + 1) * 2.2 / (self.k * 0.001 + 2.2)

    def update(self, home: str, away: str, home_goals: int, away_goals: int) -> tuple[float, float]:
        r_home = self.get_rating(home) + self.home_advantage
        r_away = self.get_rating(away)

        e_home = self.expected_score(r_home, r_away)
        e_away = 1.0 - e_home

        if home_goals > away_goals:
            s_home, s_away = 1.0, 0.0
        elif home_goals < away_goals:
            s_home, s_away = 0.0, 1.0
        else:
            s_home, s_away = 0.5, 0.5

        m = self.margin_multiplier(home_goals - away_goals)

        self.ratings[home] += self.k * m * (s_home - e_home)
        self.ratings[away] += self.k * m * (s_away - e_away)

        return self.ratings[home], self.ratings[away]

    def compute_elo_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Chronological ELO computation. Saves pre-match ratings to prevent leakage.
        Requires columns: 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG'
        """
        df = df.sort_values("Date").copy()
        elo_features = []

        for _, row in df.iterrows():
            home, away = row["HomeTeam"], row["AwayTeam"]
            r_home, r_away = self.get_rating(home), self.get_rating(away)

            e_home = self.expected_score(r_home + self.home_advantage, r_away)

            elo_features.append({
                "elo_home": r_home,
                "elo_away": r_away,
                "elo_diff": r_home - r_away,
                "elo_expected_home": e_home,
                "elo_expected_away": 1.0 - e_home,
            })

            if pd.notna(row.get("FTHG")) and pd.notna(row.get("FTAG")):
                self.update(home, away, int(row["FTHG"]), int(row["FTAG"]))

        return pd.concat([df.reset_index(drop=True), pd.DataFrame(elo_features, index=df.index)], axis=1)


# ==================== XG PROXY ====================
def compute_xg_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Approximate xG from basic shot statistics.
    Formula: xG ≈ SoT * 0.30 + (Shots - SoT) * 0.03
    Requires columns: HS, HST, AS, AST, FTHG, FTAG
    """
    df = df.copy()
    if "HST" not in df.columns or "HS" not in df.columns:
        return df

    df["home_xG_proxy"] = (
        df["HST"] * XG_SOT_CONVERSION +
        (df["HS"] - df["HST"]).clip(lower=0) * XG_OFF_TARGET_CONVERSION
    )
    df["away_xG_proxy"] = (
        df["AST"] * XG_SOT_CONVERSION +
        (df["AS"] - df["AST"]).clip(lower=0) * XG_OFF_TARGET_CONVERSION
    )

    df["home_xG_overperf"] = df["FTHG"] - df["home_xG_proxy"]
    df["away_xG_overperf"] = df["FTAG"] - df["away_xG_proxy"]
    return df


# ==================== FATIGUE & FIXTURE CONGESTION ====================
def compute_fatigue_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tracks rest days between matches. <3 days = fatigued.
    Requires columns: Date, HomeTeam, AwayTeam
    """
    df = df.sort_values("Date").copy()
    rest_home, rest_away = [], []
    last_match: Dict[str, pd.Timestamp] = {}

    for _, row in df.iterrows():
        home, away, date = row["HomeTeam"], row["AwayTeam"], row["Date"]
        
        delta_h = (date - last_match[home]).days if home in last_match else DEFAULT_INITIAL_REST
        delta_a = (date - last_match[away]).days if away in last_match else DEFAULT_INITIAL_REST
        
        rest_home.append(min(delta_h, 30))
        rest_away.append(min(delta_a, 30))
        
        last_match[home] = date
        last_match[away] = date

    df["home_rest_days"] = rest_home
    df["away_rest_days"] = rest_away
    df["rest_advantage"] = df["home_rest_days"] - df["away_rest_days"]
    df["home_fatigued"] = (df["home_rest_days"] <= FATIGUE_THRESHOLD_DAYS).astype(int)
    df["away_fatigued"] = (df["away_rest_days"] <= FATIGUE_THRESHOLD_DAYS).astype(int)
    df["is_midweek"] = df["Date"].dt.dayofweek.isin([1, 2]).astype(int)
    return df


# ==================== HEAD-TO-HEAD HISTORY ====================
def compute_h2h_features(df: pd.DataFrame, n_last: int = 5) -> pd.DataFrame:
    """
    Computes recent H2H statistics for each matchup.
    Requires columns: Date, HomeTeam, AwayTeam, FTR, FTHG, FTAG
    """
    df = df.sort_values("Date").copy()
    h2h_features = []

    for idx, row in df.iterrows():
        home, away, date = row["HomeTeam"], row["AwayTeam"], row["Date"]
        prev = df[
            (df["Date"] < date) &
            (((df["HomeTeam"] == home) & (df["AwayTeam"] == away)) |
             ((df["HomeTeam"] == away) & (df["AwayTeam"] == home)))
        ].tail(n_last)

        if len(prev) < 2:
            h2h_features.append({"h2h_home_wins": np.nan, "h2h_draws": np.nan, "h2h_total_goals_avg": np.nan})
            continue

        home_wins, draws, total_goals = 0, 0, 0
        for _, p in prev.iterrows():
            is_home_perspective = p["HomeTeam"] == home
            if p["FTR"] == "H" and is_home_perspective: home_wins += 1
            elif p["FTR"] == "A" and not is_home_perspective: home_wins += 1
            elif p["FTR"] == "D": draws += 1
            total_goals += p["FTHG"] + p["FTAG"]

        n = len(prev)
        h2h_features.append({
            "h2h_home_wins": home_wins / n,
            "h2h_draws": draws / n,
            "h2h_total_goals_avg": total_goals / n,
        })

    return pd.concat([df, pd.DataFrame(h2h_features, index=df.index)], axis=1)


# ==================== ROLLING TEAM STATS ====================
def compute_rolling_stats(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Computes rolling attack/defense form per team.
    Uses shift(1) to prevent data leakage.
    """
    df = df.sort_values("Date").copy()
    stats_cols = ["FTHG", "FTAG", "HS", "AS", "HST", "AST", "HC", "AC"]
    
    # Create unified team records
    home_rec = df[["Date", "HomeTeam"] + stats_cols].rename(columns={"HomeTeam": "Team"})
    home_rec["IsHome"] = 1
    home_rec = home_rec[["Date", "Team", "IsHome"] + stats_cols]

    away_rec = df[["Date", "AwayTeam"] + ["FTAG", "FTHG", "AS", "HS", "AST", "HST", "AC", "HC"]].copy()
    away_rec.rename(columns={"AwayTeam": "Team", "FTAG": "FTHG", "FTHG": "FTAG",
                             "AS": "HS", "HS": "AS", "AST": "HST", "HST": "AST",
                             "AC": "HC", "HC": "AC"}, inplace=True)
    away_rec["IsHome"] = 0
    away_rec = away_rec[["Date", "Team", "IsHome"] + stats_cols]

    all_rec = pd.concat([home_rec, away_rec]).sort_values("Date")
    all_rec["Points"] = all_rec.apply(lambda r: 3 if r["FTHG"] > r["FTAG"] else (1 if r["FTHG"] == r["FTAG"] else 0), axis=1)

    rolling = {}
    for team in all_rec["Team"].unique():
        t_data = all_rec[all_rec["Team"] == team].copy()
        for col in stats_cols + ["Points"]:
            t_data[f"roll_{col}"] = t_data[col].shift(1).rolling(window, min_periods=3).mean()
        rolling[team] = t_data

    team_stats = pd.concat(rolling.values())
    return pd.merge(df, team_stats, left_on=["HomeTeam", "Date"], right_on=["Team", "Date"], how="left").drop(columns=["Team_y", "IsHome_y"], errors="ignore")


# ==================== PIPELINE WRAPPER ====================
class FeatureEngineer:
    """Convenience wrapper to apply all feature engineering steps sequentially."""
    def __init__(self, elo_k: int = ELO_K_FACTOR, home_adv: int = ELO_HOME_ADVANTAGE):
        self.elo = FootballELO(k=elo_k, home_advantage=home_adv)

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.elo.compute_elo_features(df)
        df = compute_xg_proxy(df)
        df = compute_fatigue_features(df)
        df = compute_h2h_features(df)
        df = compute_rolling_stats(df)
        return df


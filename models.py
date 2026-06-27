import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import pickle

class PredictiveMaintenanceModels:
    def __init__(self):
        self.classifier = None
        self.regressor = None
        self.scaler = StandardScaler()
        self.feature_cols = [
            "Vibration", "Temperature", "Pressure", "Speed", "OilFlow",
            "Vibration_roll_mean_6", "Vibration_roll_std_6",
            "Temperature_roll_mean_6", "Temperature_roll_std_6",
            "Pressure_roll_mean_6", "Pressure_roll_std_6"
        ]
        
    def engineer_features(self, df):
        """Creates rolling window features for training/inference."""
        df = df.copy()
        df.sort_values(by=["AssetID", "Timestamp"], inplace=True)
        
        # Calculate rolling metrics per asset
        df["Vibration_roll_mean_6"] = df.groupby("AssetID")["Vibration"].transform(lambda x: x.rolling(6, min_periods=1).mean())
        df["Vibration_roll_std_6"] = df.groupby("AssetID")["Vibration"].transform(lambda x: x.rolling(6, min_periods=1).std().fillna(0.0))
        
        df["Temperature_roll_mean_6"] = df.groupby("AssetID")["Temperature"].transform(lambda x: x.rolling(6, min_periods=1).mean())
        df["Temperature_roll_std_6"] = df.groupby("AssetID")["Temperature"].transform(lambda x: x.rolling(6, min_periods=1).std().fillna(0.0))
        
        df["Pressure_roll_mean_6"] = df.groupby("AssetID")["Pressure"].transform(lambda x: x.rolling(6, min_periods=1).mean())
        df["Pressure_roll_std_6"] = df.groupby("AssetID")["Pressure"].transform(lambda x: x.rolling(6, min_periods=1).std().fillna(0.0))
        
        return df

    def train(self, historical_df):
        """Preprocesses data and trains the XGBoost models."""
        print("Starting feature engineering...")
        df_feats = self.engineer_features(historical_df)
        
        # Target for classifier: Did a failure happen in the next 12 hours?
        # Or simpler: Target failure state directly if we have simulated warning labels.
        # Let's define warning threshold: FailureProb_GT > 0.3 or FailureState == 1
        df_feats["Target_Fail"] = (df_feats["FailureProb_GT"] > 0.25).astype(int)
        
        # Target for regressor: RUL
        # Drop rows that are currently in a failed/repaired state for stable RUL training
        df_train = df_feats[df_feats["FailureState"] == 0].copy()
        
        X = df_train[self.feature_cols]
        y_class = df_train["Target_Fail"]
        y_rul = df_train["RUL"]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        print("Training XGBoost Classifier for Failure Risk...")
        self.classifier = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
            eval_metric="logloss"
        )
        self.classifier.fit(X_scaled, y_class)
        
        print("Training XGBoost Regressor for RUL prediction...")
        self.regressor = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            random_state=42
        )
        self.regressor.fit(X_scaled, y_rul)
        print("Models successfully trained!")
        
    def predict(self, current_df):
        """Predicts failure probability and RUL for current asset state."""
        df_feats = self.engineer_features(current_df)
        last_rows = df_feats.groupby("AssetID").last().reset_index()
        
        X_last = last_rows[self.feature_cols]
        X_scaled = self.scaler.transform(X_last)
        
        probs = self.classifier.predict_proba(X_scaled)[:, 1]
        ruls = self.regressor.predict(X_scaled)
        
        # Get feature contribution per asset (simple decision path approximation or feature importance scaling)
        contributions = []
        importances = self.classifier.feature_importances_
        
        for idx, row in last_rows.iterrows():
            # Estimate sensor contributions based on standard deviations and feature importance weights
            asset_contribs = {}
            total_wt = 0.0
            
            # Simple heuristic for real-time dashboard: deviation from baseline * feature importance
            # We want to display which sensors are causing the elevated risk
            for col_idx, col in enumerate(self.feature_cols):
                val = row[col]
                # Compare to baseline values to get deviation direction
                dev = 0.0
                if "Vibration" in col:
                    dev = max(0.0, val - 2.5) / 1.5
                elif "Temperature" in col:
                    dev = max(0.0, val - 65.0) / 10.0
                elif "Pressure" in col:
                    dev = max(0.0, 45.0 - val) / 10.0 # pressure drop is bad
                elif "Speed" in col:
                    dev = abs(val - 1500.0) / 200.0
                elif "OilFlow" in col:
                    dev = max(0.0, 12.0 - val) / 3.0 # flow drop is bad
                    
                score = dev * importances[col_idx]
                asset_contribs[col] = score
                total_wt += score
                
            if total_wt == 0:
                # Fallback to general importance weights if normal operation
                asset_contrib_pcts = {col.split("_")[0]: float(importances[col_idx]) for col_idx, col in enumerate(self.feature_cols) if "roll" not in col}
            else:
                # Group by base sensor names for dashboard display
                base_contribs = {}
                for k, v in asset_contribs.items():
                    base_name = k.split("_")[0]
                    base_contribs[base_name] = base_contribs.get(base_name, 0.0) + v
                
                sum_base = sum(base_contribs.values())
                asset_contrib_pcts = {k: float(v / sum_base) for k, v in base_contribs.items()}
                
            contributions.append(asset_contrib_pcts)
            
        results = {}
        for idx, row in last_rows.iterrows():
            asset_id = row["AssetID"]
            # Ensure boundaries are realistic
            prob = float(probs[idx])
            if row["FailureState"] == 1:
                prob = 1.0
                
            rul = float(ruls[idx])
            if row["FailureState"] == 1:
                rul = 0.0
            rul = max(0.0, rul)
            
            results[asset_id] = {
                "FailureProbability": prob,
                "RUL": rul,
                "Contributions": contributions[idx],
                "Vibration": row["Vibration"],
                "Temperature": row["Temperature"],
                "Pressure": row["Pressure"],
                "Speed": row["Speed"],
                "OilFlow": row["OilFlow"],
                "StressProfile": row["StressProfile"]
            }
            
        return results

#!/usr/bin/env python
"""
real_estate_investment_model.py

Consumes combined property-level features and area-level stats to train valuation and rental models.
"""
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class RealEstateInvestmentModel:
    """
    Trains on:
      - processed_investment_data.csv (contains both property-level features and area-level stats)
    Targets: PropertyValuation, AnnualRent
    """
    def __init__(self, data_path: str):
        # Load combined data
        self.df = pd.read_csv(data_path)
        
        # Validate required columns
        required_cols = ['AreaCode', 'PropertyValuation', 'AnnualRent']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in dataset: {missing_cols}")

        # Pre-save the market avg yield for scoring
        self.market_avg_yield = self.df['RentalYield'].mean()
        self.models = {}
        self.results = {}

    def train(self, test_size=0.2, random_state=42, n_iter=20):
        """
        Train RandomForestRegressor for each target via RandomizedSearchCV.
        """
        # Identify columns to exclude from features
        exclude_cols = ['PropertyValuation', 'AnnualRent', 'AreaName']
        features = [c for c in self.df.columns if c not in exclude_cols]
        X = self.df[features]

        for target in ['PropertyValuation', 'AnnualRent']:
            y = self.df[target]
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

            pipe = Pipeline([
                ('scale', StandardScaler()),
                ('rf', RandomForestRegressor(random_state=random_state))
            ])
            param_dist = {
                'rf__n_estimators':      [100, 200, 300],
                'rf__max_depth':         [None, 10, 20, 30],
                'rf__min_samples_split': [2, 5, 10]
            }
            search = RandomizedSearchCV(
                pipe, param_dist, n_iter=n_iter, cv=3,
                scoring='neg_mean_absolute_error', n_jobs=-1,
                random_state=random_state
            )
            search.fit(X_train, y_train)

            best = search.best_estimator_
            preds = best.predict(X_val)

            mse  = mean_squared_error(y_val, preds)
            mae  = mean_absolute_error(y_val, preds)
            mape = np.mean(np.abs((y_val - preds) / y_val)) * 100
            r2   = r2_score(y_val, preds)

            self.models[target] = best
            self.results[target] = {
                'best_params': search.best_params_,
                'mse':         mse,
                'mae':         mae,
                'mape':        mape,
                'r2':          r2
            }

        return self.models, self.results

    def save_models(self, directory: str):
        """
        Save each trained model to disk.
        """
        os.makedirs(directory, exist_ok=True)
        for target, model in self.models.items():
            fname = f"{target.lower()}_model.joblib"
            joblib.dump(model, os.path.join(directory, fname))
        return directory

    def load_models(self, directory: str):
        """
        Load previously saved models.
        """
        for fname in os.listdir(directory):
            if fname.endswith('_model.joblib'):
                target = fname.replace('_model.joblib','').capitalize()
                self.models[target] = joblib.load(os.path.join(directory, fname))
        return self.models

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return predictions for both targets on a new DataFrame of features.
        """
        df_feat = df.copy()
        preds = pd.DataFrame(index=df_feat.index)
        for target, model in self.models.items():
            preds[target] = model.predict(df_feat)
        return preds

    def score_investment(self) -> pd.DataFrame:
        """
        Compute investment scores for every row in self.df.
        Returns a DataFrame keyed by AreaCode.
        """
        df = self.df.copy()
        # Drop actuals & AreaName before predicting
        features_to_drop = ['AreaName', 'PropertyValuation', 'AnnualRent']
        prediction_features = df.drop(columns=features_to_drop, errors='ignore')
        preds = self.predict(prediction_features)

        # Attach predictions
        df['PredVal']  = preds['PropertyValuation']
        df['PredRent'] = preds['AnnualRent']

        # Valuation diff %
        df['ValDiffPct'] = (df['PredVal'] - df['PropertyValuation']) / df['PropertyValuation'] * 100
        df['ValScore']   = np.clip(df['ValDiffPct'] / 30 * 40, 0, 40)

        # Yield score
        df['YieldScore'] = np.clip(
            (df['RentalYield'] - self.market_avg_yield) / self.market_avg_yield * 30, 0, 30
        )

        # Rental upside %
        df['RentDiffPct'] = (df['PredRent'] - df['AnnualRent']) / df['AnnualRent'] * 100
        df['RentScore']   = np.clip(df['RentDiffPct'] / 20 * 20, 0, 20)

        # Location premium
        overall_pps     = df['PricePerSqFt'].mean()
        df['LocPremium']= df['PricePerSqFt'] / overall_pps - 1
        df['LocScore']  = np.clip(df['LocPremium'] / 0.2 * 10, 0, 10)

        df['InvestmentScore'] = df[['ValScore','YieldScore','RentScore','LocScore']].sum(axis=1)

        # Return important columns
        score_columns = [
            'AreaCode', 'AreaName', 'PropertyValuation', 'AnnualRent', 
            'PredVal', 'PredRent', 'InvestmentScore', 
            'ValScore', 'YieldScore', 'RentScore', 'LocScore',
            'ValDiffPct', 'RentDiffPct', 'RentalYield', 'LocPremium'
        ]
        
        return df[score_columns]
    
    def get_area_investment_scores(self) -> pd.DataFrame:
        """
        Calculate area-level aggregate investment scores.
        """
        property_scores = self.score_investment()
        
        # Aggregate by area
        area_scores = (
            property_scores
            .groupby(['AreaCode', 'AreaName'])
            .agg({
                'InvestmentScore': 'mean',
                'ValScore': 'mean',
                'YieldScore': 'mean', 
                'RentScore': 'mean',
                'LocScore': 'mean',
                'ValDiffPct': 'mean',
                'RentDiffPct': 'mean',
                'RentalYield': 'mean',
                'LocPremium': 'mean'
            })
            .reset_index()
            .sort_values('InvestmentScore', ascending=False)
        )
        
        return area_scores
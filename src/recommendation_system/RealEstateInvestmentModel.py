#!/usr/bin/env python
"""
real_estate_investment_model.py

Consumes combined property-level features and area-level stats to train valuation and rental models.
"""
import os
import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Any, Tuple, Optional


class RealEstateInvestmentModel:
    """
    Trains on:
      - processed_investment_data.csv (contains both property-level features and area-level stats)
    Targets: PropertyValuation, AnnualRent
    """

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.logger = self._configure_logging()

        # Initialize empty data
        self.data = None

        # Try to load data immediately if path exists
        if os.path.exists(data_path):
            self.data = pd.read_csv(data_path)
            self.logger.info(
                f"Loaded data with {len(self.data)} records from {data_path}")

            # Pre-save the market avg yield for scoring
            if 'RentalYield' in self.data.columns:
                self.market_avg_yield = self.data['RentalYield'].mean()
            else:
                self.market_avg_yield = None
                self.logger.warning("RentalYield column not found in data")
        else:
            self.logger.warning(
                f"Data path {data_path} does not exist. Load data manually.")
            self.market_avg_yield = None

        self.models = {}
        self.results = {}
        self.property_scores = None
        self.area_scores = None

    def _configure_logging(self) -> logging.Logger:
        """Configure logger for the model."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(levelname)s %(message)s'
        )
        return logging.getLogger(self.__class__.__name__)

    def load_data(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load real estate data from CSV file.

        Args:
            data_path: Optional path to data file (uses instance path if not provided)

        Returns:
            DataFrame with loaded data
        """
        if data_path is None:
            data_path = self.data_path

        try:
            self.data = pd.read_csv(data_path)
            self.logger.info(
                f"Loaded data with {len(self.data)} records from {data_path}")

            # Validate required columns
            required_cols = ['AreaCode', 'PropertyValuation', 'AnnualRent']
            missing_cols = [
                col for col in required_cols if col not in self.data.columns]
            if missing_cols:
                raise ValueError(
                    f"Missing required columns in dataset: {missing_cols}")

            # Set market avg yield
            if 'RentalYield' in self.data.columns:
                self.market_avg_yield = self.data['RentalYield'].mean()

            return self.data
        except Exception as e:
            self.logger.error(f"Error loading data from {data_path}: {str(e)}")
            raise

    def train(self, test_size=0.2, random_state=42, n_iter=20) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]]]:
        """
        Train RandomForestRegressor for each target via RandomizedSearchCV.

        Returns:
            Tuple of (trained_models, performance_metrics)
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        self.logger.info("Starting model training...")

        # Identify columns to exclude from features
        exclude_cols = ['PropertyValuation', 'AnnualRent', 'AreaName']
        features = [c for c in self.data.columns if c not in exclude_cols]
        X = self.data[features]

        for target in ['PropertyValuation', 'AnnualRent']:
            self.logger.info(f"Training model for {target}...")

            y = self.data[target]
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

            mse = mean_squared_error(y_val, preds)
            mae = mean_absolute_error(y_val, preds)
            mape = np.mean(np.abs((y_val - preds) / y_val)) * 100
            r2 = r2_score(y_val, preds)

            self.models[target] = best
            self.results[target] = {
                'best_params': search.best_params_,
                'mse':         mse,
                'mae':         mae,
                'mape':        mape,
                'r2':          r2
            }

            self.logger.info(
                f"{target} model training complete. R²: {r2:.3f}, MAPE: {mape:.2f}%")

        return self.models, self.results

    def save_models(self, directory: str) -> str:
        """
        Save each trained model to disk.

        Args:
            directory: Directory to save models

        Returns:
            Path to saved models directory
        """
        os.makedirs(directory, exist_ok=True)
        for target, model in self.models.items():
            fname = f"{target}_model.joblib"
            model_path = os.path.join(directory, fname)
            joblib.dump(model, model_path)
            self.logger.info(f"Saved {target} model to {model_path}")
        return directory

    def load_models(self, directory: str) -> Dict[str, Any]:
        """
        Load previously saved models.

        Args:
            directory: Directory containing saved models

        Returns:
            Dictionary of loaded models
        """
        loaded_models = {}
        for target in ['PropertyValuation', 'AnnualRent']:
            model_path = os.path.join(directory, f"{target}_model.joblib")

            if os.path.exists(model_path):
                model = joblib.load(model_path)
                loaded_models[target] = model
                self.logger.info(f"Loaded {target} model from {model_path}")
            else:
                self.logger.warning(f"Model file not found: {model_path}")

        # Store loaded models
        self.models = loaded_models

        return loaded_models

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return predictions for both targets on a new DataFrame of features.

        Args:
            df: DataFrame with features for prediction

        Returns:
            DataFrame with predictions
        """
        if not self.models.get('PropertyValuation') or not self.models.get('AnnualRent'):
            raise ValueError(
                "Models not trained or loaded. Train or load models first.")

        df_feat = df.copy()
        preds = pd.DataFrame(index=df_feat.index)
        for target, model in self.models.items():
            preds[target] = model.predict(df_feat)
        return preds

    def score_investment(self, top_n: int = None) -> pd.DataFrame:
        """
        Compute investment scores for every row in self.data.

        Args:
            top_n: Optional number of top properties to return (returns all if None)

        Returns:
            DataFrame with investment scores for properties
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        if not self.models.get('PropertyValuation') or not self.models.get('AnnualRent'):
            raise ValueError(
                "Models not trained or loaded. Train or load models first.")

        self.logger.info("Scoring properties for investment potential...")

        df = self.data.copy()
        # Drop actuals & AreaName before predicting
        features_to_drop = ['AreaName', 'PropertyValuation', 'AnnualRent']
        prediction_features = df.drop(
            columns=features_to_drop, errors='ignore')
        preds = self.predict(prediction_features)

        # Attach predictions
        df['PredVal'] = preds['PropertyValuation']
        df['PredRent'] = preds['AnnualRent']

        # Valuation diff %
        df['ValDiffPct'] = (
            df['PredVal'] - df['PropertyValuation']) / df['PropertyValuation'] * 100
        df['ValScore'] = np.clip(df['ValDiffPct'] / 30 * 40, 0, 40)

        # Yield score
        df['YieldScore'] = np.clip(
            (df['RentalYield'] - self.market_avg_yield) /
            self.market_avg_yield * 30, 0, 30
        )

        # Rental upside %
        df['RentDiffPct'] = (
            df['PredRent'] - df['AnnualRent']) / df['AnnualRent'] * 100
        df['RentScore'] = np.clip(df['RentDiffPct'] / 20 * 20, 0, 20)

        # Location premium
        overall_pps = df['PricePerSqFt'].mean()
        df['LocPremium'] = df['PricePerSqFt'] / overall_pps - 1
        df['LocScore'] = np.clip(df['LocPremium'] / 0.2 * 10, 0, 10)

        df['InvestmentScore'] = df[['ValScore', 'YieldScore',
                                    'RentScore', 'LocScore']].sum(axis=1)

        # Calculate price per square foot if not already in data
        if 'PricePerSqFt' not in df.columns and 'Size_sqft' in df.columns:
            df['PricePerSqFt'] = df['PropertyValuation'] / df['Size_sqft']

        # Add opportunity type classification
        conditions = [
            (df['ValScore'] > df['YieldScore']) & (
                df['ValScore'] > df['RentScore']),
            (df['YieldScore'] > df['ValScore']) & (
                df['YieldScore'] > df['RentScore']),
            (df['RentScore'] > df['ValScore']) & (
                df['RentScore'] > df['YieldScore'])
        ]
        choices = ['Undervalued', 'High Yield', 'Rental Upside']
        df['Opportunity_Type'] = np.select(
            conditions, choices, default='Balanced')

        # Return important columns
        score_columns = [
            'AreaCode', 'AreaName', 'PropertyValuation', 'AnnualRent',
            'PredVal', 'PredRent', 'InvestmentScore',
            'ValScore', 'YieldScore', 'RentScore', 'LocScore',
            'ValDiffPct', 'RentDiffPct', 'RentalYield', 'LocPremium',
            'Opportunity_Type'
        ]

        # Store property scores
        property_scores = df[[
            col for col in score_columns if col in df.columns]]
        self.property_scores = property_scores

        # Return top N properties if specified
        if top_n is not None:
            scored_properties = property_scores.sort_values(
                'InvestmentScore', ascending=False).head(top_n)
        else:
            scored_properties = property_scores

        self.logger.info(
            f"Investment scoring complete for {len(scored_properties)} properties")
        return scored_properties

    def get_area_investment_scores(self, top_n: int = None) -> pd.DataFrame:
        """
        Calculate area-level aggregate investment scores.

        Args:
            top_n: Optional number of top areas to return (returns all if None)

        Returns:
            DataFrame with investment scores for areas
        """
        if self.property_scores is None:
            self.score_investment()

        self.logger.info("Aggregating investment scores by area...")

        # Aggregate by area
        area_scores = (
            self.property_scores
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

        # Add Category column for compatibility with RecommendationEngine
        conditions = [
            (area_scores['ValDiffPct'] >= 10),  # Threshold for undervalued
            (area_scores['ValDiffPct'] <= -10),  # Threshold for overvalued
            (area_scores['RentDiffPct'] >= 15),  # Threshold for growth
            (area_scores['RentalYield'] >= 5.5)  # Threshold for high yield
        ]
        choices = ['Undervalued', 'Overvalued', 'Growth', 'High Yield']
        area_scores['Category'] = np.select(
            conditions, choices, default='Neutral')

        self.area_scores = area_scores

        # Return top N areas if specified
        if top_n is not None:
            top_areas = area_scores.head(top_n)
        else:
            top_areas = area_scores

        self.logger.info(
            f"Area investment scoring complete for {len(top_areas)} areas")
        return top_areas

    def get_undervalued_properties(self, threshold: float = 10.0, top_n: int = 10) -> pd.DataFrame:
        """
        Get undervalued properties based on valuation difference percentage.

        Args:
            threshold: Minimum percentage undervalued
            top_n: Number of top properties to return

        Returns:
            DataFrame with undervalued properties
        """
        if self.property_scores is None:
            self.score_investment()

        # Find undervalued properties (positive ValDiffPct means predicted value is higher)
        undervalued = self.property_scores[self.property_scores['ValDiffPct'] >= threshold]

        # Sort by valuation difference percentage
        top_undervalued = undervalued.sort_values(
            'ValDiffPct', ascending=False).head(top_n)

        return top_undervalued

    def get_high_yield_properties(self, threshold: float = 5.5, top_n: int = 10) -> pd.DataFrame:
        """
        Get high-yield properties based on rental yield.

        Args:
            threshold: Minimum rental yield percentage
            top_n: Number of top properties to return

        Returns:
            DataFrame with high-yield properties
        """
        if self.property_scores is None:
            self.score_investment()

        # Find high yield properties
        high_yield = self.property_scores[self.property_scores['RentalYield'] >= threshold]

        # Sort by rental yield
        top_high_yield = high_yield.sort_values(
            'RentalYield', ascending=False).head(top_n)

        return top_high_yield

    def get_growth_areas(self, threshold: float = 15.0, top_n: int = 5) -> pd.DataFrame:
        """
        Get areas with high growth potential.

        Args:
            threshold: Minimum growth potential percentage
            top_n: Number of top areas to return

        Returns:
            DataFrame with high-growth areas
        """
        if self.area_scores is None:
            self.get_area_investment_scores()

        # Find high growth areas
        growth_areas = self.area_scores[self.area_scores['RentDiffPct'] >= threshold]

        # Sort by rent difference percentage
        top_growth_areas = growth_areas.sort_values(
            'RentDiffPct', ascending=False).head(top_n)

        return top_growth_areas

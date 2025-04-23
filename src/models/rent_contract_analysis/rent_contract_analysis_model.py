"""
RentContractModel: Core model for analyzing rental contracts

This module contains the core model logic for analyzing rental contracts and predicting prices.
Separated from data processing and visualization functions.
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib


class RentContractModel:
    """
    A model for analyzing rental contracts and predicting optimal rental prices.
    
    This model focuses on the core analysis and prediction functionality,
    separating it from data processing and visualization.
    """
    
    def __init__(self):
        """Initialize the Rent Contract Model."""
        self.contracts_df = None
        self.rental_model = None
        self.market_averages = None
        self.seasonal_trends = None
        self.property_premiums = {
            'Flat': 1.0,            # Base reference
            'Villa': 1.8,           # Premium over flat
            'Office': 1.3,          # Premium over flat
            'Shop': 1.5,            # Premium over flat
            'Warehouse': 0.8        # Discount to flat
        }
        
    def load_data(self, data_path=None, dataframe=None):
        """
        Load rental contract data from a CSV file or a pandas DataFrame.
        
        Parameters:
            data_path (str, optional): Path to the CSV file containing rental contract data.
            dataframe (pd.DataFrame, optional): DataFrame containing rental contract data.
            
        Returns:
            pd.DataFrame: The loaded rental contract data.
        """
        if data_path:
            self.contracts_df = pd.read_csv(data_path)
        elif dataframe is not None:
            self.contracts_df = dataframe.copy()
        else:
            raise ValueError("Either data_path or dataframe must be provided")
            
        # Validate required columns
        required_columns = ['contract_start_date', 'contract_end_date', 'contract_amount', 
                           'ejari_property_type_en', 'property_usage_en', 'area_name_en']
        missing_columns = [col for col in required_columns if col not in self.contracts_df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        # Process date columns
        self._process_date_columns()
        
        # Calculate derived metrics
        self._calculate_derived_metrics()
        
        return self.contracts_df
    
    def _process_date_columns(self):
        """Process and standardize date columns."""
        # Convert contract_start_date to datetime
        if 'contract_start_date' in self.contracts_df.columns:
            if self.contracts_df['contract_start_date'].dtype != 'datetime64[ns]':
                self.contracts_df['contract_start_date'] = pd.to_datetime(
                    self.contracts_df['contract_start_date'], errors='coerce')
        
        # Convert contract_end_date to datetime (might have different format)
        if 'contract_end_date' in self.contracts_df.columns:
            if self.contracts_df['contract_end_date'].dtype != 'datetime64[ns]':
                # Try different formats
                try:
                    self.contracts_df['contract_end_date'] = pd.to_datetime(
                        self.contracts_df['contract_end_date'], errors='coerce')
                except:
                    # Try with different format (DD-MM-YYYY)
                    self.contracts_df['contract_end_date'] = pd.to_datetime(
                        self.contracts_df['contract_end_date'], format='%d-%m-%Y', errors='coerce')
        
        # Drop rows with missing dates
        self.contracts_df.dropna(subset=['contract_start_date', 'contract_end_date'], inplace=True)
        
        # Extract year, month from start date
        self.contracts_df['contract_year'] = self.contracts_df['contract_start_date'].dt.year
        self.contracts_df['contract_month'] = self.contracts_df['contract_start_date'].dt.month
        self.contracts_df['contract_quarter'] = self.contracts_df['contract_start_date'].dt.quarter
    
    def _calculate_derived_metrics(self):
        """Calculate derived metrics for all contracts."""
        if self.contracts_df is None or len(self.contracts_df) == 0:
            return
            
        # Calculate contract duration in months
        self.contracts_df['contract_duration_days'] = (
            pd.to_datetime(self.contracts_df['contract_end_date']) - 
            pd.to_datetime(self.contracts_df['contract_start_date'])
        ).dt.days
        
        # Approximate duration in months
        self.contracts_df['contract_duration_months'] = self.contracts_df['contract_duration_days'] / 30.44
        
        # Calculate monthly rent
        self.contracts_df['monthly_rent'] = self.contracts_df['contract_amount'] / self.contracts_df['contract_duration_months']
        
        # Calculate annualized rent
        self.contracts_df['annual_rent'] = self.contracts_df['monthly_rent'] * 12
        
        # Update market averages
        self._calculate_market_averages()
        
        # Calculate seasonal trends
        self._calculate_seasonal_trends()
    
    def _calculate_market_averages(self):
        """Calculate market averages by property type, area, and usage."""
        if self.contracts_df is None or len(self.contracts_df) == 0:
            self.market_averages = None
            return None
            
        # Overall averages
        overall_avg = {
            'avg_annual_rent': self.contracts_df['annual_rent'].mean(),
            'avg_monthly_rent': self.contracts_df['monthly_rent'].mean(),
            'avg_contract_duration': self.contracts_df['contract_duration_months'].mean(),
            'count': len(self.contracts_df)
        }
        
        # Averages by property type
        type_avg = self.contracts_df.groupby('ejari_property_type_en').agg({
            'annual_rent': 'mean',
            'monthly_rent': 'mean',
            'contract_duration_months': 'mean',
            'contract_amount': 'count'
        }).rename(columns={
            'annual_rent': 'avg_annual_rent',
            'monthly_rent': 'avg_monthly_rent',
            'contract_duration_months': 'avg_contract_duration',
            'contract_amount': 'count'
        }).to_dict('index')
        
        # Averages by area
        area_avg = self.contracts_df.groupby('area_name_en').agg({
            'annual_rent': 'mean',
            'monthly_rent': 'mean',
            'contract_duration_months': 'mean',
            'contract_amount': 'count'
        }).rename(columns={
            'annual_rent': 'avg_annual_rent',
            'monthly_rent': 'avg_monthly_rent',
            'contract_duration_months': 'avg_contract_duration',
            'contract_amount': 'count'
        }).to_dict('index')
        
        # Averages by usage
        usage_avg = self.contracts_df.groupby('property_usage_en').agg({
            'annual_rent': 'mean',
            'monthly_rent': 'mean',
            'contract_duration_months': 'mean',
            'contract_amount': 'count'
        }).rename(columns={
            'annual_rent': 'avg_annual_rent',
            'monthly_rent': 'avg_monthly_rent',
            'contract_duration_months': 'avg_contract_duration',
            'contract_amount': 'count'
        }).to_dict('index')
        
        # Averages by area and property type
        area_type_avg = self.contracts_df.groupby(['area_name_en', 'ejari_property_type_en']).agg({
            'annual_rent': 'mean',
            'monthly_rent': 'mean',
            'contract_amount': 'count'
        }).rename(columns={
            'annual_rent': 'avg_annual_rent',
            'monthly_rent': 'avg_monthly_rent',
            'contract_amount': 'count'
        })
        
        # Convert multi-index groupby result to nested dictionary
        area_type_dict = {}
        for (area, prop_type), row in area_type_avg.iterrows():
            if area not in area_type_dict:
                area_type_dict[area] = {}
            area_type_dict[area][prop_type] = row.to_dict()
        
        self.market_averages = {
            'overall': overall_avg,
            'by_property_type': type_avg,
            'by_area': area_avg,
            'by_usage': usage_avg,
            'by_area_and_type': area_type_dict
        }
        
        return self.market_averages
    
    def _calculate_seasonal_trends(self):
        """Calculate seasonal rental trends by month and quarter."""
        if self.contracts_df is None or len(self.contracts_df) == 0:
            self.seasonal_trends = None
            return None
        
        # Monthly trends
        monthly_trends = self.contracts_df.groupby('contract_month').agg({
            'monthly_rent': 'mean',
            'contract_amount': 'count'
        }).rename(columns={
            'monthly_rent': 'avg_monthly_rent',
            'contract_amount': 'contract_count'
        })
        
        # Calculate index compared to annual average (1.0 = average)
        monthly_trends['rent_index'] = monthly_trends['avg_monthly_rent'] / monthly_trends['avg_monthly_rent'].mean()
        
        # Quarterly trends
        quarterly_trends = self.contracts_df.groupby('contract_quarter').agg({
            'monthly_rent': 'mean',
            'contract_amount': 'count'
        }).rename(columns={
            'monthly_rent': 'avg_monthly_rent',
            'contract_amount': 'contract_count'
        })
        
        # Calculate index compared to annual average (1.0 = average)
        quarterly_trends['rent_index'] = quarterly_trends['avg_monthly_rent'] / quarterly_trends['avg_monthly_rent'].mean()
        
        # Yearly trends
        yearly_trends = self.contracts_df.groupby('contract_year').agg({
            'monthly_rent': 'mean',
            'contract_amount': 'count'
        }).rename(columns={
            'monthly_rent': 'avg_monthly_rent',
            'contract_amount': 'contract_count'
        })
        
        # Store the seasonal trends
        self.seasonal_trends = {
            'monthly': monthly_trends.to_dict('index'),
            'quarterly': quarterly_trends.to_dict('index'),
            'yearly': yearly_trends.to_dict('index')
        }
        
        return self.seasonal_trends
    
    def train_rental_model(self):
        """
        Train a machine learning model to predict rental prices.
        
        Returns:
            RandomForestRegressor: The trained rental model.
        """
        if self.contracts_df is None or len(self.contracts_df) < 5:  # Need sufficient data
            raise ValueError("Insufficient data to train the model. Need at least 5 contracts.")
            
        # Features and target
        X = self.contracts_df[['ejari_property_type_en', 'property_usage_en', 
                              'area_name_en', 'contract_month', 'contract_duration_months']]
        y = self.contracts_df['monthly_rent']
        
        # Preprocessing for categorical data
        categorical_features = ['ejari_property_type_en', 'property_usage_en', 'area_name_en']
        numeric_features = ['contract_month', 'contract_duration_months']
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        # Create and train the model
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        
        # Split the data for training and validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Fit the model
        model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Rental Model - Mean Squared Error: {mse:.2f}")
        print(f"Rental Model - R² Score: {r2:.2f}")
        
        self.rental_model = model
        return model
    
    def save_model(self, filepath="models/rent_prediction/trained_rent_model.pkl"):
        """
        Save the trained model to disk.
        
        Parameters:
            filepath (str): Path where to save the model
            
        Returns:
            str: Path to the saved model
        """
        if self.rental_model is None:
            raise ValueError("No trained model available. Please train the model first.")
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the model
        joblib.dump(self.rental_model, filepath)
        print(f"Model saved to {filepath}")
        return filepath
    
    def load_model(self, filepath="models/rent_prediction/trained_rent_model.pkl"):
        """
        Load a trained model from disk.
        
        Parameters:
            filepath (str): Path to the saved model
            
        Returns:
            object: The loaded model
        """
        self.rental_model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return self.rental_model
    
    def predict_rental_price(self, property_data, use_ml_model=False):
        """
        Predict the optimal rental price for a property.
        
        Parameters:
            property_data (dict): Property details including ejari_property_type_en, property_usage_en, 
                                 area_name_en, contract_month (optional).
            use_ml_model (bool): Whether to use the trained ML model for prediction.
            
        Returns:
            dict: Rental results including predicted rental price and market comparisons.
        """
        # Validate required fields
        required_fields = ['ejari_property_type_en', 'property_usage_en', 'area_name_en']
        missing_fields = [field for field in required_fields if field not in property_data]
        
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
            
        # Ensure we have market averages
        if self.market_averages is None:
            self._calculate_market_averages()
        
        if use_ml_model and self.rental_model:
            # Use the trained model for prediction
            # Default values for missing fields
            if 'contract_month' not in property_data:
                property_data['contract_month'] = datetime.now().month
            if 'contract_duration_months' not in property_data:
                property_data['contract_duration_months'] = 12  # Default to annual contract
                
            property_df = pd.DataFrame([property_data])
            X = property_df[['ejari_property_type_en', 'property_usage_en', 
                            'area_name_en', 'contract_month', 'contract_duration_months']]
            predicted_monthly_rent = self.rental_model.predict(X)[0]
            
            # Calculate annual rent
            predicted_annual_rent = predicted_monthly_rent * 12
        else:
            # Use the rule-based approach
            
            # Get relevant averages for property type and area
            property_type = property_data['ejari_property_type_en']
            area_name = property_data['area_name_en']
            property_usage = property_data['property_usage_en']
            
            # Try to get specific combination of area and property type
            try:
                area_type_avg = self.market_averages['by_area_and_type'][area_name][property_type]
                predicted_monthly_rent = area_type_avg['avg_monthly_rent']
            except (KeyError, TypeError):
                # If specific combination not found, blend area and type averages
                try:
                    type_avg = self.market_averages['by_property_type'][property_type]['avg_monthly_rent']
                except (KeyError, TypeError):
                    # If property type not found, use overall average
                    type_avg = self.market_averages['overall']['avg_monthly_rent']
                
                try:
                    area_avg = self.market_averages['by_area'][area_name]['avg_monthly_rent']
                except (KeyError, TypeError):
                    # If area not found, use overall average
                    area_avg = self.market_averages['overall']['avg_monthly_rent']
                
                # Blend area and type averages (70% area, 30% type)
                predicted_monthly_rent = (area_avg * 0.7) + (type_avg * 0.3)
            
            # Apply usage adjustment
            try:
                usage_multiplier = (self.market_averages['by_usage'][property_usage]['avg_monthly_rent'] / 
                                  self.market_averages['overall']['avg_monthly_rent'])
            except (KeyError, TypeError):
                usage_multiplier = 1.0
            
            predicted_monthly_rent *= usage_multiplier
            
            # Apply seasonal adjustment if month is provided
            if 'contract_month' in property_data and self.seasonal_trends:
                month = property_data['contract_month']
                try:
                    seasonal_index = self.seasonal_trends['monthly'][month]['rent_index']
                    predicted_monthly_rent *= seasonal_index
                except (KeyError, TypeError):
                    pass  # No adjustment if month not found
            
            # Calculate annual rent
            predicted_annual_rent = predicted_monthly_rent * 12
        
        # Get market comparisons
        market_comparisons = self._get_market_comparisons(property_data, predicted_monthly_rent)
        
        return {
            'predicted_monthly_rent': predicted_monthly_rent,
            'predicted_annual_rent': predicted_annual_rent,
            'market_comparisons': market_comparisons
        }
    
    def _get_market_comparisons(self, property_data, predicted_rent):
        """Generate market comparisons for the predicted rent."""
        comparisons = {}
        
        property_type = property_data['ejari_property_type_en']
        area_name = property_data['area_name_en']
        property_usage = property_data['property_usage_en']
        
        # Compare to overall market average
        try:
            overall_avg = self.market_averages['overall']['avg_monthly_rent']
            comparisons['overall_market'] = {
                'average_rent': overall_avg,
                'difference': predicted_rent - overall_avg,
                'difference_percent': ((predicted_rent / overall_avg) - 1) * 100
            }
        except (KeyError, TypeError):
            pass
        
        # Compare to property type average
        try:
            type_avg = self.market_averages['by_property_type'][property_type]['avg_monthly_rent']
            comparisons['property_type'] = {
                'type': property_type,
                'average_rent': type_avg,
                'difference': predicted_rent - type_avg,
                'difference_percent': ((predicted_rent / type_avg) - 1) * 100
            }
        except (KeyError, TypeError):
            pass
        
        # Compare to area average
        try:
            area_avg = self.market_averages['by_area'][area_name]['avg_monthly_rent']
            comparisons['area'] = {
                'area': area_name,
                'average_rent': area_avg,
                'difference': predicted_rent - area_avg,
                'difference_percent': ((predicted_rent / area_avg) - 1) * 100
            }
        except (KeyError, TypeError):
            pass
        
        # Compare to usage type average
        try:
            usage_avg = self.market_averages['by_usage'][property_usage]['avg_monthly_rent']
            comparisons['property_usage'] = {
                'usage': property_usage,
                'average_rent': usage_avg,
                'difference': predicted_rent - usage_avg,
                'difference_percent': ((predicted_rent / usage_avg) - 1) * 100
            }
        except (KeyError, TypeError):
            pass
        
        return comparisons
    
    def find_similar_contracts(self, property_data, count=5):
        """
        Find similar rental contracts to the specified property.
        
        Parameters:
            property_data (dict): Property details including ejari_property_type_en, area_name_en, etc.
            count (int): Number of similar contracts to return.
            
        Returns:
            pd.DataFrame: Similar contracts.
        """
        if self.contracts_df is None or len(self.contracts_df) == 0:
            return pd.DataFrame()
        
        # Filter by property type
        if 'ejari_property_type_en' in property_data:
            type_matches = self.contracts_df[
                self.contracts_df['ejari_property_type_en'] == property_data['ejari_property_type_en']
            ]
        else:
            type_matches = self.contracts_df
        
        # Further filter by area if provided
        if 'area_name_en' in property_data and len(type_matches) > 0:
            area_matches = type_matches[
                type_matches['area_name_en'] == property_data['area_name_en']
            ]
            
            # If no area matches, revert to type matches
            if len(area_matches) == 0:
                area_matches = type_matches
        else:
            area_matches = type_matches
        
        # Further filter by usage if provided
        if 'property_usage_en' in property_data and len(area_matches) > 0:
            usage_matches = area_matches[
                area_matches['property_usage_en'] == property_data['property_usage_en']
            ]
            
            # If no usage matches, revert to area matches
            if len(usage_matches) == 0:
                usage_matches = area_matches
        else:
            usage_matches = area_matches
        
        # Sort by contract_start_date (most recent first)
        sorted_matches = usage_matches.sort_values('contract_start_date', ascending=False)
        
        # Return top matches
        return sorted_matches.head(count)
    
    def analyze_rent_trends(self, area=None, property_type=None, time_period='yearly'):
        """
        Analyze rental price trends over time.
        
        Parameters:
            area (str, optional): Filter by area name.
            property_type (str, optional): Filter by property type.
            time_period (str): 'yearly', 'quarterly', or 'monthly'.
            
        Returns:
            pd.DataFrame: Rental trends over time.
        """
        if self.contracts_df is None or len(self.contracts_df) == 0:
            return pd.DataFrame()
        
        # Apply filters
        filtered_df = self.contracts_df.copy()
        
        if area:
            filtered_df = filtered_df[filtered_df['area_name_en'] == area]
            
        if property_type:
            filtered_df = filtered_df[filtered_df['ejari_property_type_en'] == property_type]
            
        if len(filtered_df) == 0:
            return pd.DataFrame()
            
        # Group by time period
        if time_period == 'yearly':
            time_col = 'contract_year'
        elif time_period == 'quarterly':
            time_col = ['contract_year', 'contract_quarter']
            # Create combined column for sorting
            filtered_df['year_quarter'] = filtered_df['contract_year'].astype(str) + '-Q' + filtered_df['contract_quarter'].astype(str)
            time_col = 'year_quarter'
        elif time_period == 'monthly':
            # Create combined column for sorting
            filtered_df['year_month'] = filtered_df['contract_year'].astype(str) + '-' + filtered_df['contract_month'].astype(str).str.zfill(2)
            time_col = 'year_month'
        else:
            raise ValueError("time_period must be 'yearly', 'quarterly', or 'monthly'")
            
        # Calculate trends
        trends = filtered_df.groupby(time_col).agg({
            'monthly_rent': ['mean', 'median', 'std', 'count'],
            'annual_rent': ['mean', 'median', 'std']
        })
        
        # Flatten multi-level columns
        trends.columns = ['_'.join(col).strip() for col in trends.columns.values]
        
        # Reset index for easier use
        trends = trends.reset_index()
        
        # Sort by time period
        trends = trends.sort_values(time_col)
        
        return trends
    
    def calculate_rent_volatility(self, area=None, property_type=None):
        """
        Calculate rent volatility (coefficient of variation) by area and property type.
        
        Parameters:
            area (str, optional): Filter by area name.
            property_type (str, optional): Filter by property type.
            
        Returns:
            dict: Volatility metrics.
        """
        if self.contracts_df is None or len(self.contracts_df) == 0:
            return {}
        
        # Apply filters
        filtered_df = self.contracts_df.copy()
        
        if area:
            filtered_df = filtered_df[filtered_df['area_name_en'] == area]
            
        if property_type:
            filtered_df = filtered_df[filtered_df['ejari_property_type_en'] == property_type]
            
        if len(filtered_df) < 2:  # Need at least 2 data points for volatility
            return {'error': 'Insufficient data for volatility calculation'}
            
        # Calculate overall volatility
        overall_mean = filtered_df['monthly_rent'].mean()
        overall_std = filtered_df['monthly_rent'].std()
        overall_cv = (overall_std / overall_mean) * 100  # Coefficient of variation in percentage
        
        # Calculate volatility by time period
        yearly_volatility = filtered_df.groupby('contract_year')['monthly_rent'].agg(['mean', 'std'])
        yearly_volatility['cv'] = (yearly_volatility['std'] / yearly_volatility['mean']) * 100
        
        # Result dictionary
        result = {
            'overall': {
                'mean': overall_mean,
                'std': overall_std,
                'coefficient_of_variation': overall_cv
            },
            'yearly': yearly_volatility.to_dict('index')
        }
        
        return result
    
    def export_market_report(self, output_path='rental_market_report.csv'):
        """
        Export rental market analysis to a CSV file.
        
        Parameters:
            output_path (str): Path to save the CSV file.
            
        Returns:
            str: Path to the saved file.
        """
        if self.contracts_df is None or len(self.contracts_df) == 0:
            print("No data available for export.")
            return None
            
        # Calculate aggregates by area and property type
        area_type_stats = self.contracts_df.groupby(['area_name_en', 'ejari_property_type_en']).agg({
            'monthly_rent': ['mean', 'median', 'std', 'count'],
            'annual_rent': ['mean', 'median'],
            'contract_duration_months': ['mean']
        })
        
        # Flatten multi-level columns
        area_type_stats.columns = ['_'.join(col).strip() for col in area_type_stats.columns.values]
        
        # Reset index for easier export
        area_type_stats = area_type_stats.reset_index()
        
        # Calculate coefficient of variation
        area_type_stats['rent_coefficient_of_variation'] = (area_type_stats['monthly_rent_std'] / area_type_stats['monthly_rent_mean']) * 100
        
        # Sort by count (descending) then area
        area_type_stats = area_type_stats.sort_values(['monthly_rent_count', 'area_name_en'], ascending=[False, True])
        
        # Export to CSV
        area_type_stats.to_csv(output_path, index=False)
        
        print(f"Market report exported to {output_path}")
        return output_path
"""
Rental Price Prediction Model

This module provides a machine learning model for predicting optimal rental prices
based on property features, and offering price optimization recommendations.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


class RentalPricePredictionModel:
    """
    A machine learning model for predicting optimal rental prices based on property features.
    
    This model uses XGBoost regression to predict rental prices, analyze feature importance,
    and provide price optimization recommendations.
    """
    
    def __init__(self):
        """Initialize the Rental Price Prediction Model."""
        self.model_pipeline = None
        self.trained_model = None
        self.feature_importance = None
        self.metrics = None
        self.categorical_cols = None
        self.numerical_cols = None
        self.target_col = 'AnnualRent'
        self.current_year = datetime.now().year
        self.data = None
        self.processed_data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self, data_path=None, dataframe=None):
        """
        Load rental property data from a CSV file or a pandas DataFrame.
        
        Parameters:
            data_path (str, optional): Path to the CSV file containing property data.
            dataframe (pd.DataFrame, optional): DataFrame containing property data.
            
        Returns:
            pd.DataFrame: The loaded property data.
        """
        if data_path:
            self.data = pd.read_csv(data_path)
        elif dataframe is not None:
            self.data = dataframe.copy()
        else:
            raise ValueError("Either data_path or dataframe must be provided")
            
        # Basic data info
        print(f"Dataset shape: {self.data.shape}")
        print(f"Columns: {self.data.columns.tolist()}")
        print(f"Missing values: {self.data.isnull().sum().sum()}")
        
        return self.data
    
    def preprocess_data(self):
        """
        Preprocess the dataset for model training.
        
        This includes handling missing values, feature engineering, and data transformation.
        
        Returns:
            pd.DataFrame: The preprocessed property data.
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
            
        # Create a copy of the dataframe
        processed_df = self.data.copy()
        
        # Handle missing values
        print("\nHandling missing values...")
        for col in processed_df.columns:
            missing = processed_df[col].isnull().sum()
            if missing > 0:
                print(f"Column {col} has {missing} missing values")
                if processed_df[col].dtype in ['int64', 'float64']:
                    # Fill numeric columns with median
                    processed_df[col] = processed_df[col].fillna(processed_df[col].median())
                else:
                    # Fill categorical columns with mode
                    processed_df[col] = processed_df[col].fillna(processed_df[col].mode()[0])
                    
        # Feature Engineering
        print("\nPerforming feature engineering...")
        
        # Create property age feature
        if 'YearBuilt' in processed_df.columns:
            processed_df['PropertyAge'] = self.current_year - processed_df['YearBuilt']
            
        # Process amenities (create binary features for each amenity)
        if 'Amenities' in processed_df.columns:
            # Convert amenities to lowercase for standardization
            processed_df['Amenities'] = processed_df['Amenities'].str.lower()
            
            # Get unique amenities across all properties
            all_amenities = []
            for amenities_list in processed_df['Amenities'].str.split(','):
                if isinstance(amenities_list, list):
                    for amenity in amenities_list:
                        amenity = amenity.strip()
                        if amenity and amenity not in all_amenities:
                            all_amenities.append(amenity)
                            
            # Create binary columns for each amenity
            for amenity in all_amenities:
                processed_df[f'Has_{amenity.replace(" ", "_")}'] = processed_df['Amenities'].str.contains(
                    amenity, case=False, na=False).astype(int)
                    
            # Create amenity count feature
            processed_df['AmenityCount'] = processed_df['Amenities'].str.split(',').apply(
                lambda x: len(x) if isinstance(x, list) else 0)
                
        # Calculate price per square foot
        if 'Size_sqft' in processed_df.columns and self.target_col in processed_df.columns:
            processed_df['PricePerSqFt'] = processed_df[self.target_col] / processed_df['Size_sqft']
            
        # Create bedroom to bathroom ratio
        if 'Bedrooms' in processed_df.columns and 'Bathrooms' in processed_df.columns:
            processed_df['BedroomToBathroomRatio'] = processed_df['Bedrooms'] / processed_df['Bathrooms']
            
        # Create property value to rent ratio (potential investment metric)
        if 'PropertyValuation' in processed_df.columns and self.target_col in processed_df.columns:
            processed_df['ValueToRentRatio'] = processed_df['PropertyValuation'] / processed_df[self.target_col]
            
        # Drop original columns that have been transformed
        columns_to_drop = ['Amenities']  # Add other columns as needed
        processed_df = processed_df.drop(columns=columns_to_drop, errors='ignore')
        
        print(f"Dataset after preprocessing: {processed_df.shape}")
        # Print new features
        new_features = set(processed_df.columns) - set(self.data.columns)
        print(f"New features created: {new_features}")
        
        self.processed_data = processed_df
        return processed_df
    
    def prepare_features_target(self, custom_target=None):
        """
        Prepare features and target variable, identify categorical and numerical columns.
        
        Parameters:
            custom_target (str, optional): Custom target column name.
            
        Returns:
            tuple: X, y, categorical_cols, numerical_cols
        """
        if self.processed_data is None:
            raise ValueError("No processed data available. Please preprocess data first.")
            
        target_col = custom_target if custom_target else self.target_col
        
        # Define target variable
        y = self.processed_data[target_col]
        
        # Define features (drop the target)
        X = self.processed_data.drop(columns=[target_col])
        
        # Identify categorical and numerical columns
        self.categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        print(f"Target variable: {target_col}")
        print(f"Number of features: {X.shape[1]}")
        print(f"Categorical features ({len(self.categorical_cols)}): {self.categorical_cols}")
        print(f"Numerical features ({len(self.numerical_cols)}): {self.numerical_cols}")
        
        # Store for later use
        self.X = X
        self.y = y
        
        return X, y, self.categorical_cols, self.numerical_cols
    
    def split_data(self, test_size=0.2, random_state=42):
        """
        Split the data into training and test sets.
        
        Parameters:
            test_size (float): Proportion of data to include in the test split.
            random_state (int): Random seed for reproducibility.
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        if self.X is None or self.y is None:
            raise ValueError("Features and target not prepared. Please run prepare_features_target first.")
            
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def build_model_pipeline(self):
        """
        Build preprocessing and modeling pipeline.
        
        Returns:
            Pipeline: The sklearn pipeline with preprocessor and XGBoost model.
        """
        if self.categorical_cols is None or self.numerical_cols is None:
            raise ValueError("Categorical and numerical columns not identified. Please run prepare_features_target first.")
            
        # Preprocessing for numerical data
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Preprocessing for categorical data
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Bundle preprocessing for numerical and categorical data
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_cols),
                ('cat', categorical_transformer, self.categorical_cols)
            ]
        )
        
        # Create the full pipeline with XGBoost model
        self.model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ))
        ])
        
        return self.model_pipeline
    
    def train_model(self, tune_hyperparameters=False):
        """
        Train the model, optionally with hyperparameter tuning.
        
        Parameters:
            tune_hyperparameters (bool): Whether to perform hyperparameter tuning.
            
        Returns:
            Pipeline: The trained model.
        """
        if self.model_pipeline is None:
            self.build_model_pipeline()
            
        if self.X_train is None or self.y_train is None:
            raise ValueError("Training data not available. Please run split_data first.")
            
        if tune_hyperparameters:
            print("Training model with hyperparameter tuning...")
            # Define hyperparameter grid
            param_grid = {
                'model__n_estimators': [50, 100, 200],
                'model__learning_rate': [0.01, 0.1, 0.2],
                'model__max_depth': [3, 6, 9],
                'model__min_child_weight': [1, 3, 5],
                'model__subsample': [0.8, 0.9, 1.0]
            }
            
            # Set up Grid Search with cross-validation
            grid_search = GridSearchCV(
                estimator=self.model_pipeline,
                param_grid=param_grid,
                cv=5,
                scoring='neg_mean_squared_error',
                verbose=1,
                n_jobs=-1
            )
            
            # Fit Grid Search
            grid_search.fit(self.X_train, self.y_train)
            
            # Print best parameters
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV score: {-grid_search.best_score_:.2f} MSE")
            
            # Update model with best estimator
            self.trained_model = grid_search.best_estimator_
        else:
            print("Training model with default parameters...")
            # Fit the model with default parameters
            self.model_pipeline.fit(self.X_train, self.y_train)
            self.trained_model = self.model_pipeline
            
        return self.trained_model
    
    def evaluate_model(self):
        """
        Evaluate the model on test data.
        
        Returns:
            dict: Dictionary with performance metrics.
        """
        if self.trained_model is None:
            raise ValueError("No trained model available. Please train the model first.")
            
        if self.X_test is None or self.y_test is None:
            raise ValueError("Test data not available. Please run split_data first.")
            
        print("Evaluating model performance...")
        # Make predictions
        y_pred = self.trained_model.predict(self.X_test)
        
        # Calculate metrics
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"Root Mean Squared Error: {rmse:.2f}")
        print(f"Mean Absolute Error: {mae:.2f}")
        print(f"R² Score: {r2:.4f}")
        
        # Store metrics
        self.metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'predictions': y_pred,
            'actual': self.y_test
        }
        
        return self.metrics
    
    def visualize_evaluation(self):
        """
        Visualize model evaluation results with predicted vs actual values and residuals.
        """
        if self.metrics is None:
            raise ValueError("No evaluation metrics available. Please evaluate the model first.")
            
        y_test = self.metrics['actual']
        y_pred = self.metrics['predictions']
        
        # Plot predicted vs actual values
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs Predicted Annual Rent')
        plt.tight_layout()
        plt.show()
        
        # Plot residuals
        residuals = y_test - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), colors='r', linestyles='--')
        plt.xlabel('Predicted')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.tight_layout()
        plt.show()
    
    def analyze_feature_importance(self):
        """
        Analyze and visualize feature importance.
        
        Returns:
            pd.DataFrame: DataFrame with feature importance scores.
        """
        if self.trained_model is None:
            raise ValueError("No trained model available. Please train the model first.")
            
        print("Analyzing feature importance...")
        try:
            # Get feature names after preprocessing
            preprocessor = self.trained_model.named_steps['preprocessor']
            feature_names = []
            
            # Get numerical feature names
            if self.numerical_cols:
                feature_names.extend(self.numerical_cols)
                
            # Get one-hot encoded feature names
            if self.categorical_cols:
                onehotencoder = preprocessor.transformers_[1][1].named_steps['onehot']
                categories = onehotencoder.categories_
                for i, category in enumerate(categories):
                    feature_names.extend([f"{self.categorical_cols[i]}_{cat}" for cat in category])
                    
            # Get feature importance
            xgb_model = self.trained_model.named_steps['model']
            importance = xgb_model.feature_importances_
            
            # Ensure feature_names and importance have the same length
            if len(feature_names) != len(importance):
                print(f"Warning: Feature names ({len(feature_names)}) and importance scores ({len(importance)}) have different lengths.")
                # Truncate to the shorter length
                min_len = min(len(feature_names), len(importance))
                feature_names = feature_names[:min_len]
                importance = importance[:min_len]
                
            # Create DataFrame for feature importance
            self.feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values(by='Importance', ascending=False)
            
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', data=self.feature_importance.head(20))
            plt.title('Top 20 Feature Importance')
            plt.tight_layout()
            plt.show()
            
            return self.feature_importance
        except Exception as e:
            print(f"Error analyzing feature importance: {e}")
            print("This may occur if the categorical features were not properly encoded or if the model structure is different.")
            return None
    
    def save_model(self, model_path='rental_price_prediction_model.pkl'):
        """
        Save the trained model to disk.
        
        Parameters:
            model_path (str): Path to save the model.
            
        Returns:
            str: Path to the saved model.
        """
        if self.trained_model is None:
            raise ValueError("No trained model available. Please train the model first.")
            
        print(f"Saving model to {model_path}...")
        joblib.dump(self.trained_model, model_path)
        print("Model successfully saved.")
        
        return model_path
    
    def load_saved_model(self, model_path='rental_price_prediction_model.pkl'):
        """
        Load a trained model from disk.
        
        Parameters:
            model_path (str): Path to the saved model.
            
        Returns:
            Pipeline: The loaded model.
        """
        print(f"Loading model from {model_path}...")
        self.trained_model = joblib.load(model_path)
        print("Model successfully loaded.")
        
        return self.trained_model
    
    def predict_rental_price(self, property_features):
        """
        Predict rental price for new properties.
        
        Parameters:
            property_features (pd.DataFrame): DataFrame containing property features.
            
        Returns:
            numpy.ndarray: Predicted rental prices.
        """
        if self.trained_model is None:
            raise ValueError("No trained model available. Please train or load a model first.")
            
        # Ensure property_features is a DataFrame
        if not isinstance(property_features, pd.DataFrame):
            raise TypeError("property_features must be a pandas DataFrame")
            
        # Make predictions
        predictions = self.trained_model.predict(property_features)
        
        return predictions
    
    def optimize_property_price(self, property_features, current_rent=None):
        """
        Optimize the price for a specific property and provide adjustment recommendations.
        
        Parameters:
            property_features (pd.DataFrame): DataFrame containing property features (one row).
            current_rent (float, optional): Current annual rent.
            
        Returns:
            dict: Dictionary with optimal price and adjustment recommendations.
        """
        # Ensure property_features is a DataFrame
        if not isinstance(property_features, pd.DataFrame):
            raise TypeError("property_features must be a pandas DataFrame")
            
        # Predict optimal price
        optimal_price = self.predict_rental_price(property_features)[0]
        
        # Prepare results
        results = {
            'optimal_annual_rent': round(optimal_price, 2),
            'key_factors': None,
            'adjustment_pct': None
        }
        
        # Calculate adjustment if current rent is provided
        if current_rent is not None:
            adjustment_pct = ((optimal_price / current_rent) - 1) * 100
            results['current_annual_rent'] = current_rent
            results['adjustment_pct'] = round(adjustment_pct, 2)
            
            # Add recommendation
            if abs(adjustment_pct) < 5:
                results['recommendation'] = "Current rent is well-aligned with market value"
            elif adjustment_pct < -10:
                results['recommendation'] = "Consider reducing rent significantly to meet market expectations"
            elif adjustment_pct < 0:
                results['recommendation'] = "Consider modest rent reduction to match market rates"
            elif adjustment_pct > 10:
                results['recommendation'] = "Significant rent increase opportunity based on property features"
            else:
                results['recommendation'] = "Consider modest rent increase to match market rates"
                
        return results
    
    def batch_optimize_prices(self, properties_df, current_rents=None):
        """
        Optimize prices for multiple properties.
        
        Parameters:
            properties_df (pd.DataFrame): DataFrame containing multiple properties.
            current_rents (pd.Series, optional): Series of current rents.
            
        Returns:
            pd.DataFrame: DataFrame with optimization results.
        """
        # Predict optimal prices for all properties
        optimal_prices = self.predict_rental_price(properties_df)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'optimal_annual_rent': optimal_prices.round(2)
        })
        
        # Add current rents and adjustments if provided
        if current_rents is not None:
            results_df['current_annual_rent'] = current_rents
            results_df['adjustment_pct'] = ((optimal_prices / current_rents) - 1) * 100
            results_df['adjustment_pct'] = results_df['adjustment_pct'].round(2)
            
            # Add recommendations
            conditions = [
                (results_df['adjustment_pct'].abs() < 5),
                (results_df['adjustment_pct'] < -10),
                (results_df['adjustment_pct'] < 0),
                (results_df['adjustment_pct'] > 10),
                (results_df['adjustment_pct'] >= 0)
            ]
            
            choices = [
                "Current rent aligned with market value",
                "Consider significant rent reduction",
                "Consider modest rent reduction",
                "Significant rent increase opportunity",
                "Consider modest rent increase"
            ]
            
            results_df['recommendation'] = np.select(conditions, choices, default="Review pricing")
            
        return results_df
    
    def analyze_market_trends(self):
        """
        Analyze market trends and provide insights.
        
        Returns:
            dict: Dictionary with market insights.
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
            
        df = self.data.copy()
        insights = {}
        
        # Average rent by area
        if 'Area' in df.columns and self.target_col in df.columns:
            area_rent = df.groupby('Area')[self.target_col].agg(['mean', 'median', 'count']).sort_values('mean', ascending=False)
            insights['top_expensive_areas'] = area_rent.head(5)
            insights['top_affordable_areas'] = area_rent.tail(5)
            
        # Average rent by property type
        if 'PropertyType' in df.columns and self.target_col in df.columns:
            type_rent = df.groupby('PropertyType')[self.target_col].agg(['mean', 'median', 'count']).sort_values('mean', ascending=False)
            insights['rent_by_property_type'] = type_rent
            
        # Price per square foot analysis
        if 'Size_sqft' in df.columns and self.target_col in df.columns:
            df['PricePerSqFt'] = df[self.target_col] / df['Size_sqft']
            if 'Area' in df.columns:
                psf_by_area = df.groupby('Area')['PricePerSqFt'].mean().sort_values(ascending=False)
                insights['price_per_sqft_by_area'] = psf_by_area.head(10)
                
        # Correlation between property age and rent
        if 'YearBuilt' in df.columns and self.target_col in df.columns:
            df['PropertyAge'] = self.current_year - df['YearBuilt']
            age_correlation = df['PropertyAge'].corr(df[self.target_col])
            insights['age_rent_correlation'] = age_correlation
            
        # Furnishing premium analysis
        if 'Furnishing' in df.columns and self.target_col in df.columns:
            furnishing_premium = df.groupby('Furnishing')[self.target_col].mean().sort_values(ascending=False)
            insights['furnishing_premium'] = furnishing_premium
            
        # Value to rent ratio analysis (investment potential)
        if 'PropertyValuation' in df.columns and self.target_col in df.columns:
            df['ValueToRentRatio'] = df['PropertyValuation'] / df[self.target_col]
            if 'Area' in df.columns:
                value_rent_ratio = df.groupby('Area')['ValueToRentRatio'].mean().sort_values()
                insights['best_investment_areas'] = value_rent_ratio.head(5)
                insights['worst_investment_areas'] = value_rent_ratio.tail(5)
                
        return insights
    
    def visualize_market_trends(self):
        """
        Visualize key market trends.
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
            
        df = self.data.copy()
        
        # Visualization 1: Rent distribution by property type
        if 'PropertyType' in df.columns and self.target_col in df.columns:
            plt.figure(figsize=(12, 8))
            for prop_type in df['PropertyType'].unique():
                sns.kdeplot(df[df['PropertyType'] == prop_type][self.target_col], label=prop_type)
            plt.title('Rent Distribution by Property Type')
            plt.xlabel('Annual Rent')
            plt.legend()
            plt.tight_layout()
            plt.show()
            
        # Visualization 2: Price per square foot vs property size
        if 'Size_sqft' in df.columns and self.target_col in df.columns:
            df['PricePerSqFt'] = df[self.target_col] / df['Size_sqft']
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x='Size_sqft', y='PricePerSqFt', data=df, 
                           hue='PropertyType' if 'PropertyType' in df.columns else None)
            plt.title('Price per Square Foot vs Property Size')
            plt.xlabel('Size (sq ft)')
            plt.ylabel('Price per Square Foot')
            plt.tight_layout()
            plt.show()
            
        # Visualization 3: Top areas by average rent
        if 'Area' in df.columns and self.target_col in df.columns:
            top_areas = df.groupby('Area')[self.target_col].mean().nlargest(10).reset_index()
            plt.figure(figsize=(12, 6))
            sns.barplot(x=self.target_col, y='Area', data=top_areas)
            plt.title('Top 10 Areas by Average Rent')
            plt.xlabel('Average Annual Rent')
            plt.tight_layout()
            plt.show()
            
        # Visualization 4: Bedrooms vs rent
        if 'Bedrooms' in df.columns and self.target_col in df.columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='Bedrooms', y=self.target_col, data=df)
            plt.title('Annual Rent by Number of Bedrooms')
            plt.tight_layout()
            plt.show()
    
    def generate_market_report(self, output_path='rental_market_report.csv'):
        """
        Generate a comprehensive market report and export to CSV.
        
        Parameters:
            output_path (str): Path to save the CSV file.
            
        Returns:
            str: Path to the saved file.
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
            
        df = self.data.copy()
        report_data = []
        
        # Area analysis
        if 'Area' in df.columns and self.target_col in df.columns:
            area_stats = df.groupby('Area').agg({
                self.target_col: ['mean', 'median', 'std', 'count'],
                'Size_sqft': ['mean'] if 'Size_sqft' in df.columns else None,
                'Bedrooms': ['mean'] if 'Bedrooms' in df.columns else None
            }).reset_index()
            
            # Flatten multi-level columns
            area_stats.columns = [' '.join(col).strip() if isinstance(col, tuple) else col for col in area_stats.columns]
            
            # Calculate price per sqft
            if 'Size_sqft mean' in area_stats.columns and f'{self.target_col} mean' in area_stats.columns:
                area_stats['Price_Per_SqFt'] = area_stats[f'{self.target_col} mean'] / area_stats['Size_sqft mean']
                
            # Sort by average rent (descending)
            area_stats = area_stats.sort_values(f'{self.target_col} mean', ascending=False)
            
            # Export to CSV
            area_stats.to_csv(output_path, index=False)
            print(f"Market report exported to {output_path}")
            
        return output_path


# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = pd.DataFrame({
        'Area': ['Downtown', 'Marina', 'Suburbs', 'Downtown', 'Marina'],
        'PropertyType': ['Apartment', 'Villa', 'Townhouse', 'Apartment', 'Villa'],
        'Size_sqft': [1200, 2500, 1800, 950, 3000],
        'Bedrooms': [2, 4, 3, 1, 5],
        'Bathrooms': [2, 4, 2.5, 1, 5.5],
        'Furnishing': ['Furnished', 'Unfurnished', 'Semi-Furnished', 'Furnished', 'Furnished'],
        'Amenities': ['pool,gym,parking', 'garden,parking,security', 'parking,security', 'gym', 'pool,gym,security,garden'],
        'YearBuilt': [2010, 2005, 2015, 2018, 2000],
        'PropertyValuation': [1500000, 3500000, 2200000, 1000000, 4000000],
        'AnnualRent': [120000, 180000, 150000, 95000, 220000]
    })
    
    # Initialize model
    model = RentalPricePredictionModel()
    
    # Load data
    model.load_data(dataframe=sample_data)
    
    # Preprocess data
    model.preprocess_data()
    
    # Prepare features and target
    X, y, cat_cols, num_cols = model.prepare_features_target()
    
    # Split data
    X_train, X_test, y_train, y_test = model.split_data()
    
    # Build and train model
    model.build_model_pipeline()
    model.train_model()
    
    # Evaluate model
    metrics = model.evaluate_model()
    
    # Example prediction
    test_property = X_test.iloc[0:1]
    predicted_rent = model.predict_rental_price(test_property)
    print(f"Predicted annual rent: {predicted_rent[0]:.2f}")
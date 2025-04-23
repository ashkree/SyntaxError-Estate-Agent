import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns


# Example usage to demonstrate the model
def demonstrate_model():
    """Example usage of the Real Estate Investment Model."""
    # Create sample data
    sample_properties = [
        {
            'area_name': 'Downtown',
            'size_sqft': 1200,
            'bed': 2,
            'bath': 2,
            'amenities': ['Pool', 'Gym', 'Parking', 'Security'],
            'annual_rental_price': 120000,
            'property_price': 1500000,
            'property_type': 'Apt'
        },
        {
            'area_name': 'Marina',
            'size_sqft': 1500,
            'bed': 3,
            'bath': 2.5,
            'amenities': ['Pool', 'Gym', 'Parking', 'Security', 'Sea View', 'Balcony'],
            'annual_rental_price': 160000,
            'property_price': 2200000,
            'property_type': 'Apt'
        },
        {
            'area_name': 'Suburbs',
            'size_sqft': 2200,
            'bed': 4,
            'bath': 3,
            'amenities': ['Garden', 'Parking', 'Security'],
            'annual_rental_price': 180000,
            'property_price': 2800000,
            'property_type': 'Villa'
        },
        {
            'area_name': 'Downtown',
            'size_sqft': 950,
            'bed': 1,
            'bath': 1.5,
            'amenities': ['Pool', 'Gym', 'Parking'],
            'annual_rental_price': 95000,
            'property_price': 1100000,
            'property_type': 'Apt'
        },
        {
            'area_name': 'Marina',
            'size_sqft': 2800,
            'bed': 4,
            'bath': 4.5,
            'amenities': ['Pool', 'Gym', 'Parking', 'Security', 'Sea View', 'Smart Home', 'Concierge'],
            'annual_rental_price': 320000,
            'property_price': 4500000,
            'property_type': 'Villa'
        },
        {
            'area_name': 'Business Bay',
            'size_sqft': 1800,
            'bed': 3,
            'bath': 3,
            'amenities': ['Pool', 'Gym', 'Parking', 'Security', 'Balcony'],
            'annual_rental_price': 170000,
            'property_price': 2400000,
            'property_type': 'Apt'
        },
        {
            'area_name': 'Palm Jumeirah',
            'size_sqft': 3200,
            'bed': 4,
            'bath': 5,
            'amenities': ['Pool', 'Gym', 'Parking', 'Security', 'Sea View', 'Smart Home', 'Private Beach', 'Garden'],
            'annual_rental_price': 450000,
            'property_price': 7500000,
            'property_type': 'Villa'
        }
    ]
    
    # Create DataFrame
    property_df = pd.DataFrame(sample_properties)
    
    # Initialize model
    model = RealEstateInvestmentModel()
    
    # Load data
    model.load_data(dataframe=property_df)
    
    # Display basic statistics
    print("\n--- Property Dataset Overview ---")
    print(model.properties_df[['area_name', 'property_type', 'size_sqft', 'bed', 'bath', 'property_price', 'annual_rental_price']].head())
    
    # Display derived metrics
    print("\n--- Derived Property Metrics ---")
    print(model.properties_df[['area_name', 'property_type', 'price_per_sqft', 'rental_yield', 'price_to_rent_ratio']].head())
    
    # Calculate market averages
    averages = model._calculate_market_averages()
    print("\n--- Market Averages ---")
    print(f"Overall Avg Price per Sqft: {averages['overall']['avg_price_per_sqft']:.2f}")
    print(f"Overall Avg Rental Yield: {averages['overall']['avg_rental_yield']:.2f}%")
    
    # Analyze a specific property
    print("\n--- Property Analysis ---")
    property_idx = 0  # First property
    valuation = model.predict_fair_market_value(property_idx)
    rental = model.predict_optimal_rental(property_idx)
    investment_score = model.calculate_investment_score(property_idx)
    
    property_name = f"{model.properties_df.iloc[property_idx]['area_name']} {model.properties_df.iloc[property_idx]['property_type']}"
    
    print(f"Property: {property_name}")
    print(f"Current Price: {valuation['current_value']:.2f}")
    print(f"Fair Market Value: {valuation['fair_market_value']:.2f}")
    print(f"Undervalued: {valuation['undervalued']}")
    print(f"Value Difference: {valuation['value_difference']:.2f} ({valuation['value_difference_percent']:.2f}%)")
    
    print(f"\nCurrent Annual Rent: {rental['current_annual_rental']:.2f}")
    print(f"Optimal Annual Rent: {rental['optimal_annual_rental']:.2f}")
    print(f"Rental Upside: {rental['rental_upside']}")
    print(f"Rental Difference: {rental['rental_difference']:.2f} ({rental['rental_difference_percent']:.2f}%)")
    
    print(f"\nOverall Investment Score: {investment_score['overall_score']:.2f}/100")
    print(f"Valuation Score: {investment_score['valuation_score']:.2f}/40")
    print(f"Yield Score: {investment_score['yield_score']:.2f}/30")
    print(f"Rental Upside Score: {investment_score['rental_upside_score']:.2f}/20")
    print(f"Location Score: {investment_score['location_score']:.2f}/10")
    
    # Find top investment opportunities
    print("\n--- Top Investment Opportunities ---")
    top_opps = model.find_top_opportunities(count=3)
    for i, opp in enumerate(top_opps, 1):
        prop = opp['property']
        score = opp['score']
        print(f"{i}. {prop['area_name']} {prop['property_type']} - Score: {score['overall_score']:.2f}/100")
        print(f"   Size: {prop['size_sqft']} sqft, Beds: {prop['bed']}, Baths: {prop['bath']}")
        print(f"   Price: {prop['property_price']}, Annual Rent: {prop['annual_rental_price']}")
        print(f"   Rental Yield: {prop['rental_yield']:.2f}%")
        if score['details']['valuation']['undervalued']:
            print(f"   Undervalued by: {score['details']['valuation']['value_difference_percent']:.2f}%")
        print()
    
    # Find highest yield properties
    print("\n--- Highest Yield Properties ---")
    high_yield = model.find_highest_yield_properties(count=3)
    for i, (_, prop) in enumerate(high_yield.iterrows(), 1):
        print(f"{i}. {prop['area_name']} {prop['property_type']} - Yield: {prop['rental_yield']:.2f}%")
        print(f"   Size: {prop['size_sqft']} sqft, Price: {prop['property_price']}")
        print()
    
    # Find undervalued properties
    print("\n--- Most Undervalued Properties ---")
    undervalued = model.find_undervalued_properties(count=3)
    for i, prop_data in enumerate(undervalued, 1):
        prop = prop_data['property']
        val = prop_data['valuation']
        print(f"{i}. {prop['area_name']} {prop['property_type']}")
        print(f"   Current Price: {val['current_value']}")
        print(f"   Fair Market Value: {val['fair_market_value']:.2f}")
        print(f"   Undervalued by: {val['value_difference_percent']:.2f}%")
        print()
    
    # Visualize market overview
    model.visualize_market_overview()
    
    # Visualize investment opportunities
    model.visualize_investment_opportunities(top_opps)
    
    # Export investment report
    model.export_investment_report(top_opps, 'investment_opportunities.csv')
    
    return model


class RealEstateInvestmentModel:
    """
    A comprehensive model for analyzing real estate investment opportunities.
    
    This model analyzes properties based on various features to identify investment
    opportunities, predict fair market values, optimize rental prices, and calculate
    investment scores.
    """
    
    def __init__(self):
        """Initialize the Real Estate Investment Model."""
        self.properties_df = None
        self.valuation_model = None
        self.rental_model = None
        self.market_averages = None
        self.amenity_value_map = {
            'pool': 0.05,           # 5% premium
            'gym': 0.03,            # 3% premium
            'parking': 0.04,        # 4% premium
            'security': 0.03,       # 3% premium
            'balcony': 0.02,        # 2% premium
            'garden': 0.04,         # 4% premium
            'furnished': 0.08,      # 8% premium
            'sea view': 0.10,       # 10% premium
            'smart home': 0.05,     # 5% premium
            'concierge': 0.03       # 3% premium
        }
        
    def load_data(self, data_path=None, dataframe=None):
        """
        Load property data from a CSV file or a pandas DataFrame.
        
        Parameters:
            data_path (str, optional): Path to the CSV file containing property data.
            dataframe (pd.DataFrame, optional): DataFrame containing property data.
            
        Returns:
            pd.DataFrame: The loaded property data.
        """
        if data_path:
            self.properties_df = pd.read_csv(data_path)
        elif dataframe is not None:
            self.properties_df = dataframe.copy()
        else:
            raise ValueError("Either data_path or dataframe must be provided")
            
        # Validate required columns
        required_columns = ['area_name', 'size_sqft', 'bed', 'bath', 
                           'annual_rental_price', 'property_price', 'property_type']
        missing_columns = [col for col in required_columns if col not in self.properties_df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        # Process amenities column if it exists
        if 'amenities' in self.properties_df.columns:
            if isinstance(self.properties_df['amenities'][0], str):
                # Convert string representation of list to actual list
                self.properties_df['amenities'] = self.properties_df['amenities'].apply(
                    lambda x: x.strip('[]').replace("'", "").split(', ') if isinstance(x, str) else [])
        else:
            self.properties_df['amenities'] = [[] for _ in range(len(self.properties_df))]
            
        # Calculate derived metrics
        self._calculate_derived_metrics()
        
        return self.properties_df
    
    def add_property(self, property_data):
        """
        Add a new property to the dataset.
        
        Parameters:
            property_data (dict): Property details including area_name, size_sqft, bed, bath,
                                 annual_rental_price, property_price, property_type, amenities.
                                 
        Returns:
            pd.DataFrame: Updated properties dataframe.
        """
        # Validate required fields
        required_fields = ['area_name', 'size_sqft', 'bed', 'bath', 
                          'annual_rental_price', 'property_price', 'property_type']
        
        missing_fields = [field for field in required_fields if field not in property_data]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
            
        # Ensure amenities is in the right format
        if 'amenities' not in property_data:
            property_data['amenities'] = []
        elif isinstance(property_data['amenities'], str):
            property_data['amenities'] = property_data['amenities'].split(',')
            
        # Create DataFrame from the new property
        new_property_df = pd.DataFrame([property_data])
        
        # Add to existing data or create new DataFrame
        if self.properties_df is None:
            self.properties_df = new_property_df
        else:
            self.properties_df = pd.concat([self.properties_df, new_property_df], ignore_index=True)
            
        # Recalculate derived metrics
        self._calculate_derived_metrics()
        
        return self.properties_df
    
    def _calculate_derived_metrics(self):
        """Calculate derived metrics for all properties."""
        if self.properties_df is None or len(self.properties_df) == 0:
            return
            
        # Calculate basic financial metrics
        self.properties_df['price_per_sqft'] = self.properties_df['property_price'] / self.properties_df['size_sqft']
        self.properties_df['rental_yield'] = (self.properties_df['annual_rental_price'] / self.properties_df['property_price']) * 100
        self.properties_df['price_to_rent_ratio'] = self.properties_df['property_price'] / self.properties_df['annual_rental_price']
        self.properties_df['bed_bath_ratio'] = self.properties_df['bed'] / self.properties_df['bath']
        
        # Calculate amenity score
        self.properties_df['amenity_score'] = self.properties_df['amenities'].apply(self._calculate_amenity_score)
        
        # Update market averages
        self._calculate_market_averages()
    
    def _calculate_amenity_score(self, amenities_list):
        """
        Calculate score based on amenities.
        
        Parameters:
            amenities_list (list): List of amenities for a property.
            
        Returns:
            float: The calculated amenity score.
        """
        if not amenities_list:
            return 0
            
        score = 0
        for amenity in amenities_list:
            amenity_lower = str(amenity).lower().strip()
            if amenity_lower in self.amenity_value_map:
                score += self.amenity_value_map[amenity_lower]
            else:
                # Default value for unlisted amenities
                score += 0.01
                
        return score
    
    def _calculate_market_averages(self):
        """Calculate market averages by property type and area."""
        if self.properties_df is None or len(self.properties_df) == 0:
            self.market_averages = None
            return None
            
        # Overall averages
        overall_avg = {
            'avg_price_per_sqft': self.properties_df['price_per_sqft'].mean(),
            'avg_rental_yield': self.properties_df['rental_yield'].mean(),
            'avg_price_to_rent_ratio': self.properties_df['price_to_rent_ratio'].mean(),
            'count': len(self.properties_df)
        }
        
        # Averages by property type
        type_avg = self.properties_df.groupby('property_type').agg({
            'price_per_sqft': 'mean',
            'rental_yield': 'mean',
            'price_to_rent_ratio': 'mean',
            'property_price': 'count'
        }).rename(columns={
            'price_per_sqft': 'avg_price_per_sqft',
            'rental_yield': 'avg_rental_yield',
            'price_to_rent_ratio': 'avg_price_to_rent_ratio',
            'property_price': 'count'
        }).to_dict('index')
        
        # Averages by area
        area_avg = self.properties_df.groupby('area_name').agg({
            'price_per_sqft': 'mean',
            'rental_yield': 'mean',
            'price_to_rent_ratio': 'mean',
            'property_price': 'count'
        }).rename(columns={
            'price_per_sqft': 'avg_price_per_sqft',
            'rental_yield': 'avg_rental_yield',
            'price_to_rent_ratio': 'avg_price_to_rent_ratio',
            'property_price': 'count'
        }).to_dict('index')
        
        self.market_averages = {
            'overall': overall_avg,
            'by_property_type': type_avg,
            'by_area': area_avg
        }
        
        return self.market_averages
    
    def train_valuation_model(self):
        """
        Train a machine learning model to predict property valuations.
        
        Returns:
            RandomForestRegressor: The trained valuation model.
        """
        if self.properties_df is None or len(self.properties_df) < 5:  # Need sufficient data
            raise ValueError("Insufficient data to train the model. Need at least 5 properties.")
            
        # Features and target
        X = self.properties_df[['area_name', 'size_sqft', 'bed', 'bath', 'property_type', 'amenity_score']]
        y = self.properties_df['property_price']
        
        # Preprocessing for categorical data
        categorical_features = ['area_name', 'property_type']
        numeric_features = ['size_sqft', 'bed', 'bath', 'amenity_score']
        
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
        
        print(f"Valuation Model - Mean Squared Error: {mse:.2f}")
        print(f"Valuation Model - R² Score: {r2:.2f}")
        
        self.valuation_model = model
        return model
    
    def train_rental_model(self):
        """
        Train a machine learning model to predict optimal rental prices.
        
        Returns:
            RandomForestRegressor: The trained rental model.
        """
        if self.properties_df is None or len(self.properties_df) < 5:  # Need sufficient data
            raise ValueError("Insufficient data to train the model. Need at least 5 properties.")
            
        # Features and target
        X = self.properties_df[['area_name', 'size_sqft', 'bed', 'bath', 'property_type', 'amenity_score']]
        y = self.properties_df['annual_rental_price']
        
        # Preprocessing for categorical data
        categorical_features = ['area_name', 'property_type']
        numeric_features = ['size_sqft', 'bed', 'bath', 'amenity_score']
        
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
    
    def predict_fair_market_value(self, property_data, use_ml_model=False):
        """
        Predict the fair market value for a property.
        
        Parameters:
            property_data (dict or int): Property details or index in the properties_df.
            use_ml_model (bool): Whether to use the trained ML model for prediction.
            
        Returns:
            dict: Valuation results including fair market value and comparison metrics.
        """
        # Get property as a dictionary
        if isinstance(property_data, int):
            # Get property by index
            if self.properties_df is None or property_data >= len(self.properties_df):
                raise ValueError(f"Invalid property index: {property_data}")
            property_dict = self.properties_df.iloc[property_data].to_dict()
        else:
            # Use provided property data
            property_dict = property_data.copy()
            
        # Ensure we have market averages
        if self.market_averages is None:
            self._calculate_market_averages()
            
        if use_ml_model and self.valuation_model:
            # Use the trained model for prediction
            property_df = pd.DataFrame([property_dict])
            X = property_df[['area_name', 'size_sqft', 'bed', 'bath', 'property_type', 'amenity_score']]
            fair_market_value = self.valuation_model.predict(X)[0]
        else:
            # Use the rule-based approach
            
            # Get relevant averages for property type and area
            property_type = property_dict['property_type']
            area_name = property_dict['area_name']
            
            type_avg = self.market_averages['by_property_type'].get(property_type, self.market_averages['overall'])
            area_avg = self.market_averages['by_area'].get(area_name, self.market_averages['overall'])
            
            # Calculate base value from area and property type average price per sqft
            base_type_value = type_avg['avg_price_per_sqft'] * property_dict['size_sqft']
            base_area_value = area_avg['avg_price_per_sqft'] * property_dict['size_sqft']
            
            # Average the two approaches with more weight on area-based valuation
            base_value = (base_area_value * 0.7) + (base_type_value * 0.3)
            
            # Apply adjustments based on beds and baths
            bed_adjustment = property_dict['bed'] * (property_dict['size_sqft'] * 0.02)  # 2% of sqft value per bedroom
            bath_adjustment = property_dict['bath'] * (property_dict['size_sqft'] * 0.015)  # 1.5% of sqft value per bathroom
            
            # Apply amenity premium if amenity score exists
            amenity_premium = 0
            if 'amenity_score' in property_dict:
                amenity_premium = base_value * property_dict['amenity_score']
            elif 'amenities' in property_dict:
                amenity_score = self._calculate_amenity_score(property_dict['amenities'])
                amenity_premium = base_value * amenity_score
                
            # Final fair market value
            fair_market_value = base_value + bed_adjustment + bath_adjustment + amenity_premium
            
        # Get current property price
        property_price = property_dict['property_price']
        
        # Calculate comparison metrics
        undervalued = fair_market_value > property_price
        value_difference = fair_market_value - property_price
        value_difference_percent = (value_difference / property_price) * 100
            
        return {
            'fair_market_value': fair_market_value,
            'current_value': property_price,
            'undervalued': undervalued,
            'value_difference': value_difference,
            'value_difference_percent': value_difference_percent
        }
    
    def predict_optimal_rental(self, property_data, use_ml_model=False):
        """
        Predict the optimal rental price for a property.
        
        Parameters:
            property_data (dict or int): Property details or index in the properties_df.
            use_ml_model (bool): Whether to use the trained ML model for prediction.
            
        Returns:
            dict: Rental results including optimal rental price and comparison metrics.
        """
        # Get property as a dictionary
        if isinstance(property_data, int):
            # Get property by index
            if self.properties_df is None or property_data >= len(self.properties_df):
                raise ValueError(f"Invalid property index: {property_data}")
            property_dict = self.properties_df.iloc[property_data].to_dict()
        else:
            # Use provided property data
            property_dict = property_data.copy()
            
        # Ensure we have market averages
        if self.market_averages is None:
            self._calculate_market_averages()
            
        if use_ml_model and self.rental_model:
            # Use the trained model for prediction
            property_df = pd.DataFrame([property_dict])
            X = property_df[['area_name', 'size_sqft', 'bed', 'bath', 'property_type', 'amenity_score']]
            optimal_rental = self.rental_model.predict(X)[0]
        else:
            # Use the rule-based approach
            
            # Get relevant averages for property type and area
            property_type = property_dict['property_type']
            area_name = property_dict['area_name']
            
            type_avg = self.market_averages['by_property_type'].get(property_type, self.market_averages['overall'])
            area_avg = self.market_averages['by_area'].get(area_name, self.market_averages['overall'])
            
            # Use price-to-rent ratio to calculate optimal rental
            type_based_rental = property_dict['property_price'] / type_avg['avg_price_to_rent_ratio']
            area_based_rental = property_dict['property_price'] / area_avg['avg_price_to_rent_ratio']
            
            # Average the two approaches with more weight on area-based calculation
            base_rental = (area_based_rental * 0.6) + (type_based_rental * 0.4)
            
            # Apply amenity premium if amenity score exists
            amenity_premium = 0
            if 'amenity_score' in property_dict:
                amenity_premium = base_rental * property_dict['amenity_score']
            elif 'amenities' in property_dict:
                amenity_score = self._calculate_amenity_score(property_dict['amenities'])
                amenity_premium = base_rental * amenity_score
                
            # Final optimal rental
            optimal_rental = base_rental + amenity_premium
            
        # Get current rental price
        annual_rental_price = property_dict['annual_rental_price']
        
        # Calculate comparison metrics
        rental_upside = optimal_rental > annual_rental_price
        rental_difference = optimal_rental - annual_rental_price
        rental_difference_percent = (rental_difference / annual_rental_price) * 100
            
        return {
            'optimal_annual_rental': optimal_rental,
            'current_annual_rental': annual_rental_price,
            'rental_upside': rental_upside,
            'rental_difference': rental_difference,
            'rental_difference_percent': rental_difference_percent
        }
    
    def calculate_investment_score(self, property_data, use_ml_models=False):
        """
        Calculate an investment score for a property (0-100).
        Higher score = better investment opportunity.
        
        Parameters:
            property_data (dict or int): Property details or index in the properties_df.
            use_ml_models (bool): Whether to use trained ML models for predictions.
            
        Returns:
            dict: Investment score and component scores.
        """
        # Get property as a dictionary
        if isinstance(property_data, int):
            # Get property by index
            if self.properties_df is None or property_data >= len(self.properties_df):
                raise ValueError(f"Invalid property index: {property_data}")
            property_dict = self.properties_df.iloc[property_data].to_dict()
            property_idx = property_data
        else:
            # Add property if it's not already in the dataframe
            property_dict = property_data.copy()
            self.add_property(property_dict)
            property_idx = len(self.properties_df) - 1
            
        # Calculate valuation and rental metrics
        valuation = self.predict_fair_market_value(property_idx, use_ml_model=use_ml_models)
        rental = self.predict_optimal_rental(property_idx, use_ml_model=use_ml_models)
        
        # Get property metrics
        rental_yield = self.properties_df.loc[property_idx, 'rental_yield']
        area_name = self.properties_df.loc[property_idx, 'area_name']
        
        # Investment score components:
        
        # 1. Valuation component (40% of total score)
        # Higher undervaluation = higher score
        valuation_score = 0
        if valuation['undervalued']:
            # Cap the valuation difference at 30% for max score
            valuation_score = min(abs(valuation['value_difference_percent']) / 30 * 40, 40)
        
        # 2. Rental yield component (30% of total score)
        # Compare to market average
        yield_comparison = rental_yield - self.market_averages['overall']['avg_rental_yield']
        # Scale: 3% above market average yields full points
        yield_score = min(max(yield_comparison, 0) / 3 * 30, 30)
        
        # 3. Rental upside component (20% of total score)
        # Higher rental upside = higher score
        rental_upside_score = 0
        if rental['rental_upside']:
            # Cap the rental upside at 20% for max score
            rental_upside_score = min(abs(rental['rental_difference_percent']) / 20 * 20, 20)
        
        # 4. Location premium (10% of total score)
        # If area average price per sqft is above overall average
        area_avg_price_per_sqft = self.market_averages['by_area'].get(area_name, {'avg_price_per_sqft': 0})['avg_price_per_sqft']
        overall_avg_price_per_sqft = self.market_averages['overall']['avg_price_per_sqft']
        location_premium = (area_avg_price_per_sqft / overall_avg_price_per_sqft) - 1 if overall_avg_price_per_sqft > 0 else 0
        # Scale: 20% premium location yields full points
        location_score = min(max(location_premium, 0) / 0.2 * 10, 10)
        
        # Total investment score (0-100)
        investment_score = valuation_score + yield_score + rental_upside_score + location_score
        
        return {
            'overall_score': investment_score,
            'valuation_score': valuation_score,
            'yield_score': yield_score,
            'rental_upside_score': rental_upside_score,
            'location_score': location_score,
            'details': {
                'valuation': valuation,
                'rental': rental,
                'rental_yield': rental_yield,
                'location_premium': location_premium
            }
        }
    
    def find_top_opportunities(self, count=5, use_ml_models=False):
        """
        Find top investment opportunities.
        
        Parameters:
            count (int): Number of top opportunities to return.
            use_ml_models (bool): Whether to use trained ML models for predictions.
            
        Returns:
            List[dict]: Top investment opportunities.
        """
        if self.properties_df is None or len(self.properties_df) == 0:
            return []
            
        # Calculate investment scores for all properties
        scored_properties = []
        for idx in range(len(self.properties_df)):
            score = self.calculate_investment_score(idx, use_ml_models=use_ml_models)
            scored_properties.append({
                'property': self.properties_df.iloc[idx].to_dict(),
                'score': score
            })
            
        # Sort by overall score (descending)
        scored_properties.sort(key=lambda x: x['score']['overall_score'], reverse=True)
        
        # Return top properties
        return scored_properties[:count]
    
    def find_highest_yield_properties(self, count=5):
        """
        Find properties with highest rental yield.
        
        Parameters:
            count (int): Number of top properties to return.
            
        Returns:
            pd.DataFrame: Top yield properties.
        """
        if self.properties_df is None or len(self.properties_df) == 0:
            return pd.DataFrame()
            
        # Sort by rental yield (descending)
        sorted_df = self.properties_df.sort_values('rental_yield', ascending=False)
        
        # Return top properties
        return sorted_df.head(count)
    
    def find_undervalued_properties(self, count=5, use_ml_model=False):
        """
        Find undervalued properties.
        
        Parameters:
            count (int): Number of top undervalued properties to return.
            use_ml_model (bool): Whether to use the trained ML model for predictions.
            
        Returns:
            List[dict]: Top undervalued properties.
        """
        if self.properties_df is None or len(self.properties_df) == 0:
            return []
            
        # Calculate valuation for all properties
        valuations = []
        for idx in range(len(self.properties_df)):
            valuation = self.predict_fair_market_value(idx, use_ml_model=use_ml_model)
            valuations.append({
                'property': self.properties_df.iloc[idx].to_dict(),
                'valuation': valuation
            })
            
        # Filter for undervalued properties
        undervalued_properties = [item for item in valuations if item['valuation']['undervalued']]
        
        # Sort by value difference percentage (descending)
        undervalued_properties.sort(
            key=lambda x: abs(x['valuation']['value_difference_percent']), 
            reverse=True
        )
        
        # Return top properties
        return undervalued_properties[:count]
    
    def visualize_market_overview(self):
        """Generate visualizations of the real estate market data."""
        if self.properties_df is None or len(self.properties_df) == 0:
            print("No data available for visualization.")
            return
        
        # Set up the figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        plt.subplots_adjust(hspace=0.3, wspace=0.3)
        
        # 1. Price per sqft by area
        area_prices = self.properties_df.groupby('area_name')['price_per_sqft'].mean().sort_values()
        sns.barplot(x=area_prices.index, y=area_prices.values, ax=axes[0, 0])
        axes[0, 0].set_title('Average Price per Sqft by Area')
        axes[0, 0].set_xlabel('Area')
        axes[0, 0].set_ylabel('Price per Sqft')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Rental yield by property type
        type_yield = self.properties_df.groupby('property_type')['rental_yield'].mean().sort_values()
        sns.barplot(x=type_yield.index, y=type_yield.values, ax=axes[0, 1])
        axes[0, 1].set_title('Average Rental Yield by Property Type')
        axes[0, 1].set_xlabel('Property Type')
        axes[0, 1].set_ylabel('Rental Yield (%)')
        
        # 3. Property price vs. size scatter plot
        sns.scatterplot(
            x='size_sqft', 
            y='property_price', 
            hue='property_type',
            data=self.properties_df, 
            ax=axes[1, 0]
        )
        axes[1, 0].set_title('Property Price vs Size')
        axes[1, 0].set_xlabel('Size (sqft)')
        axes[1, 0].set_ylabel('Property Price')
        
        # 4. Price to rent ratio by area
        area_ptr = self.properties_df.groupby('area_name')['price_to_rent_ratio'].mean().sort_values()
        sns.barplot(x=area_ptr.index, y=area_ptr.values, ax=axes[1, 1])
        axes[1, 1].set_title('Average Price-to-Rent Ratio by Area')
        axes[1, 1].set_xlabel('Area')
        axes[1, 1].set_ylabel('Price-to-Rent Ratio')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
    def visualize_investment_opportunities(self, top_opportunities):
        """
        Visualize top investment opportunities.
        
        Parameters:
            top_opportunities (List[dict]): List of top investment opportunities.
        """
        if not top_opportunities:
            print("No investment opportunities to visualize.")
            return
            
        # Create a DataFrame from the opportunities
        opp_data = []
        for opp in top_opportunities:
            opp_data.append({
                'area_name': opp['property']['area_name'],
                'property_type': opp['property']['property_type'],
                'size_sqft': opp['property']['size_sqft'],
                'property_price': opp['property']['property_price'],
                'annual_rental_price': opp['property']['annual_rental_price'],
                'rental_yield': opp['property']['rental_yield'],
                'overall_score': opp['score']['overall_score'],
                'valuation_score': opp['score']['valuation_score'],
                'yield_score': opp['score']['yield_score'],
                'rental_upside_score': opp['score']['rental_upside_score'],
                'location_score': opp['score']['location_score'],
                'undervalued_percent': opp['score']['details']['valuation']['value_difference_percent']
            })
            
        opp_df = pd.DataFrame(opp_data)
        
        # Set up the figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        plt.subplots_adjust(hspace=0.3, wspace=0.3)
        
        # 1. Overall investment score by property
        sns.barplot(
            y='area_name', 
            x='overall_score',
            data=opp_df.sort_values('overall_score'), 
            ax=axes[0, 0]
        )
        axes[0, 0].set_title('Investment Score by Property')
        axes[0, 0].set_xlabel('Investment Score (0-100)')
        axes[0, 0].set_ylabel('Area')
        
        # 2. Score breakdown for top property
        top_property = opp_df.iloc[0]
        score_breakdown = pd.DataFrame({
            'Component': ['Valuation', 'Yield', 'Rental Upside', 'Location'],
            'Score': [
                top_property['valuation_score'],
                top_property['yield_score'],
                top_property['rental_upside_score'],
                top_property['location_score']
            ]
        })
        sns.barplot(
            y='Component',
            x='Score',
            data=score_breakdown,
            ax=axes[0, 1]
        )
        axes[0, 1].set_title(f'Score Breakdown for Top Property ({top_property["area_name"]})')
        axes[0, 1].set_xlabel('Score Contribution')
        
        # 3. Rental yield vs property price
        sns.scatterplot(
            x='property_price',
            y='rental_yield',
            size='overall_score',
            hue='property_type',
            data=opp_df,
            ax=axes[1, 0]
        )
        axes[1, 0].set_title('Rental Yield vs Property Price')
        axes[1, 0].set_xlabel('Property Price')
        axes[1, 0].set_ylabel('Rental Yield (%)')
        
        # 4. Undervaluation percentage for top properties
        sns.barplot(
            y='area_name',
            x='undervalued_percent',
            data=opp_df.sort_values('undervalued_percent', ascending=False),
            ax=axes[1, 1]
        )
        axes[1, 1].set_title('Undervaluation Percentage')
        axes[1, 1].set_xlabel('Undervalued (%)')
        axes[1, 1].set_ylabel('Area')
        
        plt.tight_layout()
        plt.show()
        
    def export_investment_report(self, top_opportunities, output_path='investment_report.csv'):
        """
        Export investment opportunities to a CSV file.
        
        Parameters:
            top_opportunities (List[dict]): List of top investment opportunities.
            output_path (str): Path to save the CSV file.
            
        Returns:
            str: Path to the saved file.
        """
        if not top_opportunities:
            print("No investment opportunities to export.")
            return None
            
        # Create a DataFrame from the opportunities
        opp_data = []
        for opp in top_opportunities:
            # Extract property data
            prop = opp['property']
            score = opp['score']
            
            # Create row data
            row = {
                'area_name': prop['area_name'],
                'property_type': prop['property_type'],
                'size_sqft': prop['size_sqft'],
                'bed': prop['bed'],
                'bath': prop['bath'],
                'property_price': prop['property_price'],
                'annual_rental_price': prop['annual_rental_price'],
                'price_per_sqft': prop['price_per_sqft'],
                'rental_yield': prop['rental_yield'],
                'overall_investment_score': score['overall_score'],
                'valuation_score': score['valuation_score'],
                'yield_score': score['yield_score'],
                'rental_upside_score': score['rental_upside_score'],
                'location_score': score['location_score'],
                'fair_market_value': score['details']['valuation']['fair_market_value'],
                'undervalued_amount': score['details']['valuation']['value_difference'],
                'undervalued_percent': score['details']['valuation']['value_difference_percent'],
                'optimal_rental': score['details']['rental']['optimal_annual_rental'],
                'rental_upside_amount': score['details']['rental']['rental_difference'],
                'rental_upside_percent': score['details']['rental']['rental_difference_percent']
            }
            
            # Add amenities if available
            if 'amenities' in prop:
                if isinstance(prop['amenities'], list):
                    row['amenities'] = ', '.join(prop['amenities'])
                else:
                    row['amenities'] = str(prop['amenities'])
                    
            opp_data.append(row)
            
        # Create dataframe and export to CSV
        opp_df = pd.DataFrame(opp_data)
        opp_df.to_csv(output_path, index=False)
        
        print(f"Investment report exported to {output_path}")
        return output_path
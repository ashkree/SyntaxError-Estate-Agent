"""
RentDataProcessor: Module for processing rental data

This module handles loading, cleaning, and preprocessing of rental data.
Separated from model logic to maintain clear architectural boundaries.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os


class RentalDataProcessor:
    """
    A utility class for loading, cleaning, and preprocessing rental contract data.
    This class handles data preprocessing separately from the analysis model.
    """
    
    def __init__(self):
        """Initialize the Rental Data Processor."""
        self.raw_data = None
        self.processed_data = None
        self.data_summary = {}
    
    def load_data(self, file_path):
        """
        Load rental data from a CSV file.
        
        Parameters:
            file_path (str): Path to the CSV file.
            
        Returns:
            pd.DataFrame: The loaded data.
        """
        try:
            self.raw_data = pd.read_csv(file_path)
            print(f"Data loaded: {len(self.raw_data)} records")
            return self.raw_data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def clean_data(self):
        """
        Clean and preprocess the rental data.
        
        Returns:
            pd.DataFrame: The processed data.
        """
        if self.raw_data is None:
            print("No data loaded. Please load data first.")
            return None
            
        # Create a copy of the raw data
        df = self.raw_data.copy()
        
        # 1. Handle date columns
        try:
            # Convert contract_start_date to datetime
            df['contract_start_date'] = pd.to_datetime(df['contract_start_date'], errors='coerce')
            
            # Contract end date may be in different format
            try:
                df['contract_end_date'] = pd.to_datetime(df['contract_end_date'], errors='coerce')
            except:
                # Try with different format (DD-MM-YYYY)
                df['contract_end_date'] = pd.to_datetime(df['contract_end_date'], format='%d-%m-%Y', errors='coerce')
        except Exception as e:
            print(f"Error processing date columns: {e}")
        
        # 2. Remove rows with missing critical data
        critical_columns = ['contract_start_date', 'contract_end_date', 'contract_amount', 
                           'ejari_property_type_en', 'area_name_en']
        initial_count = len(df)
        df.dropna(subset=critical_columns, inplace=True)
        dropped_count = initial_count - len(df)
        print(f"Dropped {dropped_count} rows with missing critical data")
        
        # 3. Calculate contract duration
        df['contract_duration_days'] = (df['contract_end_date'] - df['contract_start_date']).dt.days
        
        # Remove contracts with invalid duration (negative or very short/long)
        invalid_duration = (df['contract_duration_days'] < 1) | (df['contract_duration_days'] > 1825)  # Max 5 years
        invalid_count = invalid_duration.sum()
        df = df[~invalid_duration]
        print(f"Removed {invalid_count} contracts with invalid duration")
        
        # 4. Calculate derived fields
        df['contract_year'] = df['contract_start_date'].dt.year
        df['contract_month'] = df['contract_start_date'].dt.month
        df['contract_quarter'] = df['contract_start_date'].dt.quarter
        
        # Calculate monthly rent
        df['contract_duration_months'] = df['contract_duration_days'] / 30.44  # Average days per month
        df['monthly_rent'] = df['contract_amount'] / df['contract_duration_months']
        
        # Calculate annual rent (annualized)
        df['annual_rent'] = df['monthly_rent'] * 12
        
        # 5. Handle outliers in monetary values
        # Calculate IQR for monthly rent
        Q1 = df['monthly_rent'].quantile(0.25)
        Q3 = df['monthly_rent'].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outlier bounds
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        # Filter out extreme outliers
        rent_outliers = (df['monthly_rent'] < lower_bound) | (df['monthly_rent'] > upper_bound)
        outlier_count = rent_outliers.sum()
        df = df[~rent_outliers]
        print(f"Removed {outlier_count} rent outliers")
        
        # 6. Standardize categorical columns
        if 'ejari_property_type_en' in df.columns:
            df['ejari_property_type_en'] = df['ejari_property_type_en'].str.title()
            
        if 'property_usage_en' in df.columns:
            df['property_usage_en'] = df['property_usage_en'].str.title()
            
        if 'area_name_en' in df.columns:
            df['area_name_en'] = df['area_name_en'].str.title()
        
        # Store the processed data
        self.processed_data = df
        print(f"Data preprocessing complete. Final dataset: {len(df)} records")
        
        # Update data summary
        self._update_data_summary()
        
        return df
    
    def _update_data_summary(self):
        """Update the data summary statistics."""
        if self.processed_data is None:
            return
            
        df = self.processed_data
        
        # Basic counts
        self.data_summary['total_contracts'] = len(df)
        self.data_summary['date_range'] = {
            'min_date': df['contract_start_date'].min(),
            'max_date': df['contract_start_date'].max(),
            'time_span_days': (df['contract_start_date'].max() - df['contract_start_date'].min()).days
        }
        
        # Property types and usage
        self.data_summary['property_types'] = df['ejari_property_type_en'].value_counts().to_dict()
        self.data_summary['property_usage'] = df['property_usage_en'].value_counts().to_dict()
        
        # Area coverage
        self.data_summary['areas'] = {
            'count': df['area_name_en'].nunique(),
            'top_5': df['area_name_en'].value_counts().head(5).to_dict()
        }
        
        # Rental statistics
        self.data_summary['rental_stats'] = {
            'monthly': {
                'mean': df['monthly_rent'].mean(),
                'median': df['monthly_rent'].median(),
                'std': df['monthly_rent'].std(),
                'min': df['monthly_rent'].min(),
                'max': df['monthly_rent'].max()
            },
            'annual': {
                'mean': df['annual_rent'].mean(),
                'median': df['annual_rent'].median(),
                'std': df['annual_rent'].std()
            }
        }
        
        # Contract duration
        self.data_summary['duration_stats'] = {
            'mean_months': df['contract_duration_months'].mean(),
            'median_months': df['contract_duration_months'].median(),
            'common_durations': df['contract_duration_months'].round().value_counts().head(3).to_dict()
        }
    
    def get_data_summary(self):
        """Return the data summary statistics."""
        return self.data_summary
    
    def export_processed_data(self, output_file='processed_rental_data.csv'):
        """
        Export the processed data to a CSV file.
        
        Parameters:
            output_file (str): File path to save the processed data.
            
        Returns:
            str: Path to the saved file.
        """
        if self.processed_data is None:
            print("No processed data available. Please clean data first.")
            return None
            
        self.processed_data.to_csv(output_file, index=False)
        print(f"Processed data exported to {output_file}")
        return output_file
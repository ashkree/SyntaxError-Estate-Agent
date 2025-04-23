"""
Rental Data Explorer

This module provides tools for exploring and analyzing rental property data.
It handles data loading, cleaning, feature engineering, and visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class RentalDataExplorer:
    """
    A comprehensive tool for loading, cleaning, exploring and visualizing rental property data.
    This class handles the data preparation and exploration separate from the predictive modeling.
    """
    
    def __init__(self):
        """Initialize the Rental Data Explorer."""
        self.raw_data = None
        self.processed_data = None
        self.data_summary = {}
        self.current_year = datetime.now().year
        
    def load_data(self, filepath):
        """
        Load dataset from CSV and perform initial inspection.
        
        Parameters:
            filepath (str): Path to the CSV file.
            
        Returns:
            pd.DataFrame: The loaded data.
        """
        try:
            self.raw_data = pd.read_csv(filepath)
            print(f"Dataset successfully loaded: {filepath}")
            print(f"Dataset shape: {self.raw_data.shape}")
            print(f"\nFirst 5 rows:")
            print(self.raw_data.head())
            print(f"\nData info:")
            print(self.raw_data.info())
            print(f"\nSummary statistics:")
            print(self.raw_data.describe())
            print(f"\nMissing values:")
            print(self.raw_data.isnull().sum())
            
            # Save data summary
            self._update_data_summary(self.raw_data, 'raw')
            
            return self.raw_data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def preprocess_data(self):
        """
        Preprocess the loaded dataset for analysis and modeling.
        This includes cleaning, handling missing values, and feature engineering.
        
        Returns:
            pd.DataFrame: The processed data.
        """
        if self.raw_data is None:
            print("No data loaded. Please load data first.")
            return None
            
        # Create a copy of the dataframe
        processed_df = self.raw_data.copy()
        
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
        if 'Size_sqft' in processed_df.columns and 'AnnualRent' in processed_df.columns:
            processed_df['PricePerSqFt'] = processed_df['AnnualRent'] / processed_df['Size_sqft']
            
        # Create bedroom to bathroom ratio
        if 'Bedrooms' in processed_df.columns and 'Bathrooms' in processed_df.columns:
            processed_df['BedroomToBathroomRatio'] = processed_df['Bedrooms'] / processed_df['Bathrooms']
            
        # Create property value to rent ratio (potential investment metric)
        if 'PropertyValuation' in processed_df.columns and 'AnnualRent' in processed_df.columns:
            processed_df['ValueToRentRatio'] = processed_df['PropertyValuation'] / processed_df['AnnualRent']
            
        # Save processed data
        self.processed_data = processed_df
        
        # Update data summary
        self._update_data_summary(self.processed_data, 'processed')
        
        print(f"Dataset after preprocessing: {processed_df.shape}")
        # Print new features
        new_features = set(processed_df.columns) - set(self.raw_data.columns)
        print(f"New features created: {new_features}")
        
        return processed_df
    
    def _update_data_summary(self, df, data_type='raw'):
        """
        Update the data summary dictionary.
        
        Parameters:
            df (pd.DataFrame): DataFrame to summarize.
            data_type (str): Type of data ('raw' or 'processed').
        """
        summary = {}
        summary['shape'] = df.shape
        summary['columns'] = df.columns.tolist()
        summary['missing_values'] = df.isnull().sum().to_dict()
        summary['numeric_columns'] = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        summary['categorical_columns'] = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if 'AnnualRent' in df.columns:
            summary['rent_stats'] = {
                'mean': df['AnnualRent'].mean(),
                'median': df['AnnualRent'].median(),
                'min': df['AnnualRent'].min(),
                'max': df['AnnualRent'].max(),
                'std': df['AnnualRent'].std()
            }
            
        if 'Area' in df.columns:
            summary['area_count'] = df['Area'].nunique()
            summary['top_areas'] = df['Area'].value_counts().head(5).to_dict()
            
        if 'PropertyType' in df.columns:
            summary['property_types'] = df['PropertyType'].value_counts().to_dict()
            
        self.data_summary[data_type] = summary
    
    def explore_data(self, df=None, save_figures=False, output_dir='plots'):
        """
        Explore the dataset with visualizations.
        
        Parameters:
            df (pd.DataFrame, optional): DataFrame to explore. Uses processed_data if None.
            save_figures (bool): Whether to save the figures.
            output_dir (str): Directory to save figures if save_figures is True.
        """
        if df is None:
            if self.processed_data is not None:
                df = self.processed_data
            elif self.raw_data is not None:
                df = self.raw_data
            else:
                print("No data available for exploration. Please load or preprocess data first.")
                return
                
        # Distribution of target variable
        plt.figure(figsize=(10, 6))
        sns.histplot(df['AnnualRent'], kde=True)
        plt.title('Distribution of Annual Rent')
        plt.xlabel('Annual Rent')
        plt.show()
        
        # Correlation matrix for numeric features
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        plt.figure(figsize=(12, 8))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix')
        plt.show()
        
        # Rent by Area (top 10 areas)
        if 'Area' in df.columns:
            plt.figure(figsize=(12, 8))
            area_rent = df.groupby('Area')['AnnualRent'].mean().sort_values(ascending=False).head(10)
            sns.barplot(x=area_rent.index, y=area_rent.values)
            plt.title('Average Annual Rent by Area (Top 10)')
            plt.xticks(rotation=45, ha='right')
            plt.ylabel('Average Annual Rent')
            plt.tight_layout()
            plt.show()
            
        # Rent by Property Type
        if 'PropertyType' in df.columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='PropertyType', y='AnnualRent', data=df)
            plt.title('Annual Rent by Property Type')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            
        # Rent by Size
        if 'Size_sqft' in df.columns:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x='Size_sqft', y='AnnualRent', hue='Bedrooms' if 'Bedrooms' in df.columns else None, data=df)
            plt.title('Annual Rent vs Size')
            plt.xlabel('Size (sq ft)')
            plt.ylabel('Annual Rent')
            plt.tight_layout()
            plt.show()
            
        # Rent by Furnishing Status
        if 'Furnishing' in df.columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='Furnishing', y='AnnualRent', data=df)
            plt.title('Annual Rent by Furnishing Status')
            plt.tight_layout()
            plt.show()
            
        # Distribution of property age
        if 'PropertyAge' in df.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(df['PropertyAge'], kde=True)
            plt.title('Distribution of Property Age')
            plt.xlabel('Age (years)')
            plt.tight_layout()
            plt.show()
            
            # Age vs Rent
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x='PropertyAge', y='AnnualRent', data=df)
            plt.title('Annual Rent vs Property Age')
            plt.xlabel('Age (years)')
            plt.ylabel('Annual Rent')
            plt.tight_layout()
            plt.show()
    
    def explore_with_plotly(self, df=None):
        """
        Create interactive Plotly visualizations for deeper exploration.
        
        Parameters:
            df (pd.DataFrame, optional): DataFrame to explore. Uses processed_data if None.
        """
        if df is None:
            if self.processed_data is not None:
                df = self.processed_data
            elif self.raw_data is not None:
                df = self.raw_data
            else:
                print("No data available for exploration. Please load or preprocess data first.")
                return
        
        # Rent distribution with Plotly
        fig = px.histogram(df, x='AnnualRent', 
                          nbins=30, 
                          title='Distribution of Annual Rent',
                          opacity=0.7,
                          marginal='box')
        fig.show()
        
        # Scatter plot of Size vs Rent with bedrooms
        if all(col in df.columns for col in ['Size_sqft', 'Bedrooms']):
            fig = px.scatter(df, x='Size_sqft', y='AnnualRent', 
                            color='Bedrooms', size='Bedrooms',
                            hover_data=['Area', 'PropertyType', 'Bathrooms'],
                            title='Annual Rent vs Size by Bedrooms')
            fig.show()
            
        # Bar chart for average rent by property type
        if 'PropertyType' in df.columns:
            fig = px.bar(df.groupby('PropertyType')['AnnualRent'].mean().reset_index(),
                        x='PropertyType', y='AnnualRent',
                        title='Average Annual Rent by Property Type')
            fig.show()
            
        # Boxplot for rent by property type
        if 'PropertyType' in df.columns:
            fig = px.box(df, x='PropertyType', y='AnnualRent',
                        title='Annual Rent Distribution by Property Type')
            fig.show()
            
        # 3D scatter plot for size, bedrooms, and rent
        if all(col in df.columns for col in ['Size_sqft', 'Bedrooms', 'Bathrooms']):
            fig = px.scatter_3d(df, x='Size_sqft', y='Bedrooms', z='AnnualRent',
                               color='PropertyType' if 'PropertyType' in df.columns else None,
                               size='Bathrooms',
                               title='3D Plot: Size, Bedrooms, and Rent')
            fig.show()
            
        # Heatmap for correlation matrix
        numeric_df = df.select_dtypes(include=['int64', 'float64'])
        corr = numeric_df.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale='RdBu_r',
            text=corr.values.round(2),
            texttemplate='%{text}',
            colorbar=dict(title='Correlation')
        ))
        
        fig.update_layout(
            title='Correlation Matrix of Numeric Features',
            width=800,
            height=800
        )
        
        fig.show()
    
    def analyze_amenities(self, df=None):
        """
        Analyze the impact of different amenities on rental prices.
        
        Parameters:
            df (pd.DataFrame, optional): DataFrame to analyze. Uses processed_data if None.
            
        Returns:
            pd.DataFrame: DataFrame with amenity impact analysis.
        """
        if df is None:
            if self.processed_data is not None:
                df = self.processed_data
            else:
                print("No processed data available. Please preprocess data first.")
                return
                
        if 'Amenities' not in df.columns:
            print("No amenities column found in the data.")
            return
            
        # Find all binary amenity columns
        amenity_cols = [col for col in df.columns if col.startswith('Has_')]
        
        if not amenity_cols:
            print("No binary amenity columns found. Make sure to preprocess the data first.")
            return
            
        # Calculate average rent for properties with and without each amenity
        amenity_impact = {}
        overall_avg_rent = df['AnnualRent'].mean()
        
        for amenity in amenity_cols:
            with_amenity = df[df[amenity] == 1]['AnnualRent'].mean()
            without_amenity = df[df[amenity] == 0]['AnnualRent'].mean()
            premium_pct = ((with_amenity / without_amenity) - 1) * 100 if without_amenity > 0 else 0
            
            amenity_impact[amenity] = {
                'with_amenity': with_amenity,
                'without_amenity': without_amenity,
                'premium_pct': premium_pct,
                'count': df[amenity].sum()
            }
            
        # Convert to DataFrame for easier visualization
        impact_df = pd.DataFrame.from_dict(amenity_impact, orient='index')
        impact_df = impact_df.sort_values('premium_pct', ascending=False)
        
        # Plot amenity impact
        plt.figure(figsize=(12, 8))
        sns.barplot(x=impact_df.index, y=impact_df['premium_pct'])
        plt.title('Rental Premium by Amenity (%)')
        plt.xticks(rotation=90)
        plt.ylabel('Premium (%)')
        plt.tight_layout()
        plt.show()
        
        return impact_df
    
    def analyze_price_per_sqft(self, df=None):
        """
        Analyze price per square foot across different property types and areas.
        
        Parameters:
            df (pd.DataFrame, optional): DataFrame to analyze. Uses processed_data if None.
        """
        if df is None:
            if self.processed_data is not None:
                df = self.processed_data
            else:
                print("No processed data available. Please preprocess data first.")
                return
                
        if 'PricePerSqFt' not in df.columns:
            if 'Size_sqft' in df.columns and 'AnnualRent' in df.columns:
                df['PricePerSqFt'] = df['AnnualRent'] / df['Size_sqft']
            else:
                print("Cannot calculate price per square foot: missing required columns")
                return
                
        # Overall price per sqft statistics
        print(f"Overall Price per Sqft Statistics:")
        print(df['PricePerSqFt'].describe())
        
        # Price per sqft by property type
        if 'PropertyType' in df.columns:
            psf_by_type = df.groupby('PropertyType')['PricePerSqFt'].agg(['mean', 'median', 'count']).sort_values('mean', ascending=False)
            print(f"\nPrice per Sqft by Property Type:")
            print(psf_by_type)
            
            # Visualization
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='PropertyType', y='PricePerSqFt', data=df)
            plt.title('Price per Sqft by Property Type')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            
        # Price per sqft by area (top 10)
        if 'Area' in df.columns:
            psf_by_area = df.groupby('Area')['PricePerSqFt'].mean().sort_values(ascending=False).head(10)
            print(f"\nTop 10 Areas by Price per Sqft:")
            print(psf_by_area)
            
            # Visualization
            plt.figure(figsize=(12, 6))
            sns.barplot(x=psf_by_area.index, y=psf_by_area.values)
            plt.title('Top 10 Areas by Price per Sqft')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
            
        # Price per sqft by property size (scatter plot)
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='Size_sqft', y='PricePerSqFt', hue='PropertyType' if 'PropertyType' in df.columns else None, data=df)
        plt.title('Price per Sqft vs Property Size')
        plt.xlabel('Size (sq ft)')
        plt.ylabel('Price per Sqft')
        plt.tight_layout()
        plt.show()
        
        # Show decreasing trend of price per sqft as size increases
        plt.figure(figsize=(10, 6))
        df_sorted = df.sort_values('Size_sqft')
        plt.scatter(df_sorted['Size_sqft'], df_sorted['PricePerSqFt'], alpha=0.5)
        
        # Add trendline
        slope, intercept, r_value, p_value, std_err = stats.linregress(df_sorted['Size_sqft'], df_sorted['PricePerSqFt'])
        plt.plot(df_sorted['Size_sqft'], intercept + slope*df_sorted['Size_sqft'], 'r-')
        plt.title(f'Price per Sqft vs Size (R²: {r_value**2:.2f})')
        plt.xlabel('Size (sq ft)')
        plt.ylabel('Price per Sqft')
        plt.tight_layout()
        plt.show()
    
    def analyze_investment_potential(self, df=None):
        """
        Analyze investment potential based on property metrics.
        
        Parameters:
            df (pd.DataFrame, optional): DataFrame to analyze. Uses processed_data if None.
        """
        if df is None:
            if self.processed_data is not None:
                df = self.processed_data
            else:
                print("No processed data available. Please preprocess data first.")
                return
                
        if 'PropertyValuation' not in df.columns or 'AnnualRent' not in df.columns:
            print("Missing required columns for investment analysis: PropertyValuation and/or AnnualRent")
            return
            
        # Calculate value-to-rent ratio if not already present
        if 'ValueToRentRatio' not in df.columns:
            df['ValueToRentRatio'] = df['PropertyValuation'] / df['AnnualRent']
            
        # Calculate rental yield
        df['RentalYield'] = (df['AnnualRent'] / df['PropertyValuation']) * 100
        
        # Overall investment metrics
        print("Overall Investment Metrics:")
        print(f"Average Value-to-Rent Ratio: {df['ValueToRentRatio'].mean():.2f}")
        print(f"Average Rental Yield: {df['RentalYield'].mean():.2f}%")
        
        # Investment metrics by property type
        if 'PropertyType' in df.columns:
            inv_by_type = df.groupby('PropertyType').agg({
                'ValueToRentRatio': 'mean',
                'RentalYield': 'mean',
                'PropertyValuation': 'mean',
                'AnnualRent': 'mean'
            }).sort_values('RentalYield', ascending=False)
            
            print("\nInvestment Metrics by Property Type:")
            print(inv_by_type)
            
            # Visualization
            plt.figure(figsize=(10, 6))
            sns.barplot(x=inv_by_type.index, y=inv_by_type['RentalYield'])
            plt.title('Average Rental Yield by Property Type')
            plt.xticks(rotation=45)
            plt.ylabel('Rental Yield (%)')
            plt.tight_layout()
            plt.show()
            
        # Best investment areas (top 10 by rental yield)
        if 'Area' in df.columns:
            inv_by_area = df.groupby('Area').agg({
                'ValueToRentRatio': 'mean',
                'RentalYield': 'mean',
                'PropertyValuation': 'mean',
                'AnnualRent': 'mean',
                'Area': 'count'
            }).rename(columns={'Area': 'Count'}).sort_values('RentalYield', ascending=False)
            
            # Filter areas with at least 3 properties for more reliable statistics
            # Only filter if there are areas with at least 3 properties
            if any(inv_by_area['Count'] >= 3):
                inv_by_area_filtered = inv_by_area[inv_by_area['Count'] >= 3]
                print("\nTop 10 Areas by Rental Yield (min 3 properties):")
                print(inv_by_area_filtered.head(10)[['RentalYield', 'ValueToRentRatio', 'Count']])
                
                # Visualization
                if len(inv_by_area_filtered) > 0:
                    plt.figure(figsize=(12, 6))
                    top_yield_areas = inv_by_area_filtered.head(10)
                    sns.barplot(x=top_yield_areas.index, y=top_yield_areas['RentalYield'])
                    plt.title('Top 10 Areas by Rental Yield')
                    plt.xticks(rotation=45, ha='right')
                    plt.ylabel('Rental Yield (%)')
                    plt.tight_layout()
                    plt.show()
            else:
                # If no areas have at least 3 properties, show unfiltered results
                print("\nTop 10 Areas by Rental Yield (Note: Some areas have few properties):")
                print(inv_by_area.head(10)[['RentalYield', 'ValueToRentRatio', 'Count']])
                
                # Visualization
                plt.figure(figsize=(12, 6))
                top_yield_areas = inv_by_area.head(10)
                sns.barplot(x=top_yield_areas.index, y=top_yield_areas['RentalYield'])
                plt.title('Top 10 Areas by Rental Yield')
                plt.xticks(rotation=45, ha='right')
                plt.ylabel('Rental Yield (%)')
                plt.tight_layout()
                plt.show()
            
        # Yield vs property valuation scatter plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='PropertyValuation', y='RentalYield', hue='PropertyType' if 'PropertyType' in df.columns else None, data=df)
        plt.title('Rental Yield vs Property Valuation')
        plt.xlabel('Property Valuation')
        plt.ylabel('Rental Yield (%)')
        plt.tight_layout()
        plt.show()
        
        return df[['PropertyValuation', 'AnnualRent', 'ValueToRentRatio', 'RentalYield']]
    
    def generate_summary_report(self, output_file='rental_data_summary.txt'):
        """
        Generate a comprehensive summary report of the data.
        
        Parameters:
            output_file (str): File path to save the report.
            
        Returns:
            str: Path to the saved file.
        """
        if self.processed_data is None:
            print("No processed data available. Please preprocess data first.")
            return None
            
        df = self.processed_data
        
        # Build report
        report = []
        report.append("=" * 80)
        report.append("RENTAL DATA SUMMARY REPORT")
        report.append("=" * 80)
        report.append(f"\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        report.append("\n1. DATASET OVERVIEW")
        report.append(f"Total properties: {len(df)}")
        report.append(f"Number of features: {df.shape[1]}")
        
        if 'Area' in df.columns:
            report.append(f"Areas covered: {df['Area'].nunique()}")
            
        if 'PropertyType' in df.columns:
            report.append("\nProperty Type Distribution:")
            for prop_type, count in df['PropertyType'].value_counts().items():
                report.append(f"  - {prop_type}: {count} ({count/len(df)*100:.1f}%)")
                
        report.append("\n2. RENTAL STATISTICS")
        report.append(f"Average Annual Rent: {df['AnnualRent'].mean():.2f}")
        report.append(f"Median Annual Rent: {df['AnnualRent'].median():.2f}")
        report.append(f"Minimum Annual Rent: {df['AnnualRent'].min():.2f}")
        report.append(f"Maximum Annual Rent: {df['AnnualRent'].max():.2f}")
        
        if 'PricePerSqFt' in df.columns:
            report.append(f"\nAverage Price per Sqft: {df['PricePerSqFt'].mean():.2f}")
            report.append(f"Median Price per Sqft: {df['PricePerSqFt'].median():.2f}")
            
        if 'PropertyType' in df.columns:
            report.append("\nAverage Rent by Property Type:")
            avg_by_type = df.groupby('PropertyType')['AnnualRent'].mean().sort_values(ascending=False)
            for prop_type, avg_rent in avg_by_type.items():
                report.append(f"  - {prop_type}: {avg_rent:.2f}")
                
        if 'Area' in df.columns:
            report.append("\nTop 5 Most Expensive Areas:")
            top_areas = df.groupby('Area')['AnnualRent'].mean().sort_values(ascending=False).head(5)
            for area, avg_rent in top_areas.items():
                report.append(f"  - {area}: {avg_rent:.2f}")
                
        report.append("\n3. PROPERTY CHARACTERISTICS")
        if 'Size_sqft' in df.columns:
            report.append(f"Average Property Size: {df['Size_sqft'].mean():.2f} sqft")
            report.append(f"Median Property Size: {df['Size_sqft'].median():.2f} sqft")
            
        if 'Bedrooms' in df.columns:
            bed_counts = df['Bedrooms'].value_counts().sort_index()
            report.append("\nBedroom Distribution:")
            for beds, count in bed_counts.items():
                report.append(f"  - {beds} bedroom(s): {count} ({count/len(df)*100:.1f}%)")
                
        if 'Bathrooms' in df.columns:
            bath_counts = df['Bathrooms'].value_counts().sort_index()
            report.append("\nBathroom Distribution:")
            for baths, count in bath_counts.items():
                report.append(f"  - {baths} bathroom(s): {count} ({count/len(df)*100:.1f}%)")
                
        if 'Furnishing' in df.columns:
            report.append("\nFurnishing Status Distribution:")
            for status, count in df['Furnishing'].value_counts().items():
                report.append(f"  - {status}: {count} ({count/len(df)*100:.1f}%)")
                
        if 'PropertyAge' in df.columns:
            report.append(f"\nAverage Property Age: {df['PropertyAge'].mean():.1f} years")
            report.append(f"Median Property Age: {df['PropertyAge'].median():.1f} years")
            
        report.append("\n4. AMENITIES")
        amenity_cols = [col for col in df.columns if col.startswith('Has_')]
        if amenity_cols:
            amenity_counts = df[amenity_cols].sum().sort_values(ascending=False)
            report.append("Amenity Distribution:")
            for amenity, count in amenity_counts.items():
                report.append(f"  - {amenity.replace('Has_', '')}: {count} ({count/len(df)*100:.1f}%)")
                
        if 'AmenityCount' in df.columns:
            report.append(f"\nAverage Amenity Count: {df['AmenityCount'].mean():.1f}")
            report.append(f"Median Amenity Count: {df['AmenityCount'].median():.1f}")
            
        report.append("\n5. INVESTMENT METRICS")
        if all(col in df.columns for col in ['PropertyValuation', 'AnnualRent']):
            rental_yield = (df['AnnualRent'] / df['PropertyValuation']) * 100
            df['RentalYield'] = rental_yield
            
            report.append(f"Average Rental Yield: {rental_yield.mean():.2f}%")
            report.append(f"Median Rental Yield: {rental_yield.median():.2f}%")
            
            if 'PropertyType' in df.columns:
                report.append("\nAverage Rental Yield by Property Type:")
                yield_by_type = df.groupby('PropertyType')['RentalYield'].mean().sort_values(ascending=False)
                for prop_type, avg_yield in yield_by_type.items():
                    report.append(f"  - {prop_type}: {avg_yield:.2f}%")
                    
            if 'Area' in df.columns:
                report.append("\nTop 5 Areas by Rental Yield:")
                top_yield_areas = df.groupby('Area').agg({
                    'RentalYield': 'mean', 
                    'Area': 'count'
                }).rename(columns={'Area': 'Count'})
                
                if any(top_yield_areas['Count'] >= 3):
                    top_yield_areas = top_yield_areas[top_yield_areas['Count'] >= 3]  # At least 3 properties
                
                top_yield_areas = top_yield_areas.sort_values('RentalYield', ascending=False).head(5)
                
                for area, row in top_yield_areas.iterrows():
                    report.append(f"  - {area}: {row['RentalYield']:.2f}% (based on {row['Count']} properties)")
                    
        report.append("\n6. CORRELATIONS WITH ANNUAL RENT")
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        rent_correlations = df[numeric_cols].corr()['AnnualRent'].sort_values(ascending=False)
        rent_correlations = rent_correlations[rent_correlations.index != 'AnnualRent']  # Remove self-correlation
        
        report.append("Top 10 features most correlated with Annual Rent:")
        for feature, corr in rent_correlations.head(10).items():
            report.append(f"  - {feature}: {corr:.4f}")
            
        report.append("\n" + "=" * 80)
        
        # Write report to file
        with open(output_file, 'w') as f:
            f.write("\n".join(report))
            
        print(f"Summary report saved to {output_file}")
        return output_file
    
    def export_processed_data(self, output_file='processed_rental_data.csv'):
        """
        Export the processed data to a CSV file.
        
        Parameters:
            output_file (str): File path to save the processed data.
            
        Returns:
            str: Path to the saved file.
        """
        if self.processed_data is None:
            print("No processed data available. Please preprocess data first.")
            return None
            
        self.processed_data.to_csv(output_file, index=False)
        print(f"Processed data exported to {output_file}")
        return output_file
    
    def feature_importance_analysis(self, target='AnnualRent'):
        """
        Analyze feature importance using a simple Random Forest model.
        This provides a quick way to identify potentially important features.
        
        Parameters:
            target (str): Target variable name.
            
        Returns:
            pd.DataFrame: DataFrame with feature importance scores.
        """
        if self.processed_data is None:
            print("No processed data available. Please preprocess data first.")
            return None
            
        if target not in self.processed_data.columns:
            print(f"Target variable '{target}' not found in the processed data.")
            return None
            
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        
        df = self.processed_data.copy()
        
        # Prepare the data
        y = df[target]
        X = df.drop(columns=[target])
        
        # Handle categorical features
        cat_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            X[col] = X[col].astype('category').cat.codes
            
        # Drop any remaining non-numeric columns
        X = X.select_dtypes(include=['int64', 'float64'])
        
        # Train a simple Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # Get feature importance
        importance = rf.feature_importances_
        
        # Create DataFrame for feature importance
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importance
        }).sort_values(by='Importance', ascending=False)
        
        # Visualize feature importance
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
        plt.title('Top 20 Feature Importance')
        plt.tight_layout()
        plt.show()
        
        return feature_importance


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
    
    # Initialize explorer
    explorer = RentalDataExplorer()
    
    # Load data
    explorer.raw_data = sample_data
    print("Sample data loaded successfully.")
    
    # Preprocess data
    processed_data = explorer.preprocess_data()
    
    # Explore data
    explorer.explore_data()
    
    # Export processed data for modeling
    explorer.export_processed_data("processed_rental_data.csv")
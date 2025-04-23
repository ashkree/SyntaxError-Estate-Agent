import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os


class RentalDataProcessor:
    """
    A utility class for loading, cleaning, and exploring rental contract data.
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
    
    def explore_data(self, save_plots=False, output_dir='plots'):
        """
        Generate exploratory data visualizations.
        
        Parameters:
            save_plots (bool): Whether to save plots to files.
            output_dir (str): Directory to save plots if save_plots is True.
        """
        if self.processed_data is None:
            print("No processed data available. Please clean data first.")
            return
            
        df = self.processed_data
        
        # Create output directory if saving plots
        if save_plots:
            os.makedirs(output_dir, exist_ok=True)
        
        # 1. Distribution of monthly rent
        plt.figure(figsize=(10, 6))
        sns.histplot(df['monthly_rent'], kde=True)
        plt.title('Distribution of Monthly Rent')
        plt.xlabel('Monthly Rent')
        plt.ylabel('Frequency')
        if save_plots:
            plt.savefig(f"{output_dir}/monthly_rent_distribution.png")
        plt.show()
        
        # 2. Rent by property type
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='ejari_property_type_en', y='monthly_rent', data=df)
        plt.title('Monthly Rent by Property Type')
        plt.xlabel('Property Type')
        plt.ylabel('Monthly Rent')
        plt.xticks(rotation=45)
        if save_plots:
            plt.savefig(f"{output_dir}/rent_by_property_type.png")
        plt.show()
        
        # 3. Rent by usage type
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='property_usage_en', y='monthly_rent', data=df)
        plt.title('Monthly Rent by Property Usage')
        plt.xlabel('Property Usage')
        plt.ylabel('Monthly Rent')
        if save_plots:
            plt.savefig(f"{output_dir}/rent_by_usage.png")
        plt.show()
        
        # 4. Time series of rent (yearly average)
        yearly_rent = df.groupby('contract_year')['monthly_rent'].mean().reset_index()
        plt.figure(figsize=(10, 6))
        sns.lineplot(x='contract_year', y='monthly_rent', data=yearly_rent, marker='o')
        plt.title('Average Monthly Rent by Year')
        plt.xlabel('Year')
        plt.ylabel('Average Monthly Rent')
        if save_plots:
            plt.savefig(f"{output_dir}/rent_by_year.png")
        plt.show()
        
        # 5. Top 10 areas by average rent
        top_areas = df.groupby('area_name_en')['monthly_rent'].mean().nlargest(10).reset_index()
        plt.figure(figsize=(12, 6))
        sns.barplot(x='monthly_rent', y='area_name_en', data=top_areas)
        plt.title('Top 10 Areas by Average Monthly Rent')
        plt.xlabel('Average Monthly Rent')
        plt.ylabel('Area')
        if save_plots:
            plt.savefig(f"{output_dir}/top_areas_by_rent.png")
        plt.show()
        
        # 6. Contract duration distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(df['contract_duration_months'], bins=20, kde=True)
        plt.title('Distribution of Contract Duration')
        plt.xlabel('Contract Duration (Months)')
        plt.ylabel('Frequency')
        if save_plots:
            plt.savefig(f"{output_dir}/contract_duration_distribution.png")
        plt.show()
        
        # 7. Seasonal patterns (monthly)
        monthly_rent = df.groupby('contract_month')['monthly_rent'].mean().reset_index()
        plt.figure(figsize=(10, 6))
        sns.lineplot(x='contract_month', y='monthly_rent', data=monthly_rent, marker='o')
        plt.title('Average Monthly Rent by Month (Seasonal Pattern)')
        plt.xlabel('Month')
        plt.ylabel('Average Monthly Rent')
        plt.xticks(range(1, 13))
        if save_plots:
            plt.savefig(f"{output_dir}/seasonal_rent_pattern.png")
        plt.show()
        
        # 8. Correlation between contract duration and rent
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='contract_duration_months', y='monthly_rent', data=df, alpha=0.5)
        plt.title('Relationship Between Contract Duration and Monthly Rent')
        plt.xlabel('Contract Duration (Months)')
        plt.ylabel('Monthly Rent')
        if save_plots:
            plt.savefig(f"{output_dir}/duration_vs_rent.png")
        plt.show()
        
        print("Exploratory data analysis complete.")
    
    def generate_summary_report(self, output_file='rental_data_summary.txt'):
        """
        Generate a summary report of the data.
        
        Parameters:
            output_file (str): File path to save the report.
            
        Returns:
            str: Summary report as a string.
        """
        if self.processed_data is None:
            return "No processed data available. Please clean data first."
            
        df = self.processed_data
        summary = self.get_data_summary()
        
        # Build report
        report = []
        report.append("=" * 80)
        report.append("RENTAL DATA SUMMARY REPORT")
        report.append("=" * 80)
        report.append(f"\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        report.append("\n1. DATASET OVERVIEW")
        report.append(f"Total contracts: {summary['total_contracts']:,}")
        report.append(f"Date range: {summary['date_range']['min_date']} to {summary['date_range']['max_date']}")
        report.append(f"Time span: {summary['date_range']['time_span_days']:,} days")
        
        report.append("\n2. PROPERTY COMPOSITION")
        report.append("Property types:")
        for prop_type, count in summary['property_types'].items():
            report.append(f"  - {prop_type}: {count:,} ({count/summary['total_contracts']*100:.1f}%)")
        
        report.append("\nProperty usage:")
        for usage, count in summary['property_usage'].items():
            report.append(f"  - {usage}: {count:,} ({count/summary['total_contracts']*100:.1f}%)")
        
        report.append(f"\nAreas covered: {summary['areas']['count']}")
        report.append("Top 5 areas by contract count:")
        for area, count in summary['areas']['top_5'].items():
            report.append(f"  - {area}: {count:,} contracts")
        
        report.append("\n3. RENTAL STATISTICS")
        report.append("Monthly rent:")
        report.append(f"  - Mean: {summary['rental_stats']['monthly']['mean']:,.2f}")
        report.append(f"  - Median: {summary['rental_stats']['monthly']['median']:,.2f}")
        report.append(f"  - Standard deviation: {summary['rental_stats']['monthly']['std']:,.2f}")
        report.append(f"  - Range: {summary['rental_stats']['monthly']['min']:,.2f} to {summary['rental_stats']['monthly']['max']:,.2f}")
        
        report.append("\nAnnual rent (annualized):")
        report.append(f"  - Mean: {summary['rental_stats']['annual']['mean']:,.2f}")
        report.append(f"  - Median: {summary['rental_stats']['annual']['median']:,.2f}")
        
        report.append("\n4. CONTRACT DURATION")
        report.append(f"Mean duration: {summary['duration_stats']['mean_months']:.2f} months")
        report.append(f"Median duration: {summary['duration_stats']['median_months']:.2f} months")
        report.append("Most common durations:")
        for duration, count in summary['duration_stats']['common_durations'].items():
            report.append(f"  - {duration:.0f} months: {count:,} contracts")
        
        # Additional statistics
        report.append("\n5. ADDITIONAL INSIGHTS")
        
        # Property type with highest average rent
        prop_type_rent = df.groupby('ejari_property_type_en')['monthly_rent'].mean().sort_values(ascending=False)
        report.append(f"Property type with highest average rent: {prop_type_rent.index[0]} ({prop_type_rent.iloc[0]:,.2f})")
        
        # Area with highest average rent
        area_rent = df.groupby('area_name_en')['monthly_rent'].mean().sort_values(ascending=False)
        report.append(f"Area with highest average rent: {area_rent.index[0]} ({area_rent.iloc[0]:,.2f})")
        
        # Year with highest average rent
        year_rent = df.groupby('contract_year')['monthly_rent'].mean().sort_values(ascending=False)
        report.append(f"Year with highest average rent: {year_rent.index[0]} ({year_rent.iloc[0]:,.2f})")
        
        # Month with highest average rent
        month_rent = df.groupby('contract_month')['monthly_rent'].mean().sort_values(ascending=False)
        report.append(f"Month with highest average rent: {month_rent.index[0]} ({month_rent.iloc[0]:,.2f})")
        
        report.append("\n" + "=" * 80)
        
        # Write report to file
        report_text = "\n".join(report)
        with open(output_file, 'w') as f:
            f.write(report_text)
        
        print(f"Summary report saved to {output_file}")
        return report_text
    
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


# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = RentalDataProcessor()
    
    # Load data
    processor.load_data("/home/maveron/Projects/SyntaxError-Estate-Agent/data/processed/rent_data.csv")
    
    # Clean and preprocess data
    processed_data = processor.clean_data()
    
    # Explore the data
    processor.explore_data(save_plots=True)
    
    # Generate summary report
    processor.generate_summary_report()
    
    # Export processed data
    processor.export_processed_data()
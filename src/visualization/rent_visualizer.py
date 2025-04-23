"""
RentVisualizer: Module for visualizing rental data and analysis results

This module provides visualization functions for rental data, separated from
the core model logic for better architectural organization.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


class RentVisualizer:
    """
    A class for creating visualizations of rental data and analysis results.
    """
    
    def __init__(self, style='whitegrid'):
        """
        Initialize the RentVisualizer.
        
        Parameters:
            style (str): Seaborn style for plots.
        """
        self.style = style
        sns.set_style(style)
        self.fig_size = (12, 8)
    
    def visualize_market_overview(self, contracts_df):
        """
        Generate visualizations of the rental market data.
        
        Parameters:
            contracts_df (pd.DataFrame): DataFrame containing rental contract data.
        """
        if contracts_df is None or len(contracts_df) == 0:
            print("No data available for visualization.")
            return
        
        # Set up the figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=self.fig_size)
        plt.subplots_adjust(hspace=0.3, wspace=0.3)
        
        # 1. Average rent by property type
        type_rents = contracts_df.groupby('ejari_property_type_en')['monthly_rent'].mean().sort_values()
        sns.barplot(x=type_rents.index, y=type_rents.values, ax=axes[0, 0])
        axes[0, 0].set_title('Average Monthly Rent by Property Type')
        axes[0, 0].set_xlabel('Property Type')
        axes[0, 0].set_ylabel('Monthly Rent')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Average rent by property usage
        usage_rents = contracts_df.groupby('property_usage_en')['monthly_rent'].mean().sort_values()
        sns.barplot(x=usage_rents.index, y=usage_rents.values, ax=axes[0, 1])
        axes[0, 1].set_title('Average Monthly Rent by Property Usage')
        axes[0, 1].set_xlabel('Property Usage')
        axes[0, 1].set_ylabel('Monthly Rent')
        
        # 3. Top 10 areas by average rent
        area_rents = contracts_df.groupby('area_name_en')['monthly_rent'].mean().sort_values(ascending=False).head(10)
        sns.barplot(x=area_rents.values, y=area_rents.index, ax=axes[1, 0])
        axes[1, 0].set_title('Top 10 Areas by Average Monthly Rent')
        axes[1, 0].set_xlabel('Monthly Rent')
        axes[1, 0].set_ylabel('Area')
        
        # 4. Seasonal trends (monthly)
        monthly_trends = contracts_df.groupby('contract_month')['monthly_rent'].mean()
        # Ensure all months are present
        all_months = pd.Series(index=range(1, 13), data=[monthly_trends.get(m, np.nan) for m in range(1, 13)])
        sns.lineplot(x=all_months.index, y=all_months.values, ax=axes[1, 1], marker='o')
        axes[1, 1].set_title('Seasonal Rent Trends by Month')
        axes[1, 1].set_xlabel('Month')
        axes[1, 1].set_ylabel('Average Monthly Rent')
        axes[1, 1].set_xticks(range(1, 13))
        
        plt.tight_layout()
        plt.show()
    
    def visualize_rent_trends(self, contracts_df, area=None, property_type=None):
        """
        Visualize rental price trends over time.
        
        Parameters:
            contracts_df (pd.DataFrame): DataFrame containing rental contract data.
            area (str, optional): Filter by area name.
            property_type (str, optional): Filter by property type.
        """
        if contracts_df is None or len(contracts_df) == 0:
            print("No data available for visualization.")
            return
        
        # Apply filters
        filtered_df = contracts_df.copy()
        
        if area:
            filtered_df = filtered_df[filtered_df['area_name_en'] == area]
            
        if property_type:
            filtered_df = filtered_df[filtered_df['ejari_property_type_en'] == property_type]
            
        if len(filtered_df) == 0:
            print("No data available after filtering.")
            return
        
        # Calculate yearly trends
        yearly_trends = filtered_df.groupby('contract_year')['monthly_rent'].agg(['mean', 'count']).reset_index()
        
        # Calculate quarterly trends
        filtered_df['year_quarter'] = filtered_df['contract_year'].astype(str) + '-Q' + filtered_df['contract_quarter'].astype(str)
        quarterly_trends = filtered_df.groupby('year_quarter')['monthly_rent'].agg(['mean', 'count']).reset_index()
        
        # Title addition based on filters
        title_addition = ""
        if area:
            title_addition += f" - {area}"
        if property_type:
            title_addition += f" - {property_type}"
        
        # Set up the figure with subplots
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        plt.subplots_adjust(hspace=0.3)
        
        # 1. Yearly trends
        sns.lineplot(
            x='contract_year', 
            y='mean', 
            data=yearly_trends,
            marker='o',
            ax=axes[0]
        )
        # Add count labels
        for i, row in yearly_trends.iterrows():
            axes[0].annotate(
                f"n={int(row['count'])}",
                (row['contract_year'], row['mean']),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center'
            )
        axes[0].set_title(f'Yearly Rental Price Trends{title_addition}')
        axes[0].set_xlabel('Year')
        axes[0].set_ylabel('Average Monthly Rent')
        
        # 2. Quarterly trends
        sns.lineplot(
            x='year_quarter', 
            y='mean', 
            data=quarterly_trends,
            marker='o',
            ax=axes[1]
        )
        # Add count labels (but skip some for readability if too many)
        skip_factor = max(1, len(quarterly_trends) // 10)  # Show at most ~10 labels
        for i, row in quarterly_trends.iterrows()[::skip_factor]:
            axes[1].annotate(
                f"n={int(row['count'])}",
                (i, row['mean']),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center'
            )
        axes[1].set_title(f'Quarterly Rental Price Trends{title_addition}')
        axes[1].set_xlabel('Year-Quarter')
        axes[1].set_ylabel('Average Monthly Rent')
        axes[1].tick_params(axis='x', rotation=90)
        
        plt.tight_layout()
        plt.show()
        
    def plot_area_comparison(self, contracts_df, areas, property_type=None):
        """
        Plot comparison of rental prices across different areas.
        
        Parameters:
            contracts_df (pd.DataFrame): DataFrame containing rental contract data.
            areas (list): List of area names to compare.
            property_type (str, optional): Filter by property type.
        """
        if contracts_df is None or len(contracts_df) == 0:
            print("No data available for visualization.")
            return
            
        # Filter data
        filtered_df = contracts_df.copy()
        
        if property_type:
            filtered_df = filtered_df[filtered_df['ejari_property_type_en'] == property_type]
        
        # Filter for requested areas
        filtered_df = filtered_df[filtered_df['area_name_en'].isin(areas)]
        
        if len(filtered_df) == 0:
            print("No data available for selected areas and property type.")
            return
            
        # Set up the figure
        plt.figure(figsize=(12, 6))
        
        # Box plot of rent distribution by area
        sns.boxplot(x='area_name_en', y='monthly_rent', data=filtered_df)
        
        # Add title and labels
        title = 'Monthly Rent Comparison by Area'
        if property_type:
            title += f' ({property_type})'
        plt.title(title)
        plt.xlabel('Area')
        plt.ylabel('Monthly Rent')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def plot_property_type_comparison(self, contracts_df, area=None):
        """
        Plot comparison of rental prices across different property types.
        
        Parameters:
            contracts_df (pd.DataFrame): DataFrame containing rental contract data.
            area (str, optional): Filter by area name.
        """
        if contracts_df is None or len(contracts_df) == 0:
            print("No data available for visualization.")
            return
            
        # Filter data
        filtered_df = contracts_df.copy()
        
        if area:
            filtered_df = filtered_df[filtered_df['area_name_en'] == area]
        
        if len(filtered_df) == 0:
            print("No data available for selected area.")
            return
            
        # Set up the figure
        plt.figure(figsize=(12, 6))
        
        # Box plot of rent distribution by property type
        sns.boxplot(x='ejari_property_type_en', y='monthly_rent', data=filtered_df)
        
        # Add title and labels
        title = 'Monthly Rent Comparison by Property Type'
        if area:
            title += f' ({area})'
        plt.title(title)
        plt.xlabel('Property Type')
        plt.ylabel('Monthly Rent')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def plot_seasonal_trends(self, contracts_df, property_type=None, area=None):
        """
        Plot seasonal trends in rental prices.
        
        Parameters:
            contracts_df (pd.DataFrame): DataFrame containing rental contract data.
            property_type (str, optional): Filter by property type.
            area (str, optional): Filter by area name.
        """
        if contracts_df is None or len(contracts_df) == 0:
            print("No data available for visualization.")
            return
            
        # Filter data
        filtered_df = contracts_df.copy()
        
        if property_type:
            filtered_df = filtered_df[filtered_df['ejari_property_type_en'] == property_type]
            
        if area:
            filtered_df = filtered_df[filtered_df['area_name_en'] == area]
        
        if len(filtered_df) == 0:
            print("No data available after filtering.")
            return
            
        # Calculate monthly averages
        monthly_avg = filtered_df.groupby('contract_month')['monthly_rent'].mean().reset_index()
        
        # Calculate quarterly averages
        quarterly_avg = filtered_df.groupby('contract_quarter')['monthly_rent'].mean().reset_index()
        
        # Set up the figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Monthly trends
        sns.lineplot(x='contract_month', y='monthly_rent', data=monthly_avg, marker='o', ax=axes[0])
        axes[0].set_title('Monthly Seasonal Trends')
        axes[0].set_xlabel('Month')
        axes[0].set_ylabel('Average Monthly Rent')
        axes[0].set_xticks(range(1, 13))
        
        # Quarterly trends
        sns.barplot(x='contract_quarter', y='monthly_rent', data=quarterly_avg, ax=axes[1])
        axes[1].set_title('Quarterly Seasonal Trends')
        axes[1].set_xlabel('Quarter')
        axes[1].set_ylabel('Average Monthly Rent')
        
        # Add title
        fig.suptitle(f'Seasonal Rental Price Trends{" for " + property_type if property_type else ""}{" in " + area if area else ""}', 
                    fontsize=16)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        plt.show()
    
    def plot_rent_volatility(self, volatility_data):
        """
        Plot rent volatility metrics.
        
        Parameters:
            volatility_data (dict): Dictionary with volatility metrics from calculate_rent_volatility.
        """
        if not volatility_data or 'yearly' not in volatility_data:
            print("No volatility data available for visualization.")
            return
            
        # Convert yearly volatility to DataFrame
        yearly_vol = pd.DataFrame.from_dict(volatility_data['yearly'], orient='index').reset_index()
        yearly_vol.rename(columns={'index': 'year'}, inplace=True)
        
        # Set up the figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot yearly volatility
        sns.lineplot(x='year', y='cv', data=yearly_vol, marker='o', ax=axes[0])
        axes[0].set_title('Yearly Rent Volatility')
        axes[0].set_xlabel('Year')
        axes[0].set_ylabel('Coefficient of Variation (%)')
        
        # Plot mean and std
        sns.lineplot(x='year', y='mean', data=yearly_vol, marker='o', label='Mean Rent', ax=axes[1])
        axes[1].set_title('Yearly Mean Rent and Standard Deviation')
        axes[1].set_xlabel('Year')
        axes[1].set_ylabel('Monthly Rent')
        
        # Add error bands for standard deviation
        axes[1].fill_between(yearly_vol['year'], 
                           yearly_vol['mean'] - yearly_vol['std'],
                           yearly_vol['mean'] + yearly_vol['std'],
                           alpha=0.3)
        
        # Add overall volatility as text
        if 'overall' in volatility_data:
            overall_cv = volatility_data['overall']['coefficient_of_variation']
            fig.text(0.5, 0.01, f'Overall Coefficient of Variation: {overall_cv:.2f}%', 
                    ha='center', fontsize=12)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        plt.show()
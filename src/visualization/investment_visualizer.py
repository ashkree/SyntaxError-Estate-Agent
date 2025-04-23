"""
InvestmentVisualizer: Module for visualizing real estate investment metrics

This module provides visualization functions for investment data, separated from
the core model logic for better architectural organization.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


class InvestmentVisualizer:
    """
    A class for creating visualizations of real estate investment data and analysis results.
    """
    
    def __init__(self, style='whitegrid'):
        """
        Initialize the InvestmentVisualizer.
        
        Parameters:
            style (str): Seaborn style for plots.
        """
        self.style = style
        sns.set_style(style)
        self.fig_size = (16, 12)
    
    def visualize_market_overview(self, properties_df):
        """
        Generate visualizations of the real estate market data.
        
        Parameters:
            properties_df (pd.DataFrame): DataFrame containing property data.
        """
        if properties_df is None or len(properties_df) == 0:
            print("No data available for visualization.")
            return
        
        # Set up the figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=self.fig_size)
        plt.subplots_adjust(hspace=0.3, wspace=0.3)
        
        # 1. Price per sqft by area
        area_prices = properties_df.groupby('area_name')['price_per_sqft'].mean().sort_values()
        sns.barplot(x=area_prices.index, y=area_prices.values, ax=axes[0, 0])
        axes[0, 0].set_title('Average Price per Sqft by Area')
        axes[0, 0].set_xlabel('Area')
        axes[0, 0].set_ylabel('Price per Sqft')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Rental yield by property type
        type_yield = properties_df.groupby('property_type')['rental_yield'].mean().sort_values()
        sns.barplot(x=type_yield.index, y=type_yield.values, ax=axes[0, 1])
        axes[0, 1].set_title('Average Rental Yield by Property Type')
        axes[0, 1].set_xlabel('Property Type')
        axes[0, 1].set_ylabel('Rental Yield (%)')
        
        # 3. Property price vs. size scatter plot
        sns.scatterplot(
            x='size_sqft', 
            y='property_price', 
            hue='property_type',
            data=properties_df, 
            ax=axes[1, 0]
        )
        axes[1, 0].set_title('Property Price vs Size')
        axes[1, 0].set_xlabel('Size (sqft)')
        axes[1, 0].set_ylabel('Property Price')
        
        # 4. Price to rent ratio by area
        area_ptr = properties_df.groupby('area_name')['price_to_rent_ratio'].mean().sort_values()
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
        fig, axes = plt.subplots(2, 2, figsize=self.fig_size)
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
    
    def visualize_yield_comparison(self, properties_df):
        """
        Visualize rental yield comparisons across different areas and property types.
        
        Parameters:
            properties_df (pd.DataFrame): DataFrame containing property data.
        """
        if properties_df is None or len(properties_df) == 0:
            print("No data available for visualization.")
            return
        
        # Set up the figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. Top 10 areas by yield
        top_areas_yield = properties_df.groupby('area_name')['rental_yield'].mean().nlargest(10)
        sns.barplot(y=top_areas_yield.index, x=top_areas_yield.values, ax=axes[0])
        axes[0].set_title('Top 10 Areas by Rental Yield')
        axes[0].set_xlabel('Average Rental Yield (%)')
        axes[0].set_ylabel('Area')
        
        # 2. Yield vs Property Type
        sns.boxplot(x='property_type', y='rental_yield', data=properties_df, ax=axes[1])
        axes[1].set_title('Rental Yield Distribution by Property Type')
        axes[1].set_xlabel('Property Type')
        axes[1].set_ylabel('Rental Yield (%)')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_price_trends(self, properties_df, by_area=True):
        """
        Visualize property price trends by area or property type.
        
        Parameters:
            properties_df (pd.DataFrame): DataFrame containing property data.
            by_area (bool): If True, visualize by area; otherwise, by property type.
        """
        if properties_df is None or len(properties_df) == 0:
            print("No data available for visualization.")
            return
        
        plt.figure(figsize=(12, 6))
        
        if by_area:
            # Top 10 areas by average price
            top_areas_price = properties_df.groupby('area_name')['property_price'].mean().nlargest(10)
            sns.barplot(y=top_areas_price.index, x=top_areas_price.values)
            plt.title('Top 10 Areas by Average Property Price')
            plt.xlabel('Average Property Price')
            plt.ylabel('Area')
        else:
            # Average price by property type
            avg_price_by_type = properties_df.groupby('property_type')['property_price'].mean().sort_values()
            sns.barplot(x=avg_price_by_type.index, y=avg_price_by_type.values)
            plt.title('Average Property Price by Type')
            plt.xlabel('Property Type')
            plt.ylabel('Average Property Price')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def visualize_price_per_sqft_analysis(self, properties_df):
        """
        Visualize price per square foot analysis.
        
        Parameters:
            properties_df (pd.DataFrame): DataFrame containing property data.
        """
        if properties_df is None or len(properties_df) == 0:
            print("No data available for visualization.")
            return
        
        # Set up the figure with subplots
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # 1. Top 10 areas by price per sqft
        top_areas_psf = properties_df.groupby('area_name')['price_per_sqft'].mean().nlargest(10)
        sns.barplot(y=top_areas_psf.index, x=top_areas_psf.values, ax=axes[0])
        axes[0].set_title('Top 10 Areas by Price per Square Foot')
        axes[0].set_xlabel('Price per Square Foot')
        axes[0].set_ylabel('Area')
        
        # 2. Price per sqft vs property size
        sns.scatterplot(
            x='size_sqft',
            y='price_per_sqft',
            hue='property_type',
            data=properties_df,
            ax=axes[1]
        )
        
        # Add regression line to show the trend
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            properties_df['size_sqft'], properties_df['price_per_sqft']
        )
        axes[1].plot(
            properties_df['size_sqft'], 
            intercept + slope * properties_df['size_sqft'], 
            'r--', 
            label=f'Trend Line (R²={r_value**2:.2f})'
        )
        
        axes[1].set_title('Price per Square Foot vs Property Size')
        axes[1].set_xlabel('Property Size (sqft)')
        axes[1].set_ylabel('Price per Square Foot')
        axes[1].legend()
        
        plt.tight_layout()
        plt.show()
    
    def visualize_investment_metrics(self, properties_df, n_areas=5, n_types=3):
        """
        Visualize key investment metrics for decision making.
        
        Parameters:
            properties_df (pd.DataFrame): DataFrame containing property data.
            n_areas (int): Number of top areas to display.
            n_types (int): Number of property types to display.
        """
        if properties_df is None or len(properties_df) == 0:
            print("No data available for visualization.")
            return
        
        # Set up the figure
        fig = plt.figure(figsize=(16, 12))
        
        # Layout: 2x2 grid
        gs = fig.add_gridspec(2, 2)
        
        # 1. Top areas by rental yield
        ax1 = fig.add_subplot(gs[0, 0])
        top_yield_areas = properties_df.groupby('area_name')['rental_yield'].mean().nlargest(n_areas)
        sns.barplot(y=top_yield_areas.index, x=top_yield_areas.values, ax=ax1)
        ax1.set_title(f'Top {n_areas} Areas by Rental Yield')
        ax1.set_xlabel('Average Rental Yield (%)')
        ax1.set_ylabel('Area')
        
        # 2. Property types by price-to-rent ratio (lower is better for investment)
        ax2 = fig.add_subplot(gs[0, 1])
        ptr_by_type = properties_df.groupby('property_type')['price_to_rent_ratio'].mean().nsmallest(n_types)
        sns.barplot(y=ptr_by_type.index, x=ptr_by_type.values, ax=ax2)
        ax2.set_title(f'Best {n_types} Property Types by Price-to-Rent Ratio')
        ax2.set_xlabel('Price-to-Rent Ratio (Lower is Better)')
        ax2.set_ylabel('Property Type')
        
        # 3. Scatter plot: Rental Yield vs Price-to-Rent Ratio
        ax3 = fig.add_subplot(gs[1, 0])
        sns.scatterplot(
            x='price_to_rent_ratio', 
            y='rental_yield', 
            hue='property_type',
            size='property_price',
            sizes=(50, 200),
            alpha=0.7,
            data=properties_df,
            ax=ax3
        )
        ax3.set_title('Rental Yield vs Price-to-Rent Ratio')
        ax3.set_xlabel('Price-to-Rent Ratio')
        ax3.set_ylabel('Rental Yield (%)')
        
        # 4. Investment quality matrix: high yield and low price-to-rent ratio is ideal
        ax4 = fig.add_subplot(gs[1, 1])
        
        # Calculate average values for reference lines
        avg_yield = properties_df['rental_yield'].mean()
        avg_ptr = properties_df['price_to_rent_ratio'].mean()
        
        # Create the scatter plot
        scatter = sns.scatterplot(
            x='price_to_rent_ratio',
            y='rental_yield',
            hue='area_name',
            data=properties_df,
            ax=ax4
        )
        
        # Add reference lines for average values
        ax4.axhline(y=avg_yield, color='r', linestyle='--', alpha=0.5)
        ax4.axvline(x=avg_ptr, color='r', linestyle='--', alpha=0.5)
        
        # Add quadrant labels
        ax4.text(
            properties_df['price_to_rent_ratio'].max() * 0.9, 
            properties_df['rental_yield'].max() * 0.9,
            "Ideal Investment\n(High Yield, Low P/R)",
            ha='right',
            va='top',
            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3)
        )
        
        ax4.set_title('Investment Quality Matrix')
        ax4.set_xlabel('Price-to-Rent Ratio (Lower is Better)')
        ax4.set_ylabel('Rental Yield (%) (Higher is Better)')
        
        # Adjust legend for readability
        handles, labels = scatter.get_legend_handles_labels()
        ax4.legend(handles=handles[:5], labels=labels[:5], title="Top Areas")
        
        plt.tight_layout()
        plt.show()
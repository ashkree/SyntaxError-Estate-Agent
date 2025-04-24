#!/usr/bin/env python
"""
recommendation_engine.py

A data-focused recommendation engine that consumes outputs from the RealEstateInvestmentModel
to generate structured investment recommendations based on user context.

Outputs investment data insights for:
- Undervalued areas
- Overvalued areas
- Growth opportunity areas
- High yield areas
"""
import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional, Union


class RecommendationEngine:
    """
    Real estate recommendation engine that generates tailored recommendations
    based on investment model outputs and user context.
    """

    def __init__(self, property_scores_path: str, area_scores_path: str):
        """
        Initialize the recommendation engine with model outputs.

        Args:
            property_scores_path: Path to property investment scores CSV
            area_scores_path: Path to area investment scores CSV
        """
        self.property_scores = pd.read_csv(property_scores_path)
        self.area_scores = pd.read_csv(area_scores_path)

        # User context placeholder - will be set via set_user_context
        self.user_context = {
            "user_type": None,  # "investor" or "property_owner"
            "investment_horizon": None,  # "short_term", "medium_term", "long_term"
            "risk_profile": None,  # "conservative", "moderate", "aggressive"
            "budget_min": None,
            "budget_max": None,
            "target_areas": [],
            "property_types": [],
            "existing_properties": []  # For property owners
        }

        # Decision thresholds with defaults
        self.thresholds = {
            "high_investment_score": 60,
            "undervalued_threshold": 10,  # % undervalued
            "high_yield_threshold": 5.5,  # % rental yield
            "growth_potential_threshold": 15,  # % rental upside
            "auto_recommend_threshold": 75  # Auto recommendation threshold
        }

        # Create opportunity labels for properties
        self._prepare_data()

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(levelname)s %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _prepare_data(self):
        """Prepare and categorize the data for recommendations."""
        # Add opportunity type classification to properties
        conditions = [
            (self.property_scores['ValScore'] > self.property_scores['YieldScore']) &
            (self.property_scores['ValScore'] >
             self.property_scores['RentScore']),

            (self.property_scores['YieldScore'] > self.property_scores['ValScore']) &
            (self.property_scores['YieldScore'] >
             self.property_scores['RentScore']),

            (self.property_scores['RentScore'] > self.property_scores['ValScore']) &
            (self.property_scores['RentScore'] >
             self.property_scores['YieldScore'])
        ]
        choices = ['Undervalued', 'High Yield', 'Rental Upside']
        self.property_scores['Opportunity_Type'] = np.select(
            conditions, choices, default='Balanced'
        )

        # Add opportunity type to areas based on dominant property types
        area_opportunity_counts = (
            self.property_scores
            .groupby(['AreaCode', 'Opportunity_Type'])
            .size()
            .unstack(fill_value=0)
        )

        # Find dominant opportunity type for each area
        self.area_scores['Dominant_Opportunity'] = (
            area_opportunity_counts
            .idxmax(axis=1)
            .reindex(self.area_scores['AreaCode'])
            .values
        )

        # Add price tier categories
        price_quantiles = self.property_scores['PropertyValuation'].quantile([
                                                                             0.33, 0.67])
        self.property_scores['Price_Tier'] = pd.cut(
            self.property_scores['PropertyValuation'],
            bins=[0, price_quantiles.iloc[0],
                  price_quantiles.iloc[1], float('inf')],
            labels=['Budget', 'Mid-range', 'Premium']
        )

        # Calculate area growth metrics
        if 'ValDiffPct' in self.area_scores.columns:
            self.area_scores['Growth_Potential'] = (
                self.area_scores['ValDiffPct'] * 0.6 +
                self.area_scores['RentDiffPct'] * 0.4
            )

    def set_user_context(self, user_context: Dict[str, Any]):
        """
        Set or update user context.

        Args:
            user_context: Dictionary with user preferences and constraints
        """
        # Update user context with provided values
        for key, value in user_context.items():
            if key in self.user_context:
                self.user_context[key] = value

        self.logger.info(f"User context updated: {self.user_context}")
        return self.user_context

    def set_decision_thresholds(self, thresholds: Dict[str, float]):
        """
        Set or update decision thresholds.

        Args:
            thresholds: Dictionary with threshold values
        """
        # Update thresholds with provided values
        for key, value in thresholds.items():
            if key in self.thresholds:
                self.thresholds[key] = value

        self.logger.info(f"Decision thresholds updated: {self.thresholds}")
        return self.thresholds

    def _filter_by_user_context(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter properties or areas based on user context.

        Args:
            df: DataFrame to filter (either properties or areas)

        Returns:
            Filtered DataFrame
        """
        filtered_df = df.copy()

        # Apply budget filter if applicable
        if 'PropertyValuation' in filtered_df.columns:
            if self.user_context["budget_min"] is not None:
                filtered_df = filtered_df[
                    filtered_df["PropertyValuation"] >= self.user_context["budget_min"]
                ]

            if self.user_context["budget_max"] is not None:
                filtered_df = filtered_df[
                    filtered_df["PropertyValuation"] <= self.user_context["budget_max"]
                ]

        # Apply area filter if applicable
        if self.user_context["target_areas"] and 'AreaName' in filtered_df.columns:
            if len(self.user_context["target_areas"]) > 0:
                filtered_df = filtered_df[
                    filtered_df["AreaName"].isin(
                        self.user_context["target_areas"])
                ]

        # Apply property type filter if applicable
        if self.user_context["property_types"] and 'PropertyType' in filtered_df.columns:
            if len(self.user_context["property_types"]) > 0:
                filtered_df = filtered_df[
                    filtered_df["PropertyType"].isin(
                        self.user_context["property_types"])
                ]

        return filtered_df

    def _get_recommended_areas(self, top_n: int = 5) -> pd.DataFrame:
        """
        Get recommended areas based on investment scores and user context.

        Args:
            top_n: Number of top areas to recommend

        Returns:
            DataFrame with recommended areas
        """
        # Filter areas by user context
        filtered_areas = self._filter_by_user_context(self.area_scores)

        # Sort by investment score
        recommended_areas = filtered_areas.sort_values(
            'InvestmentScore', ascending=False).head(top_n)

        return recommended_areas

    def _get_undervalued_properties(self, threshold: Optional[float] = None, top_n: int = 10) -> pd.DataFrame:
        """
        Get undervalued properties based on valuation difference.

        Args:
            threshold: Minimum percentage undervalued (defaults to threshold in self.thresholds)
            top_n: Number of top properties to return

        Returns:
            DataFrame with undervalued properties
        """
        if threshold is None:
            threshold = self.thresholds["undervalued_threshold"]

        # Filter properties by user context
        filtered_props = self._filter_by_user_context(self.property_scores)

        # Find undervalued properties
        undervalued = filtered_props[
            (filtered_props['ValDiffPct'] >= threshold) &
            (filtered_props['Opportunity_Type'] == 'Undervalued')
        ]

        # Sort by valuation difference percentage
        top_undervalued = undervalued.sort_values(
            'ValDiffPct', ascending=False).head(top_n)

        return top_undervalued

    def _get_high_yield_properties(self, threshold: Optional[float] = None, top_n: int = 10) -> pd.DataFrame:
        """
        Get high yield properties based on rental yield.

        Args:
            threshold: Minimum yield percentage (defaults to threshold in self.thresholds)
            top_n: Number of top properties to return

        Returns:
            DataFrame with high yield properties
        """
        if threshold is None:
            threshold = self.thresholds["high_yield_threshold"]

        # Filter properties by user context
        filtered_props = self._filter_by_user_context(self.property_scores)

        # Find high yield properties
        high_yield = filtered_props[
            (filtered_props['RentalYield'] >= threshold) &
            (filtered_props['Opportunity_Type'] == 'High Yield')
        ]

        # Sort by yield
        top_yield = high_yield.sort_values(
            'RentalYield', ascending=False).head(top_n)

        return top_yield

    def _get_growth_opportunities(self, threshold: Optional[float] = None, top_n: int = 10) -> pd.DataFrame:
        """
        Get properties with growth potential based on rental upside.

        Args:
            threshold: Minimum rental upside percentage (defaults to threshold in self.thresholds)
            top_n: Number of top properties to return

        Returns:
            DataFrame with growth opportunity properties
        """
        if threshold is None:
            threshold = self.thresholds["growth_potential_threshold"]

        # Filter properties by user context
        filtered_props = self._filter_by_user_context(self.property_scores)

        # Find properties with rental upside
        growth_props = filtered_props[
            (filtered_props['RentDiffPct'] >= threshold) &
            (filtered_props['Opportunity_Type'] == 'Rental Upside')
        ]

        # Sort by rental difference percentage
        top_growth = growth_props.sort_values(
            'RentDiffPct', ascending=False).head(top_n)

        return top_growth

    def _optimize_rent_for_property_owners(self) -> pd.DataFrame:
        """
        Generate rent optimization recommendations for property owners.

        Returns:
            DataFrame with rent optimization recommendations
        """
        # Check if user has existing properties
        if not self.user_context["existing_properties"]:
            return pd.DataFrame()

        # Process existing properties
        owner_properties = pd.DataFrame(
            self.user_context["existing_properties"])

        # Match with our property data where possible
        recommendations = []

        for _, prop in owner_properties.iterrows():
            # Try to find property in our dataset based on area and other attributes
            area_matches = self.property_scores[
                self.property_scores['AreaName'] == prop['area']
            ]

            if len(area_matches) == 0:
                continue

            # Find similar properties (simplified matching)
            similar_props = area_matches

            # Check for size column (might be 'Size_sqft' or 'size_sqft')
            size_column = None
            for col in ['Size_sqft', 'size_sqft']:
                if col in similar_props.columns:
                    size_column = col
                    break

            # Filter by size if the column exists and size is provided in property data
            if size_column and 'size_sqft' in prop:
                size_min = prop['size_sqft'] * 0.8
                size_max = prop['size_sqft'] * 1.2
                similar_props = similar_props[
                    (similar_props[size_column] >= size_min) &
                    (similar_props[size_column] <= size_max)
                ]

            # Filter by bedrooms if applicable
            if 'bedrooms' in prop and 'Bedrooms' in similar_props.columns:
                similar_props = similar_props[
                    similar_props['Bedrooms'] == prop['bedrooms']
                ]

            if len(similar_props) == 0:
                continue

            # Calculate market rent stats
            market_mean_rent = similar_props['AnnualRent'].mean()
            market_median_rent = similar_props['AnnualRent'].median()
            market_p75_rent = similar_props['AnnualRent'].quantile(0.75)
            market_p25_rent = similar_props['AnnualRent'].quantile(0.25)

            # Current rent vs market rent
            current_rent = prop.get('current_rent', 0)
            if current_rent > 0:
                rent_diff_pct = (market_median_rent -
                                 current_rent) / current_rent * 100

                recommendation = {
                    'property_id': prop.get('id', 'Unknown'),
                    'area': prop['area'],
                    'current_rent': current_rent,
                    'market_mean_rent': market_mean_rent,
                    'market_median_rent': market_median_rent,
                    'rent_diff_pct': rent_diff_pct,
                    'recommended_rent': market_median_rent,
                    'rent_range_low': market_p25_rent,
                    'rent_range_high': market_p75_rent,
                    'recommendation': ''
                }

                # Generate recommendation text
                if rent_diff_pct > 10:
                    recommendation['recommendation'] = f"Increase rent by {rent_diff_pct:.1f}% to match market rates"
                elif rent_diff_pct < -10:
                    recommendation[
                        'recommendation'] = f"Your property is rented {abs(rent_diff_pct):.1f}% above market rates"
                else:
                    recommendation['recommendation'] = "Current rent is within market range"

                recommendations.append(recommendation)

        return pd.DataFrame(recommendations)

    def generate_investor_data(self) -> Dict[str, Any]:
        """
        Generate structured investment data for investors.

        Returns:
            Dictionary with comprehensive investment data
        """
        if self.user_context["user_type"] != "investor":
            self.logger.warning("User type is not set to 'investor'")

        # Get detailed data insights
        return self.get_investment_recommendations()

    def generate_property_owner_data(self) -> Dict[str, Any]:
        """
        Generate structured data for property owners.

        Returns:
            Dictionary with rent optimization data and market insights
        """
        if self.user_context["user_type"] != "property_owner":
            self.logger.warning("User type is not set to 'property_owner'")

        # Generate rent optimization recommendations
        rent_recommendations = self._optimize_rent_for_property_owners()

        # Get market insights for relevant areas
        market_insights = {}
        if self.user_context["existing_properties"]:
            owner_areas = set(prop['area']
                              for prop in self.user_context["existing_properties"])

            for area in owner_areas:
                area_data = self.area_scores[self.area_scores['AreaName'] == area]
                if len(area_data) > 0:
                    area_row = area_data.iloc[0]
                    market_insights[area] = {
                        "avg_rental_yield": area_row['RentalYield'],
                        "price_trend": area_row['ValDiffPct'],
                        "rent_trend": area_row['RentDiffPct'],
                        "investment_score": area_row['InvestmentScore']
                    }

        # Prepare recommendations as structured data
        recommendations = {
            "rent_optimization": rent_recommendations.to_dict('records'),
            "market_insights": market_insights,
            "area_metrics": self.get_area_insights()
        }

        return recommendations

    def generate_recommendations(self) -> Dict[str, Any]:
        """
        Generate appropriate data-focused recommendations based on user context.
        Main entry point for getting recommendations.

        Returns:
            Dictionary with structured data
        """
        if self.user_context["user_type"] == "investor":
            return self.generate_investor_data()
        elif self.user_context["user_type"] == "property_owner":
            return self.generate_property_owner_data()
        else:
            self.logger.warning("User type not set or invalid")
            return {"error": "Invalid user type"}

    def get_area_insights(self) -> Dict[str, pd.DataFrame]:
        """
        Generate structured area investment insights focusing on data.

        Returns:
            Dictionary containing DataFrames for different area categories:
            - undervalued_areas: Areas with positive ValDiffPct
            - overvalued_areas: Areas with negative ValDiffPct
            - growth_areas: Areas with high RentDiffPct
            - high_yield_areas: Areas with high RentalYield
            - all_areas: All areas with complete metrics
        """
        # Apply user context filtering if any
        filtered_areas = self._filter_by_user_context(self.area_scores)

        # Get different area types
        undervalued_areas = filtered_areas[
            filtered_areas['ValDiffPct'] >= self.thresholds["undervalued_threshold"]
        ].sort_values('ValDiffPct', ascending=False)

        overvalued_areas = filtered_areas[
            filtered_areas['ValDiffPct'] <= -
            self.thresholds["undervalued_threshold"]
        ].sort_values('ValDiffPct')

        growth_areas = filtered_areas[
            filtered_areas['RentDiffPct'] >= self.thresholds["growth_potential_threshold"]
        ].sort_values('RentDiffPct', ascending=False)

        high_yield_areas = filtered_areas[
            filtered_areas['RentalYield'] >= self.thresholds["high_yield_threshold"]
        ].sort_values('RentalYield', ascending=False)

        # Create combined metrics dataframe with all metrics and a categorization column
        all_areas = filtered_areas.copy()

        # Add area categorization if missing
        if 'Category' not in all_areas.columns:
            # Create a simple category based on available metrics
            conditions = [
                (all_areas['ValDiffPct'] >=
                 self.thresholds["undervalued_threshold"]),
                (all_areas['ValDiffPct'] <= -
                 self.thresholds["undervalued_threshold"]),
                (all_areas['RentDiffPct'] >=
                 self.thresholds["growth_potential_threshold"]),
                (all_areas['RentalYield'] >=
                 self.thresholds["high_yield_threshold"])
            ]
            choices = ['Undervalued', 'Overvalued', 'Growth', 'High Yield']
            all_areas['Category'] = np.select(
                conditions, choices, default='Neutral')

        # Select relevant columns
        area_columns = [
            'AreaCode', 'AreaName', 'InvestmentScore', 'ValScore',
            'YieldScore', 'RentScore', 'LocScore', 'ValDiffPct',
            'RentDiffPct', 'RentalYield', 'LocPremium', 'Category'
        ]

        # Only include columns that exist in the dataframe
        area_columns = [
            col for col in area_columns if col in all_areas.columns]

        # Return filtered dataframes with only existing columns
        return {
            "undervalued_areas": undervalued_areas[
                [col for col in area_columns if col in undervalued_areas.columns]
            ],
            "overvalued_areas": overvalued_areas[
                [col for col in area_columns if col in overvalued_areas.columns]
            ],
            "growth_areas": growth_areas[
                [col for col in area_columns if col in growth_areas.columns]
            ],
            "high_yield_areas": high_yield_areas[
                [col for col in area_columns if col in high_yield_areas.columns]
            ],
            "all_areas": all_areas[
                [col for col in area_columns if col in all_areas.columns]
            ]
        }

    def get_property_insights(self) -> Dict[str, pd.DataFrame]:
        """
        Generate structured property investment insights.

        Returns:
            Dictionary containing DataFrames for different property categories:
            - undervalued_properties: Properties significantly undervalued
            - high_yield_properties: Properties with high rental yields
            - growth_properties: Properties with rental upside potential
            - all_properties: All properties with complete metrics
        """
        # Apply user context filtering
        filtered_props = self._filter_by_user_context(self.property_scores)

        # Get undervalued properties
        undervalued = filtered_props[
            (filtered_props['ValDiffPct'] >= self.thresholds["undervalued_threshold"]) &
            (filtered_props['Opportunity_Type'] == 'Undervalued')
        ].sort_values('ValDiffPct', ascending=False)

        # Get high yield properties
        high_yield = filtered_props[
            (filtered_props['RentalYield'] >= self.thresholds["high_yield_threshold"]) &
            (filtered_props['Opportunity_Type'] == 'High Yield')
        ].sort_values('RentalYield', ascending=False)

        # Get growth properties
        growth = filtered_props[
            (filtered_props['RentDiffPct'] >= self.thresholds["growth_potential_threshold"]) &
            (filtered_props['Opportunity_Type'] == 'Rental Upside')
        ].sort_values('RentDiffPct', ascending=False)

        # Select relevant columns
        property_columns = [
            'AreaCode', 'AreaName', 'PropertyValuation', 'AnnualRent',
            'PredVal', 'PredRent', 'InvestmentScore', 'ValScore',
            'YieldScore', 'RentScore', 'LocScore', 'ValDiffPct',
            'RentDiffPct', 'RentalYield', 'Opportunity_Type'
        ]

        # Only include columns that exist
        property_columns = [
            col for col in property_columns if col in filtered_props.columns]

        return {
            "undervalued_properties": undervalued[property_columns],
            "high_yield_properties": high_yield[property_columns],
            "growth_properties": growth[property_columns],
            "all_properties": filtered_props[property_columns]
        }

    def get_investment_recommendations(self, top_n: int = 5) -> Dict[str, Any]:
        """
        Generate pure data-based investment recommendations.

        Args:
            top_n: Number of top results to include for each category

        Returns:
            Dictionary with structured investment data insights
        """
        # Get area insights
        area_insights = self.get_area_insights()

        # Get property insights
        property_insights = self.get_property_insights()

        # Create structured data response
        recommendations = {
            # Best areas to invest in by category
            "top_areas": {
                "overall": self._get_recommended_areas(top_n).to_dict('records'),
                "undervalued": area_insights["undervalued_areas"].head(top_n).to_dict('records'),
                "growth_potential": area_insights["growth_areas"].head(top_n).to_dict('records'),
                "high_yield": area_insights["high_yield_areas"].head(top_n).to_dict('records')
            },

            # Top properties by category
            "top_properties": {
                "undervalued": property_insights["undervalued_properties"].head(top_n).to_dict('records'),
                "high_yield": property_insights["high_yield_properties"].head(top_n).to_dict('records'),
                "growth_potential": property_insights["growth_properties"].head(top_n).to_dict('records')
            },

            # Investment metrics aggregations
            "market_metrics": {
                "avg_yield": self.property_scores['RentalYield'].mean(),
                "median_yield": self.property_scores['RentalYield'].median(),
                "yield_percentiles": {
                    "p25": self.property_scores['RentalYield'].quantile(0.25),
                    "p75": self.property_scores['RentalYield'].quantile(0.75)
                },
                "avg_price_per_sqft": self.property_scores['PricePerSqFt'].mean() if 'PricePerSqFt' in self.property_scores.columns else None,
                "area_count": len(area_insights["all_areas"]),
                "property_count": len(property_insights["all_properties"])
            }
        }

        return recommendations

#!/usr/bin/env python3
"""
recommendation_pipeline.py

Orchestrates the end-to-end real estate investment recommendation pipeline
using the pretrained RealEstateInvestmentModel, RecommendationEngine and DecisionEngine.

This updated version enhances the pipeline with:
- More robust error handling
- Extended configuration options
- Detailed logging
- Support for custom model parameters
- Integration with visualization capabilities
"""
from RealEstateInvestmentModel import RealEstateInvestmentModel
from DecisionEngine import DecisionEngine
from RecommendationEngine import RecommendationEngine
import logging
import json
import time
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np


# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "data_path": "data/training_data/investment/processed_investment_data.csv",
    "model_dir": "src/models/pretrained_models",
    "report_dir": "reports/investment",
    "property_scores_filename": "property_investment_scores.csv",
    "area_scores_filename": "area_investment_scores.csv",
    "recommendations_filename": "investment_recommendations.json",
    "decisions_filename": "investment_decisions.json",
    "visualizations_dir": "visualizations",
    "log_level": "INFO",
    "default_user_context": {
        "user_type": "investor",              # "investor" or "property_owner"
        "investment_horizon": "medium_term",  # "short_term", "medium_term", "long_term"
        "risk_profile": "moderate",           # "conservative", "moderate", "aggressive"
        "budget_min": 300000,
        "budget_max": 900000,
        "target_areas": [],                   # List of area names to focus on
        "property_types": [1, 2],             # Property type IDs of interest
        "existing_properties": []             # For property owners
    },
    "recommendation_thresholds": {
        "high_investment_score": 60,
        "undervalued_threshold": 10.0,        # % undervalued
        "high_yield_threshold": 5.5,          # % rental yield
        "growth_potential_threshold": 15.0,   # % rental upside
        "auto_recommend_threshold": 75        # Auto recommendation threshold
    },
    "decision_thresholds": {
        "auto_buy_undervalued": 15.0,         # Auto-buy if undervalued by more than 15%
        "auto_sell_overvalued": 10.0,         # Auto-sell if overvalued by more than 10%
        # Auto increase rent if below market by more than 12%
        "auto_rent_increase": 12.0,
        "high_growth_threshold": 20.0,        # High growth area threshold
        "exceptional_yield_threshold": 7.0,   # Exceptional yield threshold
        "auto_decision_threshold": 85         # Confidence threshold for auto decisions
    },
    "report_format": "json",                  # "json" or "csv"
    # Whether to generate visualization charts
    "generate_visualizations": True,
    # Include all properties in reports or just top ones
    "include_all_properties": False,
    # Maximum number of recommendations to include
    "max_recommendations": 20,
    "notification_settings": {
        "email_notifications": False,
        "email_recipients": [],
        "notify_on_exceptional_opportunities": True,
        "opportunity_notification_threshold": 80
    }
}


class InvestmentRecommendationPipeline:
    """
    Pipeline orchestrator that loads pretrained investment models and generates
    real estate investment recommendations and decisions.

    The pipeline handles:
    1. Loading of pretrained investment models
    2. Scoring properties and areas
    3. Generating tailored recommendations based on user context
    4. Making automated decisions based on thresholds
    5. Producing structured output reports
    6. Optional visualization generation
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the pipeline with configuration.

        Args:
            config: Dictionary with pipeline configuration parameters
        """
        self.config = config
        self.logger = self._configure_logging()
        self._setup_paths()

        # Components will be instantiated in init steps
        self.model: Optional[RealEstateInvestmentModel] = None
        self.rec_engine: Optional[RecommendationEngine] = None
        self.dec_engine: Optional[DecisionEngine] = None

        # Status tracking
        self.initialized = False
        self.run_timestamp = None
        self.run_duration = None
        self.last_run_status = None

    def _configure_logging(self) -> logging.Logger:
        """Configure logging based on settings."""
        log_level = getattr(logging, self.config.get("log_level", "INFO"))
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        logger = logging.getLogger(self.__class__.__name__)
        return logger

    def _setup_paths(self) -> None:
        """Set up file paths for data, models, and reports."""
        # Essential paths
        self.data_path = Path(self.config["data_path"])
        self.model_dir = Path(self.config["model_dir"])
        self.report_dir = Path(self.config["report_dir"])
        self.report_dir.mkdir(parents=True, exist_ok=True)

        # Output file paths
        self.property_scores_file = self.report_dir / \
            self.config["property_scores_filename"]
        self.area_scores_file = self.report_dir / \
            self.config["area_scores_filename"]
        self.recommendations_file = self.report_dir / \
            self.config["recommendations_filename"]
        self.decisions_file = self.report_dir / \
            self.config["decisions_filename"]

        # Visualization directory
        if self.config.get("generate_visualizations", False):
            self.viz_dir = self.report_dir / self.config["visualizations_dir"]
            self.viz_dir.mkdir(parents=True, exist_ok=True)

    def initialize(self) -> bool:
        """
        Initialize pipeline components but don't run analysis.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            self._init_pretrained_model()

            # Explicitly load data after initializing the model
            if self.model is not None:
                if not os.path.exists(self.data_path):
                    self.logger.error(
                        f"Data file not found at {self.data_path}")
                    return False

                self.model.load_data(str(self.data_path))
                self.logger.info(f"Data loaded from {self.data_path}")

            self.initialized = True
            self.logger.info("Pipeline initialized successfully")
            return True
        except Exception as e:
            self.logger.error(
                f"Failed to initialize pipeline: {str(e)}", exc_info=True)
            self.initialized = False
            return False

    def run(self, user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute the full pipeline using pretrained models and return structured results.

        Args:
            user_context: Optional user context dictionary (uses default if not provided)

        Returns:
            Dictionary with structured pipeline results
        """
        if user_context is None:
            user_context = self.config["default_user_context"]

        self.logger.info(
            f"Starting pipeline run for user type: {user_context['user_type']}")
        self.run_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        start_time = time.time()

        try:
            # 1. Initialize if not already done
            if not self.initialized:
                if not self.initialize():
                    raise RuntimeError("Failed to initialize pipeline")

            # 2. Generate property and area scores
            prop_scores, area_scores = self._generate_scores()

            # Save scores to disk
            prop_scores.to_csv(self.property_scores_file, index=False)
            area_scores.to_csv(self.area_scores_file, index=False)
            self.logger.info(
                f"Investment scores saved to {self.property_scores_file} and {self.area_scores_file}")

            # 3. Initialize recommendation engine
            self._init_recommendation_engine(
                user_context, prop_scores, area_scores)

            # 4. Generate recommendations based on user type
            recommendations = self._generate_recommendations(user_context)

            # Save recommendations to disk
            self._save_recommendations(recommendations)

            # 5. Initialize decision engine
            self._init_decision_engine()

            # 6. Generate decisions
            decisions = self._generate_decisions()

            # Save decisions to disk
            with open(self.decisions_file, 'w') as f:
                json.dump(decisions, f, indent=2, default=str)
            self.logger.info(f"Decisions exported to {self.decisions_file}")

            # 7. Generate visualizations if enabled
            if self.config.get("generate_visualizations", False):
                self._generate_visualizations(
                    prop_scores, area_scores, recommendations, decisions)

            # 8. Update run statistics
            self.run_duration = time.time() - start_time
            self.last_run_status = "success"

            # 9. Return structured output
            self.logger.info(
                f"Pipeline run completed successfully in {self.run_duration:.2f} seconds")
            return self._prepare_output(user_context, prop_scores, area_scores, recommendations, decisions)

        except Exception as e:
            self.last_run_status = "error"
            self.run_duration = time.time() - start_time
            self.logger.error(f"Pipeline run failed: {str(e)}", exc_info=True)

            # Return error information
            return {
                "status": "error",
                "error": str(e),
                "timestamp": self.run_timestamp,
                "duration": self.run_duration
            }

    def _init_pretrained_model(self) -> None:
        """
        Initialize the model and train it directly without trying to load pretrained models.
        """
        self.logger.info("Initializing real estate investment model")

        # Create model directory if it doesn't exist
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Initialize model with path to data
        self.model = RealEstateInvestmentModel(data_path=str(self.data_path))

        # Ensure data directory exists
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found at {self.data_path}")

        # Load data if not already loaded
        if self.model.data is None:
            self.model.load_data(str(self.data_path))
            self.logger.info(f"Data loaded from {self.data_path}")

        # Train models directly
        self.logger.info("Training new models...")
        models, metrics = self.model.train()

        # Save trained models
        self.model.save_models(str(self.model_dir))
        self.logger.info(f"New models trained and saved to {self.model_dir}")

        # Print some metrics about the trained models
        val_metrics = metrics.get('PropertyValuation', {})
        rent_metrics = metrics.get('AnnualRent', {})

        self.logger.info(
            f"PropertyValuation model - R²: {val_metrics.get('r2', 0):.3f}, MAPE: {val_metrics.get('mape', 0):.2f}%")
        self.logger.info(
            f"AnnualRent model - R²: {rent_metrics.get('r2', 0):.3f}, MAPE: {rent_metrics.get('mape', 0):.2f}%")

    def _generate_scores(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute investment scores for properties and areas using loaded models.

        Returns:
            Tuple of (property_scores, area_scores) DataFrames
        """
        if self.model is None:
            raise RuntimeError(
                "Model not initialized. Call initialize() first.")

        self.logger.info(
            "Generating investment scores for properties and areas")

        # Check if data is loaded
        if not hasattr(self.model, 'data') or self.model.data is None:
            self.logger.info(
                "Data not loaded in model, attempting to load now")
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(
                    f"Data file not found at {self.data_path}")
            self.model.load_data(str(self.data_path))

        prop_scores = self.model.score_investment()
        area_scores = self.model.get_area_investment_scores()

        self.logger.info(
            f"Generated scores for {len(prop_scores)} properties and {len(area_scores)} areas")
        return prop_scores, area_scores

    def _init_recommendation_engine(self, user_context: Dict[str, Any],
                                    prop_scores: Optional[pd.DataFrame] = None,
                                    area_scores: Optional[pd.DataFrame] = None) -> None:
        """
        Initialize the recommendation engine with scores and user context.

        Args:
            user_context: User preferences and constraints
            prop_scores: Optional DataFrame with property scores (reads from file if not provided)
            area_scores: Optional DataFrame with area scores (reads from file if not provided)
        """
        self.logger.info("Initializing recommendation engine")

        # Use provided dataframes or read from disk
        if prop_scores is None or area_scores is None:
            self.rec_engine = RecommendationEngine(
                property_scores_path=str(self.property_scores_file),
                area_scores_path=str(self.area_scores_file)
            )
        else:
            # Create temporary files for the recommendation engine
            # since it expects file paths (could be improved in future versions)
            temp_prop_file = self.report_dir / "temp_property_scores.csv"
            temp_area_file = self.report_dir / "temp_area_scores.csv"

            prop_scores.to_csv(temp_prop_file, index=False)
            area_scores.to_csv(temp_area_file, index=False)

            self.rec_engine = RecommendationEngine(
                property_scores_path=str(temp_prop_file),
                area_scores_path=str(temp_area_file)
            )

            # Clean up temporary files
            if os.path.exists(temp_prop_file):
                os.remove(temp_prop_file)
            if os.path.exists(temp_area_file):
                os.remove(temp_area_file)

        # Set user context and thresholds
        self.rec_engine.set_user_context(user_context)
        self.rec_engine.set_decision_thresholds(
            self.config["recommendation_thresholds"])
        self.logger.info(
            "Recommendation engine initialized with user context and thresholds")

    def _generate_recommendations(self, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate recommendations based on user type.

        Args:
            user_context: User preferences and constraints

        Returns:
            Dictionary with structured recommendations
        """
        if self.rec_engine is None:
            raise RuntimeError("Recommendation engine not initialized")

        self.logger.info(
            f"Generating recommendations for user type: {user_context['user_type']}")

        # Generate appropriate recommendations based on user type
        if user_context["user_type"] == "investor":
            recommendations = self.rec_engine.generate_investor_data()
            self.logger.info("Generated investor recommendations")
        elif user_context["user_type"] == "property_owner":
            recommendations = self.rec_engine.generate_property_owner_data()
            self.logger.info("Generated property owner recommendations")
        else:
            self.logger.warning(
                f"Unknown user type: {user_context['user_type']}, defaulting to investor")
            recommendations = self.rec_engine.generate_investor_data()

        return recommendations

    def _save_recommendations(self, recommendations: Dict[str, Any]) -> None:
        """
        Save recommendations to disk in the configured format.

        Args:
            recommendations: Dictionary with structured recommendations
        """
        # Save as JSON (default)
        with open(self.recommendations_file, 'w') as f:
            json.dump(recommendations, f, indent=2, default=str)
        self.logger.info(
            f"Recommendations exported to {self.recommendations_file}")

        # Save as CSV if configured (only for specific recommendation types)
        if self.config.get("report_format") == "csv":
            try:
                # Save top areas recommendations
                if "top_areas" in recommendations:
                    for area_type, areas in recommendations["top_areas"].items():
                        if areas and isinstance(areas, list):
                            df = pd.DataFrame(areas)
                            csv_file = self.report_dir / \
                                f"top_{area_type}_areas.csv"
                            df.to_csv(csv_file, index=False)

                # Save top properties recommendations
                if "top_properties" in recommendations:
                    for prop_type, props in recommendations["top_properties"].items():
                        if props and isinstance(props, list):
                            df = pd.DataFrame(props)
                            csv_file = self.report_dir / \
                                f"top_{prop_type}_properties.csv"
                            df.to_csv(csv_file, index=False)

                self.logger.info("CSV exports of recommendations completed")
            except Exception as e:
                self.logger.warning(
                    f"Error exporting recommendations as CSV: {str(e)}")

    def _init_decision_engine(self) -> None:
        """
        Initialize the decision engine with the recommendation engine and thresholds.
        """
        if self.rec_engine is None:
            raise RuntimeError(
                "Recommendation engine must be initialized first")

        self.logger.info("Initializing decision engine")
        self.dec_engine = DecisionEngine(self.rec_engine)

        # Set decision thresholds
        self.dec_engine.set_decision_thresholds(
            self.config["decision_thresholds"])

        # Set auto decision threshold if specified
        if "auto_decision_threshold" in self.config["decision_thresholds"]:
            self.dec_engine.set_auto_decision_threshold(
                self.config["decision_thresholds"]["auto_decision_threshold"]
            )

        self.logger.info("Decision engine initialized with thresholds")

    def _generate_decisions(self) -> Dict[str, Any]:
        """
        Generate automated decisions based on recommendations and thresholds.

        Returns:
            Dictionary with structured decisions and analysis
        """
        if self.dec_engine is None:
            raise RuntimeError("Decision engine not initialized")

        self.logger.info("Generating automated decisions")
        decisions = self.dec_engine.generate_auto_decisions()

        # Log decision statistics
        num_decisions = len(decisions.get("decisions", []))
        num_opportunities = sum(len(opps) for opps in decisions.get("opportunities", {}).values()
                                if isinstance(opps, list))

        self.logger.info(
            f"Generated {num_decisions} decisions and identified {num_opportunities} opportunities")
        return decisions

    def _generate_visualizations(self, prop_scores: pd.DataFrame, area_scores: pd.DataFrame,
                                 recommendations: Dict[str, Any], decisions: Dict[str, Any]) -> None:
        """
        Generate visualizations of the investment data and recommendations.

        Args:
            prop_scores: DataFrame with property scores
            area_scores: DataFrame with area scores
            recommendations: Dictionary with structured recommendations
            decisions: Dictionary with structured decisions
        """
        self.logger.info(
            "Visualization generation is enabled, but requires external visualization code")
        self.logger.info(
            f"Visualization files would be saved to {self.viz_dir}")

        # This is a placeholder for visualization generation
        # In a real implementation, you would call visualization functions here
        # Similar to those in investment_model_training_pipeline.py

        # Example visualization that could be implemented:
        # 1. Top Areas by Investment Score
        # 2. Score Components for Top Areas
        # 3. Rental Yield vs Price per Sqft
        # 4. Opportunity Type Distribution
        # 5. Decision Distribution
        # 6. Property vs Area Investment Scores

        pass  # Actual visualization implementation would go here

    def _prepare_output(self, user_context: Dict[str, Any], prop_scores: pd.DataFrame,
                        area_scores: pd.DataFrame, recommendations: Dict[str, Any],
                        decisions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare the final structured output with all pipeline results.

        Args:
            user_context: User preferences and constraints
            prop_scores: DataFrame with property scores
            area_scores: DataFrame with area scores
            recommendations: Dictionary with structured recommendations
            decisions: Dictionary with structured decisions

        Returns:
            Dictionary with comprehensive pipeline results
        """
        # Determine how many properties to include in output
        include_all = self.config.get("include_all_properties", False)
        max_items = self.config.get("max_recommendations", 20)

        # Convert DataFrames to records or just include top items
        if include_all:
            property_scores_out = prop_scores.to_dict('records')
            area_scores_out = area_scores.to_dict('records')
        else:
            property_scores_out = prop_scores.sort_values(
                'InvestmentScore', ascending=False).head(max_items).to_dict('records')
            area_scores_out = area_scores.sort_values(
                'InvestmentScore', ascending=False).head(max_items).to_dict('records')

        # Construct output with metadata
        output = {
            "status": "success",
            "timestamp": self.run_timestamp,
            "duration": self.run_duration,
            "user_context": user_context,
            "property_scores": property_scores_out,
            "area_scores": area_scores_out,
            "recommendations": recommendations,
            "decisions": decisions,
            "metadata": {
                "total_properties": len(prop_scores),
                "total_areas": len(area_scores),
                "config": {
                    k: v for k, v in self.config.items()
                    if k not in ["default_user_context", "notification_settings"]
                }
            }
        }

        return output

    def get_exceptional_opportunities(self, threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Get a list of exceptional investment opportunities above a threshold.

        Args:
            threshold: Optional investment score threshold (defaults to exceptional_yield_threshold)

        Returns:
            List of dictionaries with exceptional opportunities
        """
        if self.model is None or not hasattr(self.model, 'score_investment'):
            raise RuntimeError(
                "Model not initialized or doesn't support scoring")

        # Use provided threshold or get from config
        if threshold is None:
            threshold = self.config["decision_thresholds"].get(
                "exceptional_yield_threshold", 7.0)

        # Generate scores if needed
        try:
            prop_scores = self.model.score_investment()

            # Filter for exceptional opportunities
            exceptional = prop_scores[
                (prop_scores['InvestmentScore'] >= 70) &
                (prop_scores['RentalYield'] >= threshold)
            ].sort_values('InvestmentScore', ascending=False)

            # Convert to list of dictionaries
            return exceptional.head(10).to_dict('records')

        except Exception as e:
            self.logger.error(
                f"Error finding exceptional opportunities: {str(e)}")
            return []

    def generate_summary_report(self) -> Dict[str, Any]:
        """
        Generate a high-level summary report of investment opportunities.

        Returns:
            Dictionary with summary metrics and top opportunities
        """
        if not self.initialized:
            raise RuntimeError("Pipeline not initialized")

        try:
            # Generate scores
            prop_scores, area_scores = self._generate_scores()

            # Get top areas and properties
            top_areas = area_scores.sort_values(
                'InvestmentScore', ascending=False).head(5)
            top_properties = prop_scores.sort_values(
                'InvestmentScore', ascending=False).head(5)

            # Calculate summary metrics
            summary = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "market_summary": {
                    "avg_rental_yield": prop_scores['RentalYield'].mean(),
                    "median_rental_yield": prop_scores['RentalYield'].median(),
                    "avg_investment_score": prop_scores['InvestmentScore'].mean(),
                    "top_opportunity_types": prop_scores['Opportunity_Type'].value_counts().to_dict(),
                    "property_count": len(prop_scores),
                    "area_count": len(area_scores)
                },
                "top_areas": top_areas.to_dict('records'),
                "top_properties": top_properties.to_dict('records')
            }

            return summary

        except Exception as e:
            self.logger.error(f"Error generating summary report: {str(e)}")
            return {"status": "error", "error": str(e)}

    def optimize_rent_for_property(self, property_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate rent optimization recommendation for a single property.

        Args:
            property_details: Dictionary with property details including area, size, bedrooms, etc.

        Returns:
            Dictionary with rent optimization recommendation
        """
        if self.rec_engine is None:
            # Initialize recommendation engine if not already done
            if not self.initialized:
                self.initialize()

            prop_scores, area_scores = self._generate_scores()

            # Initialize with property owner context
            user_context = self.config["default_user_context"].copy()
            user_context["user_type"] = "property_owner"
            user_context["existing_properties"] = [property_details]

            self._init_recommendation_engine(
                user_context, prop_scores, area_scores)
        else:
            # Update user context
            user_context = self.rec_engine.user_context.copy()
            user_context["user_type"] = "property_owner"
            user_context["existing_properties"] = [property_details]
            self.rec_engine.set_user_context(user_context)

        # Generate rent optimization
        recommendations = self.rec_engine.generate_property_owner_data()

        # Extract rent optimization for the specific property
        if "rent_optimization" in recommendations and recommendations["rent_optimization"]:
            return {
                "property": property_details,
                "recommendation": recommendations["rent_optimization"][0],
                "market_insights": recommendations.get("market_insights", {})
            }
        else:
            return {
                "property": property_details,
                "status": "No recommendation available",
                "reason": "Could not find similar properties in the dataset"
            }


if __name__ == "__main__":
    # Example usage
    pipeline = InvestmentRecommendationPipeline(DEFAULT_CONFIG)

    # Initialize the pipeline
    pipeline.initialize()

    # Run the pipeline with default user context
    results = pipeline.run()

    print("Pipeline completed. Results available in reports directory.")

    # Example of generating a summary report
    summary = pipeline.generate_summary_report()
    print(
        f"Summary report generated with {len(summary['top_areas'])} top areas and {len(summary['top_properties'])} top properties")

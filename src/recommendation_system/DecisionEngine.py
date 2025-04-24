#!/usr/bin/env python
"""
decision_engine.py

A decision engine that works with the RecommendationEngine to automatically
identify exceptional investment opportunities and generate actionable decisions.
"""
import datetime
import pandas as pd
from typing import Dict, List, Any, Optional


class DecisionEngine:
    """
    Decision engine that makes automated decisions based on investment data
    and specified thresholds.
    """

    def __init__(self, recommendation_engine):
        """
        Initialize decision engine with recommendation engine.

        Args:
            recommendation_engine: RecommendationEngine instance
        """
        self.recommendation_engine = recommendation_engine
        self.decision_history = []
        self.auto_decision_threshold = 85  # Confidence threshold for auto decisions

        # Decision thresholds
        self.thresholds = {
            "auto_buy_undervalued": 15.0,    # Auto-buy if undervalued by more than 15%
            "auto_sell_overvalued": 10.0,    # Auto-sell if overvalued by more than 10%
            "auto_rent_increase": 12.0,      # Auto increase rent if below market by more than 12%
            "high_growth_threshold": 20.0,   # High growth area threshold
            "exceptional_yield_threshold": 7.0  # Exceptional yield threshold
        }

    def set_auto_decision_threshold(self, threshold: float):
        """Set threshold for automatic decisions."""
        self.auto_decision_threshold = threshold

    def set_decision_thresholds(self, thresholds: Dict[str, float]):
        """Update decision thresholds."""
        for key, value in thresholds.items():
            if key in self.thresholds:
                self.thresholds[key] = value

    def analyze_investment_data(self) -> Dict[str, Any]:
        """
        Analyze investment data to generate actionable insights and decisions.

        Returns:
            Dictionary with structured analysis results and decisions
        """
        # Get detailed investment data
        investment_data = self.recommendation_engine.get_investment_recommendations()
        area_insights = self.recommendation_engine.get_area_insights()

        # Initialize analysis results
        analysis = {
            "decisions": [],
            "opportunities": [],
            "market_summary": {},
            "alert_metrics": {}
        }

        # Find exceptional opportunities
        extreme_undervalued = [
            area for area in area_insights["undervalued_areas"].to_dict('records')
            if area["ValDiffPct"] >= self.thresholds["auto_buy_undervalued"]
        ]

        extreme_overvalued = [
            area for area in area_insights["overvalued_areas"].to_dict('records')
            if area["ValDiffPct"] <= -self.thresholds["auto_sell_overvalued"]
        ]

        high_growth = [
            area for area in area_insights["growth_areas"].to_dict('records')
            if area["RentDiffPct"] >= self.thresholds["high_growth_threshold"]
        ]

        exceptional_yield = [
            area for area in area_insights["high_yield_areas"].to_dict('records')
            if area["RentalYield"] >= self.thresholds["exceptional_yield_threshold"]
        ]

        # Generate automatic decisions for extreme cases
        for area in extreme_undervalued:
            analysis["decisions"].append({
                "action": "BUY",
                "area_code": area["AreaCode"],
                "area_name": area["AreaName"],
                "reason": "Significantly undervalued",
                "metrics": {
                    "undervalued_pct": area["ValDiffPct"],
                    "investment_score": area["InvestmentScore"],
                    "rental_yield": area["RentalYield"]
                }
            })

        for area in extreme_overvalued:
            analysis["decisions"].append({
                "action": "SELL",
                "area_code": area["AreaCode"],
                "area_name": area["AreaName"],
                "reason": "Significantly overvalued",
                "metrics": {
                    "overvalued_pct": -area["ValDiffPct"],
                    "investment_score": area["InvestmentScore"]
                }
            })

        # Generate opportunity alerts
        analysis["opportunities"] = {
            "extreme_undervalued": extreme_undervalued,
            "high_growth": high_growth,
            "exceptional_yield": exceptional_yield
        }

        # Create market summary
        analysis["market_summary"] = {
            "avg_yield": investment_data["market_metrics"]["avg_yield"],
            "median_yield": investment_data["market_metrics"]["median_yield"],
            "undervalued_area_count": len(area_insights["undervalued_areas"]),
            "overvalued_area_count": len(area_insights["overvalued_areas"]),
            "growth_area_count": len(area_insights["growth_areas"]),
            "high_yield_area_count": len(area_insights["high_yield_areas"])
        }

        # Create alert metrics for dashboard
        analysis["alert_metrics"] = {
            "highest_undervalued": extreme_undervalued[0]["ValDiffPct"] if extreme_undervalued else 0,
            "highest_yield": exceptional_yield[0]["RentalYield"] if exceptional_yield else 0,
            "highest_growth": high_growth[0]["RentDiffPct"] if high_growth else 0,
            "decision_count": len(analysis["decisions"]),
            "opportunity_count": len(extreme_undervalued) + len(high_growth) + len(exceptional_yield)
        }

        return analysis

    def make_decisions(self) -> Dict[str, Any]:
        """
        Make decisions based on recommendations and thresholds.

        Returns:
            Dictionary with automatic decisions and notifications
        """
        # Get recommendations
        recommendations = self.recommendation_engine.generate_recommendations()

        # Initialize decision response
        decision_response = {
            "automatic_decisions": [],
            "notifications": []
        }

        # Process automatic recommendations
        for rec in recommendations.get("automatic_recommendations", []):
            decision = {
                "recommendation": rec,
                "decision_type": "automatic" if rec.get("confidence", 0) >= self.auto_decision_threshold else "notification",
                "timestamp": datetime.datetime.now().isoformat()
            }

            if decision["decision_type"] == "automatic":
                decision_response["automatic_decisions"].append(decision)
            else:
                decision_response["notifications"].append(decision)

        # Log automatic decisions
        for decision in decision_response["automatic_decisions"]:
            self.log_decision(decision)

        return decision_response

    def log_decision(self, decision_data: Dict[str, Any]):
        """
        Log a decision for historical tracking and model retraining.

        Args:
            decision_data: Decision data to log
        """
        # Add timestamp if not present
        if "timestamp" not in decision_data:
            decision_data["timestamp"] = datetime.datetime.now().isoformat()

        # Add to history
        self.decision_history.append(decision_data)

        return len(self.decision_history)

    def get_decision_history(self) -> List[Dict[str, Any]]:
        """Get historical decisions."""
        return self.decision_history

    def generate_auto_decisions(self) -> Dict[str, Any]:
        """
        Generate automatic decisions based on investment data.

        Returns:
            Dictionary with decisions and metrics
        """
        # Analyze data and get decisions
        analysis = self.analyze_investment_data()

        # Log auto decisions
        for decision in analysis["decisions"]:
            self.log_decision(decision)

        return analysis

    def get_decision_summary(self) -> Dict[str, Any]:
        """
        Generate a summary of all decisions made.

        Returns:
            Dictionary with decision summary statistics
        """
        if not self.decision_history:
            return {
                "total_decisions": 0,
                "decision_types": {},
                "area_decisions": {}
            }

        # Count decision types
        decision_types = {}
        area_decisions = {}

        for decision in self.decision_history:
            # Count by action type
            action = decision.get("action", "UNKNOWN")
            decision_types[action] = decision_types.get(action, 0) + 1

            # Count by area
            area = decision.get("area_name", "UNKNOWN")
            if area not in area_decisions:
                area_decisions[area] = {
                    "BUY": 0, "SELL": 0, "HOLD": 0, "OTHER": 0}

            if action in ["BUY", "SELL", "HOLD"]:
                area_decisions[area][action] += 1
            else:
                area_decisions[area]["OTHER"] += 1

        return {
            "total_decisions": len(self.decision_history),
            "decision_types": decision_types,
            "area_decisions": area_decisions
        }

    def evaluate_past_decisions(self, current_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate past decisions against current data to measure decision quality.

        Args:
            current_data: Optional current market data for comparison

        Returns:
            Dictionary with decision evaluation metrics
        """
        if not self.decision_history:
            return {"evaluation": "No decisions to evaluate"}

        # If no current data provided, get latest from recommendation engine
        if current_data is None:
            current_data = self.analyze_investment_data()

        # Initialize evaluation metrics
        evaluation = {
            "correct_decisions": 0,
            "incorrect_decisions": 0,
            "accuracy": 0.0,
            "decision_performance": []
        }

        # Simple evaluation example - in real system would be more sophisticated
        # Just checking if areas that were 'BUY' decisions are still undervalued
        current_undervalued = {
            area["AreaName"]: area["ValDiffPct"]
            for area in current_data["opportunities"].get("extreme_undervalued", [])
        }

        current_overvalued = {
            area["AreaName"]: area["ValDiffPct"]
            for area in current_data["opportunities"].get("extreme_overvalued", [])
        }

        for decision in self.decision_history:
            area_name = decision.get("area_name")
            action = decision.get("action")

            # Skip if missing key data
            if not area_name or not action:
                continue

            decision_eval = {
                "area": area_name,
                "action": action,
                "timestamp": decision.get("timestamp"),
                "outcome": "Unknown"
            }

            if action == "BUY":
                # A buy decision was good if area is still undervalued
                if area_name in current_undervalued:
                    decision_eval["outcome"] = "Correct"
                    evaluation["correct_decisions"] += 1
                else:
                    decision_eval["outcome"] = "Incorrect"
                    evaluation["incorrect_decisions"] += 1

            elif action == "SELL":
                # A sell decision was good if area is still overvalued
                if area_name in current_overvalued:
                    decision_eval["outcome"] = "Correct"
                    evaluation["correct_decisions"] += 1
                else:
                    decision_eval["outcome"] = "Incorrect"
                    evaluation["incorrect_decisions"] += 1

            evaluation["decision_performance"].append(decision_eval)

        # Calculate accuracy
        total_evaluated = evaluation["correct_decisions"] + \
            evaluation["incorrect_decisions"]
        if total_evaluated > 0:
            evaluation["accuracy"] = evaluation["correct_decisions"] / \
                total_evaluated

        return evaluation

#!/usr/bin/env python3
"""
run_real_estate_agent.py

Demo script for running the real estate investment agentic AI system.
Shows how to use the pipeline for both property investors and owners.
"""
import json
import os
import sys
import argparse
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

# Import the pipeline components
from RecommendationEnginePipeline import InvestmentRecommendationPipeline, DEFAULT_CONFIG


def setup_demo_directories(data_file_path: Optional[str] = None):
    """
    Create necessary directories for the demo and set up data files.

    Args:
        data_file_path: Optional path to the data file to use

    Returns:
        Tuple of (success, message) indicating if setup succeeded
    """
    # Create essential directories
    dirs = [
        "data/training_data/investment",
        "src/models/pretrained_models",
        "reports/investment",
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

    # Check for data file
    data_dest = "data/training_data/investment/processed_investment_data.csv"

    # If a specific path is provided, try to use that
    if data_file_path and os.path.exists(data_file_path):
        try:
            shutil.copy(data_file_path, data_dest)
            print(f"Data file copied from {data_file_path} to {data_dest}")
            return True, f"Setup complete. Using data from {data_file_path}"
        except Exception as e:
            print(f"Error copying data file: {str(e)}")

    # Otherwise look in common locations
    potential_data_locations = [
        "processed_investment_data.csv",  # Current directory
        "../data/processed_investment_data.csv",  # Parent data directory
        "../processed_investment_data.csv",  # Parent directory
        "data/processed_investment_data.csv",  # Local data directory
    ]

    for location in potential_data_locations:
        if os.path.exists(location):
            try:
                shutil.copy(location, data_dest)
                print(f"Data file copied from {location} to {data_dest}")
                return True, f"Setup complete. Using data from {location}"
            except Exception as e:
                print(f"Error copying data file from {location}: {str(e)}")

    # If no data file was found or copied, create a simple test dataset
    if not os.path.exists(data_dest):
        try:
            # Create a minimal example dataset for testing
            create_sample_dataset(data_dest)
            print(f"Created sample dataset at {data_dest}")
            return True, "Setup complete with sample dataset"
        except Exception as e:
            print(f"Error creating sample dataset: {str(e)}")
            return False, f"Failed to set up data: {str(e)}"

    return False, "No data file found and failed to create sample data"


def create_sample_dataset(output_path: str):
    """
    Create a simple sample dataset for testing when no real data is available.

    Args:
        output_path: Path where to save the sample dataset
    """
    import pandas as pd
    import numpy as np

    # Create a simple dataset with required columns
    areas = ['Downtown', 'Suburb North', 'Suburb East', 'Suburb South', 'Suburb West',
             'Business District', 'University Area', 'Waterfront']

    # Generate 100 sample properties
    np.random.seed(42)  # For reproducibility

    # Basic property features
    n_samples = 100
    data = {
        'AreaCode': np.random.randint(1, len(areas) + 1, n_samples),
        'Size_sqft': np.random.randint(600, 3000, n_samples),
        'Bedrooms': np.random.randint(1, 5, n_samples),
        'Bathrooms': np.random.randint(1, 4, n_samples),
        'Furnishing': np.random.randint(0, 3, n_samples),
        'View': np.random.randint(1, 5, n_samples),
        'Developer': np.random.randint(1, 10, n_samples),
        'ParkingSpaces': np.random.randint(0, 3, n_samples),
        'PropertyType': np.random.randint(1, 4, n_samples),
        'PropertyAge': np.random.randint(0, 30, n_samples),
        'AmenityCount': np.random.randint(0, 10, n_samples),
        'AgeBucket': np.random.randint(1, 4, n_samples),
    }

    # Add Area names based on AreaCode
    data['AreaName'] = [areas[code-1] for code in data['AreaCode']]

    # Add aggregated area statistics
    for area_code in range(1, len(areas) + 1):
        area_mask = data['AreaCode'] == area_code
        area_count = np.sum(area_mask)

        data[f'Count_Area'] = area_count

        # Add more area aggregates
        for col in ['PropertyAge']:
            if col in data:
                area_values = [val for i, val in enumerate(
                    data[col]) if data['AreaCode'][i] == area_code]
                data[f'AvgAge_Area'] = np.mean(
                    area_values) if area_values else 0
                data[f'MedianAge_Area'] = np.median(
                    area_values) if area_values else 0

    # Generate dependent variables
    # Base property values around $400k with variation by area and features
    base_val = 400000
    data['PropertyValuation'] = [
        int(base_val * (1 + 0.1 * data['AreaCode'][i]) * (1 + 0.0003 * data['Size_sqft'][i]) *
            (1 + 0.05 * data['Bedrooms'][i]) * (1 + 0.03 * data['Bathrooms'][i]) *
            (1 - 0.01 * data['PropertyAge'][i]) * (1 + 0.02 * data['View'][i]) *
            (0.9 + 0.2 * np.random.random()))  # Add randomness
        for i in range(n_samples)
    ]

    # Annual rent at roughly 5% of property value with variation
    data['AnnualRent'] = [
        int(data['PropertyValuation'][i] *
            (0.045 + 0.015 * np.random.random()))
        for i in range(n_samples)
    ]

    # Calculate derived metrics
    data['PricePerSqFt'] = [data['PropertyValuation'][i] /
                            data['Size_sqft'][i] for i in range(n_samples)]
    data['RentToValueRatio'] = [data['AnnualRent'][i] /
                                data['PropertyValuation'][i] for i in range(n_samples)]
    data['RentalYield'] = [100 * data['RentToValueRatio'][i]
                           for i in range(n_samples)]
    data['BedBathRatio'] = [data['Bedrooms'][i] /
                            max(1, data['Bathrooms'][i]) for i in range(n_samples)]

    # Add price and rent per room metrics
    data['PricePerBedroom'] = [data['PropertyValuation'][i] /
                               max(1, data['Bedrooms'][i]) for i in range(n_samples)]
    data['PricePerBathroom'] = [data['PropertyValuation'][i] /
                                max(1, data['Bathrooms'][i]) for i in range(n_samples)]
    data['RentPerBedroom'] = [data['AnnualRent'][i] /
                              max(1, data['Bedrooms'][i]) for i in range(n_samples)]
    data['RentPerBathroom'] = [data['AnnualRent'][i] /
                               max(1, data['Bathrooms'][i]) for i in range(n_samples)]

    # Add area statistics
    for area_code in range(1, len(areas) + 1):
        area_mask = [i for i, code in enumerate(
            data['AreaCode']) if code == area_code]
        if area_mask:
            area_vals = [data['PropertyValuation'][i] for i in area_mask]
            area_rents = [data['AnnualRent'][i] for i in area_mask]
            area_price_per_sqft = [data['PricePerSqFt'][i] for i in area_mask]
            area_rent_to_value = [data['RentToValueRatio'][i]
                                  for i in area_mask]

            for i in range(n_samples):
                if data['AreaCode'][i] == area_code:
                    data['AvgVal_Area'] = np.mean(area_vals)
                    data['MedianVal_Area'] = np.median(area_vals)
                    data['AvgRent_Area'] = np.mean(area_rents)
                    data['MedianRent_Area'] = np.median(area_rents)
                    data['AvgPricePerSqFt_Area'] = np.mean(area_price_per_sqft)
                    data['AvgRentToValue_Area'] = np.mean(area_rent_to_value)

    # Create DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(
        f"Created sample dataset with {len(df)} properties in {len(areas)} areas")


def create_investor_context(min_budget: int = 300000,
                            max_budget: int = 800000,
                            risk_profile: str = "moderate",
                            property_types: list = [1, 2]) -> Dict[str, Any]:
    """
    Create a sample investor user context.

    Args:
        min_budget: Minimum budget
        max_budget: Maximum budget
        risk_profile: Risk profile (conservative, moderate, aggressive)
        property_types: List of property type IDs

    Returns:
        Dictionary with investor context
    """
    return {
        "user_type": "investor",
        "investment_horizon": "medium_term",
        "risk_profile": risk_profile,
        "budget_min": min_budget,
        "budget_max": max_budget,
        "target_areas": [],  # No specific area restrictions
        "property_types": property_types,
        "existing_properties": []
    }


def create_property_owner_context(properties: list = None) -> Dict[str, Any]:
    """
    Create a sample property owner user context.

    Args:
        properties: List of owned properties with details

    Returns:
        Dictionary with property owner context
    """
    if properties is None:
        # Sample properties if none provided
        properties = [
            {
                "id": "prop001",
                "area": "Downtown",
                "bedrooms": 2,
                "bathrooms": 1,
                "size_sqft": 950,
                "current_rent": 22000,
                "property_type": 1
            },
            {
                "id": "prop002",
                "area": "Suburb North",
                "bedrooms": 3,
                "bathrooms": 2,
                "size_sqft": 1500,
                "current_rent": 30000,
                "property_type": 2
            }
        ]

    return {
        "user_type": "property_owner",
        "investment_horizon": "long_term",
        "risk_profile": "conservative",
        "budget_min": None,
        "budget_max": None,
        "target_areas": [],
        "property_types": [],
        "existing_properties": properties
    }


def run_investor_demo(pipeline, custom_budget=None):
    """Run a demo for an investor user."""
    print("\n" + "="*80)
    print("REAL ESTATE INVESTMENT RECOMMENDATION DEMO - INVESTOR".center(80))
    print("="*80)

    # Create investor context with optional custom budget
    if custom_budget:
        min_budget, max_budget = custom_budget
        investor_context = create_investor_context(min_budget, max_budget)
        print(
            f"\nRunning with custom budget range: ${min_budget:,} - ${max_budget:,}")
    else:
        investor_context = create_investor_context()
        print(
            f"\nRunning with default budget range: ${investor_context['budget_min']:,} - ${investor_context['budget_max']:,}")

    # Run the pipeline for an investor
    results = pipeline.run(investor_context)

    if results.get("status") == "error":
        print(f"Error: {results.get('error')}")
        return

    # Display top investment recommendations
    print("\nTop Investment Recommendations:")
    print("-" * 60)

    # Top undervalued properties
    print("\n👑 TOP UNDERVALUED PROPERTIES:")
    if "top_properties" in results["recommendations"] and "undervalued" in results["recommendations"]["top_properties"]:
        undervalued = results["recommendations"]["top_properties"]["undervalued"]
        for i, prop in enumerate(undervalued[:5]):
            print(
                f"{i+1}. {prop['AreaName']} - ${prop['PropertyValuation']:,}")
            print(f"   Undervalued by: {prop['ValDiffPct']:.1f}%")
            print(f"   Rental Yield: {prop['RentalYield']:.2f}%")
            print(f"   Investment Score: {prop['InvestmentScore']:.1f}")
            print()
    else:
        print("No undervalued property recommendations available")

    # Top areas to invest in
    print("\n🌆 TOP AREAS FOR INVESTMENT:")
    if "top_areas" in results["recommendations"] and "overall" in results["recommendations"]["top_areas"]:
        top_areas = results["recommendations"]["top_areas"]["overall"]
        for i, area in enumerate(top_areas[:5]):
            print(f"{i+1}. {area['AreaName']}")
            print(f"   Investment Score: {area['InvestmentScore']:.1f}")
            print(f"   Avg Rental Yield: {area['RentalYield']:.2f}%")
            print(
                f"   Value Growth Potential: {area.get('ValDiffPct', 0):.1f}%")
            print()
    else:
        print("No area recommendations available")

    # Print automated decisions
    print("\n⚡ AUTOMATED INVESTMENT DECISIONS:")
    if "decisions" in results["decisions"]:
        decisions = results["decisions"]["decisions"]
        if decisions:
            for i, decision in enumerate(decisions[:5]):
                print(
                    f"{i+1}. {decision['action']} in {decision['area_name']}")
                print(f"   Reason: {decision['reason']}")
                print(f"   Metrics: {', '.join([f'{k}: {v:.1f}%' for k, v in decision['metrics'].items(
                ) if isinstance(v, (int, float))])}")
                print()
        else:
            print("No automated decisions generated")
    else:
        print("No decisions available")

    # Market summary
    print("\n📊 MARKET SUMMARY:")
    if "market_metrics" in results["recommendations"]:
        metrics = results["recommendations"]["market_metrics"]
        print(f"Average Yield: {metrics['avg_yield']:.2f}%")
        print(f"Median Yield: {metrics['median_yield']:.2f}%")
        print(f"Properties Analyzed: {metrics['property_count']}")
        print(f"Areas Analyzed: {metrics['area_count']}")
    else:
        print("No market metrics available")

    return results


def run_property_owner_demo(pipeline, custom_properties=None):
    """Run a demo for a property owner user."""
    print("\n" + "="*80)
    print("RENT OPTIMIZATION RECOMMENDATION DEMO - PROPERTY OWNER".center(80))
    print("="*80)

    # Create property owner context
    owner_context = create_property_owner_context(custom_properties)

    print(
        f"\nAnalyzing {len(owner_context['existing_properties'])} properties for rent optimization")

    # Run the pipeline for a property owner
    results = pipeline.run(owner_context)

    if results.get("status") == "error":
        print(f"Error: {results.get('error')}")
        return

    # Display rent optimization recommendations
    print("\n💰 RENT OPTIMIZATION RECOMMENDATIONS:")
    print("-" * 60)

    if "rent_optimization" in results["recommendations"]:
        recommendations = results["recommendations"]["rent_optimization"]
        if recommendations:
            for i, rec in enumerate(recommendations):
                print(f"{i+1}. Property in {rec['area']}")
                print(f"   Current Rent: ${rec['current_rent']:,}/year")
                print(
                    f"   Recommended Rent: ${rec['recommended_rent']:,.0f}/year")
                print(f"   Difference: {rec['rent_diff_pct']:.1f}%")
                print(f"   Recommendation: {rec['recommendation']}")
                print()
        else:
            print("No rent optimization recommendations available")
    else:
        print("No rent optimization data available")

    # Market insights for the areas
    print("\n📈 MARKET INSIGHTS FOR YOUR AREAS:")
    if "market_insights" in results["recommendations"]:
        insights = results["recommendations"]["market_insights"]
        for area, data in insights.items():
            print(f"\nArea: {area}")
            print(f"  Average Rental Yield: {data['avg_rental_yield']:.2f}%")
            print(f"  Price Trend: {data['price_trend']:.1f}%")
            print(f"  Rent Trend: {data['rent_trend']:.1f}%")
            print(f"  Investment Score: {data['investment_score']:.1f}")
    else:
        print("No market insights available")

    # Print automated decisions related to rent
    print("\n⚡ AUTOMATED RENT DECISIONS:")
    if "decisions" in results["decisions"]:
        decisions = [d for d in results["decisions"].get("decisions", [])
                     if "rent" in d.get("reason", "").lower()]

        if decisions:
            for i, decision in enumerate(decisions):
                print(
                    f"{i+1}. {decision['action']} for property in {decision['area_name']}")
                print(f"   Reason: {decision['reason']}")
                if "metrics" in decision:
                    metrics_str = ", ".join([f"{k}: {v:.1f}%" for k, v in decision["metrics"].items()
                                             if isinstance(v, (int, float))])
                    print(f"   Metrics: {metrics_str}")
                print()
        else:
            print("No automated rent decisions generated")
    else:
        print("No decisions available")

    return results


def main():
    """Main function to run the demo."""
    parser = argparse.ArgumentParser(
        description='Real Estate Investment AI Demo')
    parser.add_argument('--user-type', choices=['investor', 'owner', 'both'],
                        default='both', help='Type of user to run demo for')
    parser.add_argument('--budget-min', type=int, default=None,
                        help='Minimum budget for investor (default: 300,000)')
    parser.add_argument('--budget-max', type=int, default=None,
                        help='Maximum budget for investor (default: 800,000)')
    parser.add_argument('--output-dir', default='demo_output',
                        help='Directory to save output files')
    parser.add_argument('--data-file', default=None,
                        help='Path to data file (CSV) to use for analysis')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup directories and data
    setup_success, setup_message = setup_demo_directories(args.data_file)
    print(setup_message)

    if not setup_success:
        print("Failed to set up directory structure and data files. Exiting.")
        sys.exit(1)

    # Modify config to point to output directory and always train models
    config = DEFAULT_CONFIG.copy()
    config["report_dir"] = args.output_dir

    # Always train new models
    config["train_if_no_models"] = True
    print("Model training is enabled - will train new models.")

    # Initialize pipeline
    print("Initializing Real Estate Investment AI Pipeline...")
    pipeline = InvestmentRecommendationPipeline(config)

    # Initialize the pipeline
    init_success = pipeline.initialize()
    if not init_success:
        print("Failed to initialize pipeline. Exiting.")
        return

    # Custom budget if provided
    custom_budget = None
    if args.budget_min is not None and args.budget_max is not None:
        custom_budget = (args.budget_min, args.budget_max)

    try:
        # Run appropriate demos based on user type
        if args.user_type in ['investor', 'both']:
            investor_results = run_investor_demo(pipeline, custom_budget)

            if investor_results and investor_results.get("status") != "error":
                # Save investor results to file
                investor_output_path = os.path.join(
                    args.output_dir, "investor_recommendations.json")
                with open(investor_output_path, 'w') as f:
                    json.dump(investor_results, f, indent=2, default=str)
                print(
                    f"\nInvestor recommendations saved to {investor_output_path}")

        if args.user_type in ['owner', 'both']:
            owner_results = run_property_owner_demo(pipeline)

            if owner_results and owner_results.get("status") != "error":
                # Save property owner results to file
                owner_output_path = os.path.join(
                    args.output_dir, "property_owner_recommendations.json")
                with open(owner_output_path, 'w') as f:
                    json.dump(owner_results, f, indent=2, default=str)
                print(
                    f"\nProperty owner recommendations saved to {owner_output_path}")

        print("\nDemo completed. Check the output files for detailed results.")

    except Exception as e:
        print(f"Error running demo: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from real_estate_investment_model import RealEstateInvestmentModel

def main():
    """
    Demonstrate the Real Estate Investment Model with sample data
    """
    print("=" * 80)
    print("REAL ESTATE INVESTMENT OPPORTUNITIES MODEL")
    print("=" * 80)
    
    # 1. Create sample data
    print("\n1. Creating sample property data...")
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
    
    # Create DataFrame from sample data
    property_df = pd.DataFrame(sample_properties)
    print(f"Created sample dataset with {len(property_df)} properties")
    
    # 2. Initialize the model and load data
    print("\n2. Initializing the investment model...")
    model = RealEstateInvestmentModel()
    model.load_data(dataframe=property_df)
    
    # 3. Display basic property information
    print("\n3. Viewing property dataset overview:")
    print(model.properties_df[['area_name', 'property_type', 'size_sqft', 'bed', 'bath', 
                              'property_price', 'annual_rental_price']].head())
    
    # 4. Display derived financial metrics
    print("\n4. Derived property metrics:")
    print(model.properties_df[['area_name', 'property_type', 'price_per_sqft', 
                              'rental_yield', 'price_to_rent_ratio']].head())
    
    # 5. Calculate and display market averages
    print("\n5. Calculating market averages...")
    averages = model._calculate_market_averages()
    
    print("\nOverall Market Averages:")
    print(f"  Average Price per Sqft: ${averages['overall']['avg_price_per_sqft']:.2f}")
    print(f"  Average Rental Yield: {averages['overall']['avg_rental_yield']:.2f}%")
    print(f"  Average Price-to-Rent Ratio: {averages['overall']['avg_price_to_rent_ratio']:.2f}")
    
    print("\nProperty Type Averages:")
    for prop_type, metrics in averages['by_property_type'].items():
        print(f"  {prop_type}:")
        print(f"    Average Price per Sqft: ${metrics['avg_price_per_sqft']:.2f}")
        print(f"    Average Rental Yield: {metrics['avg_rental_yield']:.2f}%")
    
    print("\nArea Averages:")
    for area, metrics in averages['by_area'].items():
        print(f"  {area}:")
        print(f"    Average Price per Sqft: ${metrics['avg_price_per_sqft']:.2f}")
        print(f"    Average Rental Yield: {metrics['avg_rental_yield']:.2f}%")
    
    # 6. Analyze a specific property
    print("\n6. Analyzing a specific property...")
    property_idx = 0  # First property in the dataset
    property_name = f"{model.properties_df.iloc[property_idx]['area_name']} {model.properties_df.iloc[property_idx]['property_type']}"
    
    print(f"\nProperty: {property_name}")
    print(f"  Size: {model.properties_df.iloc[property_idx]['size_sqft']} sqft")
    print(f"  Bedrooms: {model.properties_df.iloc[property_idx]['bed']}")
    print(f"  Bathrooms: {model.properties_df.iloc[property_idx]['bath']}")
    print(f"  Price: ${model.properties_df.iloc[property_idx]['property_price']:,}")
    print(f"  Annual Rental: ${model.properties_df.iloc[property_idx]['annual_rental_price']:,}")
    
    # 7. Valuation analysis
    print("\n7. Property valuation analysis:")
    valuation = model.predict_fair_market_value(property_idx)
    
    print(f"  Current Price: ${valuation['current_value']:,.2f}")
    print(f"  Fair Market Value: ${valuation['fair_market_value']:,.2f}")
    print(f"  {'Undervalued' if valuation['undervalued'] else 'Overvalued'} " +
          f"by: ${abs(valuation['value_difference']):,.2f} " +
          f"({abs(valuation['value_difference_percent']):.2f}%)")
    
    # 8. Rental optimization
    print("\n8. Rental price optimization:")
    rental = model.predict_optimal_rental(property_idx)
    
    print(f"  Current Annual Rent: ${rental['current_annual_rental']:,.2f}")
    print(f"  Optimal Annual Rent: ${rental['optimal_annual_rental']:,.2f}")
    print(f"  {'Rental Upside' if rental['rental_upside'] else 'Rental Downside'} " +
          f"of: ${abs(rental['rental_difference']):,.2f} " +
          f"({abs(rental['rental_difference_percent']):.2f}%)")
    
    # 9. Calculate investment score
    print("\n9. Investment score calculation:")
    score = model.calculate_investment_score(property_idx)
    
    print(f"  Overall Investment Score: {score['overall_score']:.2f}/100")
    print(f"  Score Breakdown:")
    print(f"    Valuation Score: {score['valuation_score']:.2f}/40")
    print(f"    Yield Score: {score['yield_score']:.2f}/30")
    print(f"    Rental Upside Score: {score['rental_upside_score']:.2f}/20")
    print(f"    Location Score: {score['location_score']:.2f}/10")
    
    # 10. Find top investment opportunities
    print("\n10. Finding top investment opportunities...")
    top_opps = model.find_top_opportunities(count=3)
    
    print("\nTop 3 Investment Opportunities:")
    for i, opp in enumerate(top_opps, 1):
        prop = opp['property']
        score = opp['score']
        print(f"  #{i}: {prop['area_name']} {prop['property_type']} - Score: {score['overall_score']:.2f}/100")
        print(f"    Size: {prop['size_sqft']} sqft, Beds: {prop['bed']}, Baths: {prop['bath']}")
        print(f"    Price: ${prop['property_price']:,}, Annual Rent: ${prop['annual_rental_price']:,}")
        print(f"    Rental Yield: {prop['rental_yield']:.2f}%")
        if score['details']['valuation']['undervalued']:
            print(f"    Undervalued by: {score['details']['valuation']['value_difference_percent']:.2f}%")
        print()
    
    # 11. Find highest yield properties
    print("\n11. Finding highest yield properties...")
    high_yield = model.find_highest_yield_properties(count=3)
    
    print("\nTop 3 Highest Yield Properties:")
    for i, (_, prop) in enumerate(high_yield.iterrows(), 1):
        print(f"  #{i}: {prop['area_name']} {prop['property_type']} - Yield: {prop['rental_yield']:.2f}%")
        print(f"    Size: {prop['size_sqft']} sqft, Price: ${prop['property_price']:,}")
        print(f"    Annual Rental: ${prop['annual_rental_price']:,}")
        print()
    
    # 12. Find undervalued properties
    print("\n12. Finding undervalued properties...")
    undervalued = model.find_undervalued_properties(count=3)
    
    print("\nTop 3 Most Undervalued Properties:")
    for i, prop_data in enumerate(undervalued, 1):
        prop = prop_data['property']
        val = prop_data['valuation']
        print(f"  #{i}: {prop['area_name']} {prop['property_type']}")
        print(f"    Current Price: ${val['current_value']:,}")
        print(f"    Fair Market Value: ${val['fair_market_value']:,.2f}")
        print(f"    Undervalued by: {val['value_difference_percent']:.2f}%")
        print(f"    Size: {prop['size_sqft']} sqft, Beds: {prop['bed']}, Baths: {prop['bath']}")
        print()
    
    # 13. Visualize market overview
    print("\n13. Generating market visualizations...")
    try:
        model.visualize_market_overview()
        print("Market overview visualization complete.")
    except Exception as e:
        print(f"Error generating visualizations: {e}")
    
    # 14. Visualize investment opportunities
    print("\n14. Generating investment opportunity visualizations...")
    try:
        model.visualize_investment_opportunities(top_opps)
        print("Investment opportunities visualization complete.")
    except Exception as e:
        print(f"Error generating visualizations: {e}")
    
    # 15. Export investment report
    print("\n15. Exporting investment report...")
    try:
        output_path = 'investment_opportunities.csv'
        model.export_investment_report(top_opps, output_path)
        print(f"Investment report exported to {output_path}")
    except Exception as e:
        print(f"Error exporting report: {e}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
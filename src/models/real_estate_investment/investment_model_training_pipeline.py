#!/usr/bin/env python
"""
investment_model_training_pipeline.py

Orchestrates the training of the RealEstateInvestmentModel with enhanced:
  - Section dividers for clarity
  - Extended model evaluation
  - Visualizations for model assessment
  - Investment opportunity identification
"""
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_percentage_error
from real_estate_investment_model import RealEstateInvestmentModel

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
DATA_PATH        = "data/training_data/investment/processed_investment_data.csv"
MODEL_DIR        = "src/models/pretrained_models"
REPORT_DIR       = "reports/investment"
PROPERTY_SCORES  = os.path.join(REPORT_DIR, "property_investment_scores.csv")
AREA_SCORES      = os.path.join(REPORT_DIR, "area_investment_scores.csv")
EVAL_DIR         = os.path.join(REPORT_DIR, "model_evaluation")
TOP_OPPORTUNITIES = os.path.join(REPORT_DIR, "top_investment_opportunities.csv")

# -----------------------------------------------------------------------------
# UTILITIES
# -----------------------------------------------------------------------------
def print_section_header(title):
    """Print a section header to make the output more readable."""
    separator = "=" * 80
    print("\n" + separator)
    print(f" {title} ".center(80, "="))
    print(separator + "\n")

def evaluate_predictions(y_true, y_pred, title):
    """Calculate and print normalized evaluation metrics."""
    # Calculate standard metrics
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    
    # Calculate additional metrics
    median_abs_error = np.median(np.abs(y_true - y_pred))
    max_error = np.max(np.abs(y_true - y_pred))
    
    # Calculate metrics relative to mean value (normalized)
    mean_value = np.mean(y_true)
    norm_mse = mse / (mean_value ** 2)
    norm_rmse = rmse / mean_value
    norm_mae = mae / mean_value
    norm_median_error = median_abs_error / mean_value
    norm_max_error = max_error / mean_value
    
    # Calculate percentage of predictions within thresholds
    within_10pct = np.mean(np.abs((y_true - y_pred) / y_true) <= 0.1) * 100
    within_20pct = np.mean(np.abs((y_true - y_pred) / y_true) <= 0.2) * 100
    
    print(f"\n{title} Prediction Evaluation:")
    print(f"  Mean Squared Error (MSE):     {mse:.2f} (Normalized: {norm_mse:.4f})")
    print(f"  Root Mean Squared Error:      {rmse:.2f} (Normalized: {norm_rmse:.4f})")
    print(f"  Mean Absolute Error (MAE):    {mae:.2f} (Normalized: {norm_mae:.4f})")
    print(f"  Median Absolute Error:        {median_abs_error:.2f} (Normalized: {norm_median_error:.4f})")
    print(f"  Mean Absolute % Error (MAPE): {mape:.2f}%")
    print(f"  R² Score:                     {r2:.4f}")
    print(f"  Maximum Error:                {max_error:.2f} (Normalized: {norm_max_error:.4f})")
    print(f"  Predictions within 10%:       {within_10pct:.1f}%")
    print(f"  Predictions within 20%:       {within_20pct:.1f}%")
    
    return {
        'mse': mse,
        'norm_mse': norm_mse,
        'rmse': rmse,
        'norm_rmse': norm_rmse,
        'mae': mae,
        'norm_mae': norm_mae,
        'mape': mape,
        'r2': r2,
        'median_abs_error': median_abs_error,
        'norm_median_error': norm_median_error,
        'max_error': max_error,
        'norm_max_error': norm_max_error,
        'within_10pct': within_10pct,
        'within_20pct': within_20pct
    }
def plot_prediction_performance(df, target_col, pred_col, title, output_path):
    """Create visualization of model prediction performance."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Predicted vs Actual
    ax = axes[0, 0]
    ax.scatter(df[target_col], df[pred_col], alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(df[target_col].min(), df[pred_col].min())
    max_val = max(df[target_col].max(), df[pred_col].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    ax.set_xlabel('Actual Value')
    ax.set_ylabel('Predicted Value')
    ax.set_title('Predicted vs Actual Values')
    
    # Plot 2: Residuals
    ax = axes[0, 1]
    residuals = df[target_col] - df[pred_col]
    ax.scatter(df[pred_col], residuals, alpha=0.5)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_xlabel('Predicted Value')
    ax.set_ylabel('Residual (Actual - Predicted)')
    ax.set_title('Residual Plot')
    
    # Plot 3: Distribution of errors
    ax = axes[1, 0]
    pct_errors = (df[target_col] - df[pred_col]) / df[target_col] * 100
    ax.hist(pct_errors, bins=50)
    ax.axvline(x=0, color='r', linestyle='--')
    ax.set_xlabel('Percentage Error')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Percentage Errors')
    
    # Plot 4: QQ plot of residuals
    ax = axes[1, 1]
    residuals_sorted = np.sort(residuals)
    n = len(residuals_sorted)
    quantiles = np.arange(1, n + 1) / (n + 1)
    theoretical_quantiles = np.quantile(np.random.normal(0, 1, 1000), quantiles)
    
    ax.scatter(theoretical_quantiles, np.sort(residuals / np.std(residuals)))
    ax.plot([min(theoretical_quantiles), max(theoretical_quantiles)], 
           [min(theoretical_quantiles), max(theoretical_quantiles)], 'r--')
    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Sample Quantiles (Normalized)')
    ax.set_title('QQ Plot of Residuals')
    
    # Add overall title
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the figure
    plt.savefig(output_path)
    plt.close()

def identify_top_opportunities(df, n=20):
    """Identify and return top investment opportunities."""
    # Sort by investment score
    top_props = df.sort_values('InvestmentScore', ascending=False).head(n).copy()
    
    # Add opportunity classification
    conditions = [
        (top_props['ValScore'] > top_props['YieldScore']) & (top_props['ValScore'] > top_props['RentScore']),
        (top_props['YieldScore'] > top_props['ValScore']) & (top_props['YieldScore'] > top_props['RentScore']),
        (top_props['RentScore'] > top_props['ValScore']) & (top_props['RentScore'] > top_props['YieldScore'])
    ]
    choices = ['Undervalued', 'High Yield', 'Rental Upside']
    top_props['Opportunity_Type'] = np.select(conditions, choices, default='Balanced')
    
    return top_props

def visualize_investment_opportunities(df, area_scores_df, output_dir):
    """Create visualization of top investment opportunities."""
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Top Areas by Investment Score
    plt.figure(figsize=(12, 8))
    top_areas = area_scores_df.sort_values('InvestmentScore', ascending=False).head(15)
    sns.barplot(x='InvestmentScore', y='AreaName', data=top_areas)
    plt.title('Top 15 Areas by Investment Score', fontsize=14)
    plt.xlabel('Investment Score')
    plt.ylabel('Area')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_areas_by_score.png'))
    plt.close()
    
    # 2. Score Components for Top 10 Areas
    plt.figure(figsize=(14, 10))
    top_10_areas = area_scores_df.sort_values('InvestmentScore', ascending=False).head(10)
    
    # Reshape data for stacked bar chart
    components = pd.melt(
        top_10_areas, 
        id_vars=['AreaName'], 
        value_vars=['ValScore', 'YieldScore', 'RentScore', 'LocScore'],
        var_name='Component', 
        value_name='Score'
    )
    
    sns.barplot(x='Score', y='AreaName', hue='Component', data=components)
    plt.title('Investment Score Components for Top 10 Areas', fontsize=14)
    plt.xlabel('Score Contribution')
    plt.ylabel('Area')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'score_components_by_area.png'))
    plt.close()
    
    # 3. Rental Yield vs Price per Sqft
    plt.figure(figsize=(12, 8))
    scatter = sns.scatterplot(
        x='PricePerSqFt', 
        y='RentalYield', 
        hue='InvestmentScore',
        size='InvestmentScore',
        sizes=(20, 200),
        palette='viridis',
        data=df
    )
    plt.title('Rental Yield vs Price per Square Foot', fontsize=14)
    plt.xlabel('Price per Square Foot')
    plt.ylabel('Rental Yield (%)')
    
    # Add area labels for top opportunities
    top_opportunities = df.sort_values('InvestmentScore', ascending=False).head(10)
    for _, row in top_opportunities.iterrows():
        plt.text(row['PricePerSqFt'], row['RentalYield'], row['AreaName'], 
                fontsize=9, ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'yield_vs_price_per_sqft.png'))
    plt.close()
    
    # 4. Opportunity Type Distribution
    top_props = identify_top_opportunities(df, n=100)
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Opportunity_Type', data=top_props)
    plt.title('Distribution of Investment Opportunity Types (Top 100)', fontsize=14)
    plt.xlabel('Opportunity Type')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'opportunity_type_distribution.png'))
    plt.close()

# -----------------------------------------------------------------------------
# MAIN PIPELINE FUNCTION
# -----------------------------------------------------------------------------
def main():
    """Main training and evaluation pipeline."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Create output directories
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)
    
    # -----------------------------------------------------------------------------
    # DATA LOADING
    # -----------------------------------------------------------------------------
    print_section_header("DATA LOADING")
    
    # Ensure data file exists
    if not os.path.exists(DATA_PATH):
        logger.error(f"Required data file not found: {DATA_PATH}")
        return

    # Load data
    logger.info(f"Loading investment data from {DATA_PATH}...")
    investment_data = pd.read_csv(DATA_PATH)
    logger.info(f"Loaded {len(investment_data)} records with {investment_data.shape[1]} features")
    
    # Display data summary
    print("\nData Summary:")
    print(f"  Total properties: {len(investment_data)}")
    print(f"  Unique areas: {investment_data['AreaName'].nunique()}")
    if 'PropertyType' in investment_data.columns:
        print(f"  Property types: {investment_data['PropertyType'].unique()}")
    
    # -----------------------------------------------------------------------------
    # MODEL TRAINING
    # -----------------------------------------------------------------------------
    print_section_header("MODEL TRAINING")
    
    # Initialize and train model
    model = RealEstateInvestmentModel(data_path=DATA_PATH)
    logger.info("Starting training of valuation and rental models...")
    models, metrics = model.train()

    # Log training results
    for target, res in metrics.items():
        logger.info(f"{target} training metrics:")
        logger.info(f"  MSE:  {res['mse']:.2f}")
        logger.info(f"  MAE:  {res['mae']:.2f}")
        logger.info(f"  MAPE: {res['mape']:.2f}%")
        logger.info(f"  R²:   {res['r2']:.3f}")
        logger.info(f"  Best params: {res['best_params']}")

    # Save trained models
    model.save_models(MODEL_DIR)
    logger.info(f"Saved models to {MODEL_DIR}")
    
    # -----------------------------------------------------------------------------
    # MODEL EVALUATION
    # -----------------------------------------------------------------------------
    print_section_header("MODEL EVALUATION")
    
    # Generate predictions for the entire dataset
    exclude_cols = ['PropertyValuation', 'AnnualRent', 'AreaName']
    features = investment_data.drop(columns=exclude_cols, errors='ignore')
    
    predictions = {}
    for target, trained_model in models.items():
        predictions[target] = trained_model.predict(features)
    
    # Create dataframe with true values and predictions
    eval_df = pd.DataFrame({
        'PropertyValuation': investment_data['PropertyValuation'],
        'AnnualRent': investment_data['AnnualRent'],
        'PredictedValuation': predictions['PropertyValuation'],
        'PredictedRent': predictions['AnnualRent'],
        'AreaName': investment_data['AreaName']
    })
    
    # Calculate evaluation metrics
    valuation_metrics = evaluate_predictions(
        eval_df['PropertyValuation'], 
        eval_df['PredictedValuation'],
        'Property Valuation'
    )
    
    rent_metrics = evaluate_predictions(
        eval_df['AnnualRent'],
        eval_df['PredictedRent'],
        'Annual Rent'
    )
    
    # Create visualizations
    logger.info("Generating prediction performance visualizations...")
    plot_prediction_performance(
        eval_df, 
        'PropertyValuation', 
        'PredictedValuation',
        'Property Valuation Model Performance',
        os.path.join(EVAL_DIR, 'valuation_model_performance.png')
    )
    
    plot_prediction_performance(
        eval_df, 
        'AnnualRent', 
        'PredictedRent',
        'Annual Rent Model Performance',
        os.path.join(EVAL_DIR, 'rent_model_performance.png')
    )
    
    # Save evaluation metrics
    eval_metrics = {
        'PropertyValuation': valuation_metrics,
        'AnnualRent': rent_metrics
    }
    pd.DataFrame(eval_metrics).to_csv(os.path.join(EVAL_DIR, 'model_metrics.csv'))
    logger.info(f"Evaluation metrics saved to {os.path.join(EVAL_DIR, 'model_metrics.csv')}")
    
    # -----------------------------------------------------------------------------
    # INVESTMENT SCORING
    # -----------------------------------------------------------------------------
    print_section_header("INVESTMENT SCORING")
    
    # Compute and export property-level investment scores
    logger.info("Computing property-level investment scores...")
    property_scores = model.score_investment()
    property_scores.to_csv(PROPERTY_SCORES, index=False)
    logger.info(f"Property investment scores exported to {PROPERTY_SCORES}")
    
    # Print sample scores
    print("\nSample Property Investment Scores:")
    sample_scores = property_scores.sort_values('InvestmentScore', ascending=False).head(5)
    for _, row in sample_scores.iterrows():
        print(f"  Area: {row['AreaName']}")
        print(f"    Investment Score: {row['InvestmentScore']:.2f}")
        print(f"    Components: Valuation={row['ValScore']:.1f}, Yield={row['YieldScore']:.1f}, " +
              f"Rental={row['RentScore']:.1f}, Location={row['LocScore']:.1f}")
        print(f"    Value Diff: {row['ValDiffPct']:.1f}%, Rental Yield: {row['RentalYield']:.2f}%")
        print()
    
    # Compute and export area-level investment scores
    logger.info("Computing area-level investment scores...")
    area_scores = model.get_area_investment_scores()
    area_scores.to_csv(AREA_SCORES, index=False)
    logger.info(f"Area investment scores exported to {AREA_SCORES}")
    
    # Print top area scores
    print("\nTop 5 Areas by Investment Score:")
    top_areas = area_scores.head(5)
    for _, row in top_areas.iterrows():
        print(f"  {row['AreaName']}: {row['InvestmentScore']:.2f}")
        print(f"    Components: Valuation={row['ValScore']:.1f}, Yield={row['YieldScore']:.1f}, " +
              f"Rental={row['RentScore']:.1f}, Location={row['LocScore']:.1f}")
        print(f"    Value Diff: {row['ValDiffPct']:.1f}%, Rental Yield: {row['RentalYield']:.2f}%")
        print()
    
    # -----------------------------------------------------------------------------
    # INVESTMENT OPPORTUNITIES
    # -----------------------------------------------------------------------------
    print_section_header("INVESTMENT OPPORTUNITIES")
    
    # Identify top investment opportunities
    top_opportunities = identify_top_opportunities(property_scores)
    top_opportunities.to_csv(TOP_OPPORTUNITIES, index=False)
    logger.info(f"Top investment opportunities exported to {TOP_OPPORTUNITIES}")
    
    # Print top opportunities
    print("\nTop Investment Opportunities:")
    for i, (_, row) in enumerate(top_opportunities.head(5).iterrows()):
        print(f"  {i+1}. {row['AreaName']} - {row['Opportunity_Type']}")
        print(f"     Investment Score: {row['InvestmentScore']:.2f}")
        print(f"     Value Diff: {row['ValDiffPct']:.1f}%, Rental Yield: {row['RentalYield']:.2f}%")
        if row['Opportunity_Type'] == 'Undervalued':
            print(f"     Property appears undervalued by {row['ValDiffPct']:.1f}%")
        elif row['Opportunity_Type'] == 'High Yield':
            print(f"     Offers strong rental yield of {row['RentalYield']:.2f}%")
        elif row['Opportunity_Type'] == 'Rental Upside':
            print(f"     Rental upside potential of {row['RentDiffPct']:.1f}%")
        print()
    
    logger.info("Investment model training and evaluation pipeline completed successfully!")

# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    main()
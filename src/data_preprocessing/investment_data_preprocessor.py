#!/usr/bin/env python
"""
investment_data_preprocessor.py

Loads raw investment data, performs feature engineering, EDA, area-level stats,
and balances under-/over-represented strata using CTGANSynthesizer (from SDV v1.0+).
Exports a single combined file with processed features and area-level statistics.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

from sdv.single_table import CTGANSynthesizer
from sdv.metadata import Metadata
from sdv.sampling import Condition

# Paths
RAW_PATH       = "data/training_data/investment/investment_data.csv"
PROCESSED_PATH = "data/training_data/investment/processed_investment_data.csv"
OUTPUT_DIR     = "data/training_data/investment"

# -----------------------------------------------------------------------------

def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    """
    Load raw CSV into a DataFrame.
    """
    return pd.read_csv(path)

# -----------------------------------------------------------------------------

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering:
      - PropertyAge
      - AmenityCount
      - PricePerSqFt, RentToValueRatio, BedBathRatio
      - RentalYield
      - PricePerBedroom, PricePerBathroom
      - RentPerBedroom, RentPerBathroom
      - PropertyAge bucket (AgeBucket)
      - Label-encode categoricals, preserving AreaName + AreaCode
      - Drop unused raw columns
    """
    df = df.copy()

    df['Area'] = df['Area'].str.strip()
    
    df['Area'] = df['Area'].str.title()
    
    print(f"Areas before normalization: {df['Area'].nunique()} unique values")
    print(df['Area'].value_counts())

    # 1) Property age
    df['PropertyAge']       = pd.Timestamp.now().year - df['YearBuilt']
    # 2) Amenity count
    df['AmenityCount']      = df['Amenities'].fillna('').str.count(',').add(1)
    # 3) Ratios for price/rent
    df['PricePerSqFt']      = df['PropertyValuation'] / df['Size_sqft']
    df['RentToValueRatio']  = df['AnnualRent']       / df['PropertyValuation']
    df['BedBathRatio']      = df['Bedrooms']         / df['Bathrooms']
    # 4) Rental yield
    df['RentalYield']       = df['AnnualRent'] / df['PropertyValuation'] * 100
    # 5) Per-room metrics
    df['PricePerBedroom']   = df['PropertyValuation'] / df['Bedrooms']
    df['PricePerBathroom']  = df['PropertyValuation'] / df['Bathrooms']
    df['RentPerBedroom']    = df['AnnualRent']       / df['Bedrooms']
    df['RentPerBathroom']   = df['AnnualRent']       / df['Bathrooms']
    # 6) Age buckets
    df['AgeBucket']         = pd.cut(
        df['PropertyAge'], bins=[0,10,20,30,100],
        labels=['<10','10-20','20-30','30+']
    )
    # 7) Preserve the human-readable area, then label-encode
    df['AreaName'] = df['Area']  # keep original area name
    for col in ['Area','Furnishing','View','PropertyType','Developer','AgeBucket']:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    # rename encoded Area → AreaCode
    df = df.rename(columns={'Area': 'AreaCode'})
    # 8) Drop raw temporal & amenities columns
    df = df.drop(columns=['YearBuilt','Amenities'], errors='ignore')
    return df

# -----------------------------------------------------------------------------

def balance_with_ctgan(
    df: pd.DataFrame,
    target_col: str = 'Area',
    target_n: int = 150,
    epochs: int = 500,
    batch_size: int = 500
) -> pd.DataFrame:
    """
    Upsample under-represented strata in `df[target_col]` so that
    each unique value has at least `target_n` rows, using SDV's CTGANSynthesizer.
    """
    # 1) build metadata and synthesizer
    metadata = Metadata.detect_from_dataframe(data=df)
    synthesizer = CTGANSynthesizer(metadata, epochs=epochs, batch_size=batch_size)

    # 2) fit
    synthesizer.fit(df)

    # 3) figure out how many more rows each stratum needs
    counts = df[target_col].value_counts()
    synth_parts = []

    for val, cnt in counts.items():
        needed = max(target_n - cnt, 0)
        if needed > 0:
            cond = Condition(
                num_rows=needed,
                column_values={target_col: val}
            )
            synth = synthesizer.sample_from_conditions(conditions=[cond])
            synth_parts.append(synth)

    # 4) concatenate original + synthetic
    if synth_parts:
        synthetic_df = pd.concat(synth_parts, ignore_index=True)
        return pd.concat([df, synthetic_df], ignore_index=True)
    else:
        return df.copy()

# -----------------------------------------------------------------------------

def eda(df: pd.DataFrame, output_dir: str = 'eda_plots') -> None:
    """
    Perform exploratory data analysis with visualizations using processed features.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(df.info())
    print(df.describe())
    print("Missing values:\n", df.isnull().sum())

    # Histograms
    for col in ['AnnualRent','PropertyValuation','Size_sqft','PropertyAge']:
        if col not in df.columns:
            continue
        plt.figure(figsize=(8,4))
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f"Distribution of {col}")
        plt.savefig(os.path.join(output_dir, f"hist_{col}.png"))
        plt.close()

    # Scatter plots
    if 'Size_sqft' in df.columns and 'AnnualRent' in df.columns:
        plt.figure(figsize=(6,6))
        sns.scatterplot(x='Size_sqft', y='AnnualRent', data=df)
        plt.title('Size vs AnnualRent')
        plt.savefig(os.path.join(output_dir, 'scatter_size_rent.png'))
        plt.close()
    if 'PropertyAge' in df.columns and 'AnnualRent' in df.columns:
        plt.figure(figsize=(6,6))
        sns.scatterplot(x='PropertyAge', y='AnnualRent', data=df)
        plt.title('PropertyAge vs AnnualRent')
        plt.savefig(os.path.join(output_dir, 'scatter_age_rent.png'))
        plt.close()

    # Correlation heatmap
    num_cols = df.select_dtypes(include=np.number).columns
    corr = df[num_cols].corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'corr_heatmap.png'))
    plt.close()
    print(f"EDA plots saved to {output_dir}")

# -----------------------------------------------------------------------------

def calculate_area_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute area-level investment metrics on processed data.
    Returns a DataFrame with area statistics.
    """
    stats = (
        df
        .groupby(['AreaCode','AreaName'])
        .agg(
            Count=('AreaCode','size'),
            AvgAge=('PropertyAge','mean'),
            MedianAge=('PropertyAge','median'),
            AvgVal=('PropertyValuation','mean'),
            MedianVal=('PropertyValuation','median'),
            AvgRent=('AnnualRent','mean'),
            MedianRent=('AnnualRent','median'),
            AvgPricePerSqFt=('PricePerSqFt','mean'),
            AvgRentToValue=('RentToValueRatio','mean')
        )
        .reset_index()
        .sort_values('AvgRentToValue', ascending=False)
    )
    
    return stats

# -----------------------------------------------------------------------------

def visualize_area_stats(stats: pd.DataFrame, output_dir: str = 'eda_plots') -> None:
    """
    Visualize area-level investment metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for metric, fname, title in [
        ('AvgRentToValue','bar_rent_to_value_area.png','Avg Rent-to-Value Ratio by Area'),
        ('AvgPricePerSqFt','bar_price_per_sqft_area.png','Avg PricePerSqFt by Area'),
        ('AvgAge','bar_age_area.png','Avg Property Age by Area')
    ]:
        plt.figure(figsize=(8,6))
        sns.barplot(x=metric, y='AreaName', data=stats)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir,fname))
        plt.close()

    print(f"Area-level visualization saved to {output_dir}")

# -----------------------------------------------------------------------------

def combine_property_and_area_data(properties_df: pd.DataFrame, area_stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine property-level features with area-level statistics.
    """
    # Merge area statistics back to property data
    # We'll add an '_Area' suffix to area-level metrics to distinguish them
    area_stats_df_renamed = area_stats_df.copy()
    
    # Rename columns to add suffix except AreaCode and AreaName
    for col in area_stats_df.columns:
        if col not in ['AreaCode', 'AreaName']:
            area_stats_df_renamed = area_stats_df_renamed.rename(columns={col: f"{col}_Area"})
    
    # Merge on AreaCode
    combined_df = properties_df.merge(area_stats_df_renamed, on=['AreaCode', 'AreaName'], how='left')
    
    return combined_df

# -----------------------------------------------------------------------------

def save_processed_data(df: pd.DataFrame, path: str = PROCESSED_PATH) -> None:
    """
    Save processed DataFrame to CSV.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Combined processed data saved to {path}")

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load raw data
    raw = load_raw()
    print(f"Loaded raw data: {len(raw)} rows")

    # Balance strata via CTGAN
    synth = balance_with_ctgan(raw)
    if len(synth) > len(raw):
        raw = synth
        print(f"Added synthetic rows to balance strata; new total = {len(raw)}")

    print(raw['Area'].value_counts())
    print("Unique views:", raw['View'].unique())

    # Preprocess
    processed_df = preprocess(raw)
    
    # Calculate area statistics
    area_stats_df = calculate_area_stats(processed_df)
    
    # Save area stats separately for reference
    area_stats_path = os.path.join(OUTPUT_DIR, 'area_investment_stats.csv')
    area_stats_df.to_csv(area_stats_path, index=False)
    print(f"Area statistics saved to {area_stats_path}")
    
    # Visualize area statistics
    visualize_area_stats(area_stats_df, output_dir=os.path.join(OUTPUT_DIR, 'eda_plots'))
    
    # Combine property and area data
    combined_df = combine_property_and_area_data(processed_df, area_stats_df)
    
    # Run EDA
    eda(combined_df, output_dir=os.path.join(OUTPUT_DIR, 'eda_plots'))
    
    # Save combined data
    save_processed_data(combined_df, path=PROCESSED_PATH)
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, html, dcc, Input, Output, dash_table
import dash_bootstrap_components as dbc

# Function to generate sample investment property data (for testing)
def generate_sample_investment_data(n_properties=100):
    # Areas
    areas = ['Downtown', 'Marina', 'Suburbs', 'Business Bay', 'Palm Jumeirah', 
            'JBR', 'Silicon Oasis', 'Sports City', 'Discovery Gardens', 'JLT']
    
    # Property types
    property_types = ['Apartment', 'Villa', 'Townhouse', 'Office', 'Retail']
    
    # Generate property data
    data = {
        'area_name': np.random.choice(areas, n_properties),
        'property_type': np.random.choice(property_types, n_properties, 
                                         p=[0.5, 0.2, 0.15, 0.1, 0.05]),  # More apartments
        'size_sqft': np.random.normal(1500, 500, n_properties).astype(int),
        'bed': np.random.choice([1, 2, 3, 4, 5], n_properties, p=[0.2, 0.35, 0.25, 0.15, 0.05]),
        'bath': np.random.choice([1, 2, 3, 4], n_properties, p=[0.3, 0.4, 0.2, 0.1])
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Deal with negative sizes (from normal distribution)
    df['size_sqft'] = df['size_sqft'].apply(lambda x: max(x, 500))
    
    # Base property price calculations with adjustments for area and property type
    df['base_price'] = np.random.normal(1000000, 300000, n_properties)
    
    # Premium areas have higher prices
    area_multipliers = {
        'Palm Jumeirah': 3.0,
        'Marina': 2.0,
        'Downtown': 2.2,
        'Business Bay': 1.8,
        'JBR': 1.9,
        'JLT': 1.5,
        'Sports City': 1.1,
        'Silicon Oasis': 0.9,
        'Discovery Gardens': 0.95,
        'Suburbs': 0.8
    }
    
    # Property type multipliers
    type_multipliers = {
        'Villa': 2.5,
        'Townhouse': 1.8,
        'Apartment': 1.0,
        'Office': 1.5,
        'Retail': 1.7
    }
    
    # Apply size, area, and type adjustments to base price
    for i, row in df.iterrows():
        area_mult = area_multipliers.get(row['area_name'], 1.0)
        type_mult = type_multipliers.get(row['property_type'], 1.0)
        size_factor = row['size_sqft'] / 1000  # per 1000 sqft
        
        # Add slight randomization (±15%)
        random_factor = np.random.uniform(0.85, 1.15)
        
        df.at[i, 'property_price'] = max(500000, row['base_price'] * area_mult * type_mult * size_factor * random_factor)
    
    # Generate rental yield - generally 4-7%, but varies by property type and area
    yield_base = {
        'Apartment': 0.06,  # 6%
        'Villa': 0.045,     # 4.5%
        'Townhouse': 0.05,  # 5%
        'Office': 0.07,     # 7%
        'Retail': 0.075     # 7.5%
    }
    
    # Area yield adjustments (premium areas often have lower yields)
    area_yield_adj = {
        'Palm Jumeirah': -0.01,   # -1% lower yield
        'Marina': -0.005,
        'Downtown': -0.005,
        'Business Bay': 0,
        'JBR': -0.005,
        'JLT': 0,
        'Sports City': 0.01,     # +1% higher yield
        'Silicon Oasis': 0.015,
        'Discovery Gardens': 0.01,
        'Suburbs': 0.02
    }
    
    # Calculate rental yield with randomization
    for i, row in df.iterrows():
        base_yield = yield_base.get(row['property_type'], 0.055)
        area_adj = area_yield_adj.get(row['area_name'], 0)
        # Add randomization (±1%)
        random_adj = np.random.uniform(-0.01, 0.01)
        
        rental_yield = base_yield + area_adj + random_adj
        # Keep yields within reasonable bounds
        rental_yield = max(0.03, min(rental_yield, 0.1))
        
        df.at[i, 'rental_yield'] = rental_yield
    
    # Calculate annual rental based on yield
    df['annual_rental_price'] = df['property_price'] * df['rental_yield']
    
    # Calculate price per square foot
    df['price_per_sqft'] = df['property_price'] / df['size_sqft']
    
    # Calculate price to rent ratio
    df['price_to_rent_ratio'] = df['property_price'] / df['annual_rental_price']
    
    # Calculate market valuation (fair market value)
    # Some properties are over/undervalued
    df['valuation_factor'] = np.random.normal(1.0, 0.12, n_properties)  # ±12% from true value
    df['fair_market_value'] = df['property_price'] * df['valuation_factor']
    
    # Calculate value difference and percentage
    df['value_difference'] = df['fair_market_value'] - df['property_price']
    df['value_difference_percent'] = (df['value_difference'] / df['property_price']) * 100
    
    # Generate bedroom to bathroom ratio
    df['bed_bath_ratio'] = df['bed'] / df['bath']
    
    # Generate a sample amenity score (0-0.2 representing 0-20% premium)
    df['amenity_score'] = np.random.uniform(0, 0.2, n_properties)
    
    # Generate investment score (0-100)
    # Based on rental yield, valuation, location, etc.
    
    # 1. Valuation component (40% of total score)
    df['valuation_score'] = df.apply(
        lambda row: min(40, max(0, (row['value_difference_percent'] / 30) * 40)) if row['value_difference'] > 0 else 0, 
        axis=1
    )
    
    # 2. Rental yield component (30% of total score)
    avg_yield = df['rental_yield'].mean()
    df['yield_score'] = df.apply(
        lambda row: min(30, max(0, ((row['rental_yield'] - avg_yield) / 0.03) * 30)), 
        axis=1
    )
    
    # 3. Price to rent ratio component (20% of total score)
    avg_ptr = df['price_to_rent_ratio'].mean()
    df['ptr_score'] = df.apply(
        lambda row: min(20, max(0, ((avg_ptr - row['price_to_rent_ratio']) / 5) * 20)),
        axis=1
    )
    
    # 4. Location premium (10% of total score)
    area_premiums = {
        'Palm Jumeirah': 9,
        'Marina': 8,
        'Downtown': 8,
        'Business Bay': 7,
        'JBR': 7,
        'JLT': 6,
        'Sports City': 5,
        'Silicon Oasis': 4,
        'Discovery Gardens': 4,
        'Suburbs': 3
    }
    
    df['location_score'] = df['area_name'].map(lambda x: area_premiums.get(x, 5))
    
    # Overall investment score
    df['investment_score'] = df['valuation_score'] + df['yield_score'] + df['ptr_score'] + df['location_score']
    
    # Round numerical values for cleaner display
    df['property_price'] = df['property_price'].round(0)
    df['annual_rental_price'] = df['annual_rental_price'].round(0)
    df['rental_yield'] = df['rental_yield'] * 100  # Convert to percentage
    df['fair_market_value'] = df['fair_market_value'].round(0)
    df['value_difference'] = df['value_difference'].round(0)
    df['value_difference_percent'] = df['value_difference_percent'].round(1)
    df['price_per_sqft'] = df['price_per_sqft'].round(0)
    
    return df

# Load or generate data
def load_data():
    # Try to load from CSV file
    file_path = '/home/maveron/Projects/SyntaxError-Estate-Agent/data/training_data/investment_data.csv'
    try:
        # Replace with your actual file path'
        
        df = pd.read_csv(file_path)
        print(f"Data loaded from {file_path}")
        return df
    except:
        # If file not found, generate sample data
        print("CSV file not found, generating sample investment data")
        return generate_sample_investment_data()

# Load the data
investment_data = load_data()

# Create a Dash application
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define the layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Investment Property Dashboard", className="text-center my-4")
        ], width=12)
    ]),
    
    # Summary statistics cards
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Total Properties", className="card-title"),
                    html.H2(f"{len(investment_data):,}", className="card-text text-primary")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Avg Property Price", className="card-title"),
                    html.H2(f"{investment_data['property_price'].mean():,.0f} AED", className="card-text text-success")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Avg Rental Yield", className="card-title"),
                    html.H2(f"{investment_data['rental_yield'].mean():.2f}%", className="card-text text-info")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Avg Investment Score", className="card-title"),
                    html.H2(f"{investment_data['investment_score'].mean():.1f}/100", className="card-text text-warning")
                ])
            ])
        ], width=3),
    ], className="mb-4"),
    
    # Property Type and Area filters
    dbc.Row([
        dbc.Col([
            html.H4("Filter Data"),
            dbc.Row([
                dbc.Col([
                    html.Label("Property Type"),
                    dcc.Dropdown(
                        id='property-type-dropdown',
                        options=[{'label': t, 'value': t} for t in sorted(investment_data['property_type'].unique())],
                        multi=True,
                        placeholder="Select property type(s)"
                    )
                ], width=6),
                dbc.Col([
                    html.Label("Area"),
                    dcc.Dropdown(
                        id='area-dropdown',
                        options=[{'label': a, 'value': a} for a in sorted(investment_data['area_name'].unique())],
                        multi=True,
                        placeholder="Select area(s)"
                    )
                ], width=6)
            ])
        ], width=12)
    ], className="mb-4"),
    
    # Charts - First row
    dbc.Row([
        # Rental Yield by Property Type
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Rental Yield by Property Type"),
                dbc.CardBody([
                    dcc.Graph(id="yield-property-type-chart")
                ])
            ])
        ], width=6),
        
        # Price Per Sqft by Area
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Price Per Sqft by Area (Top 8)"),
                dbc.CardBody([
                    dcc.Graph(id="price-sqft-area-chart")
                ])
            ])
        ], width=6)
    ], className="mb-4"),
    
    # Charts - Second row
    dbc.Row([
        # Investment Score Components
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Investment Score Components"),
                dbc.CardBody([
                    dcc.Graph(id="investment-score-components-chart")
                ])
            ])
        ], width=6),
        
        # Top Investment Opportunities
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Top 10 Investment Opportunities"),
                dbc.CardBody([
                    dcc.Graph(id="top-investments-chart")
                ])
            ])
        ], width=6)
    ], className="mb-4"),
    
    # Charts - Third row
    dbc.Row([
        # Price vs Size Scatter
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Property Price vs Size"),
                dbc.CardBody([
                    dcc.Graph(id="price-vs-size-chart")
                ])
            ])
        ], width=6),
        
        # Rental Yield vs Price Scatter
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Rental Yield vs Property Price"),
                dbc.CardBody([
                    dcc.Graph(id="yield-vs-price-chart")
                ])
            ])
        ], width=6)
    ], className="mb-4"),
    
    # Charts - Fourth row
    dbc.Row([
        # Price to Rent Ratio by Area
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Price to Rent Ratio by Area (Lower is Better)"),
                dbc.CardBody([
                    dcc.Graph(id="price-rent-ratio-chart")
                ])
            ])
        ], width=6),
        
        # Valuation Analysis
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Valuation Analysis (Under/Overvalued)"),
                dbc.CardBody([
                    dcc.Graph(id="valuation-analysis-chart")
                ])
            ])
        ], width=6)
    ], className="mb-4"),
    
    # Data Preview
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Data Preview"),
                dbc.CardBody([
                    dash_table.DataTable(
                        id="data-preview-table",
                        columns=[{"name": col, "id": col} for col in investment_data.columns],
                        data=investment_data.head(10).to_dict('records'),
                        style_table={'overflowX': 'auto'},
                        style_cell={
                            'textAlign': 'left',
                            'minWidth': '100px', 'width': '100px', 'maxWidth': '200px',
                            'overflow': 'hidden',
                            'textOverflow': 'ellipsis',
                        }
                    )
                ])
            ])
        ], width=12)
    ])
], fluid=True)

# Define callbacks
@app.callback(
    [
        Output("yield-property-type-chart", "figure"),
        Output("price-sqft-area-chart", "figure"),
        Output("investment-score-components-chart", "figure"),
        Output("top-investments-chart", "figure"),
        Output("price-vs-size-chart", "figure"),
        Output("yield-vs-price-chart", "figure"),
        Output("price-rent-ratio-chart", "figure"),
        Output("valuation-analysis-chart", "figure"),
        Output("data-preview-table", "data")
    ],
    [
        Input("property-type-dropdown", "value"),
        Input("area-dropdown", "value")
    ]
)
def update_charts(selected_property_types, selected_areas):
    # Filter the data based on selections
    filtered_data = investment_data.copy()
    
    if selected_property_types and len(selected_property_types) > 0:
        filtered_data = filtered_data[filtered_data['property_type'].isin(selected_property_types)]
        
    if selected_areas and len(selected_areas) > 0:
        filtered_data = filtered_data[filtered_data['area_name'].isin(selected_areas)]
    
    # If no data after filtering, return empty charts
    if len(filtered_data) == 0:
        empty_figs = [px.bar(), px.bar(), px.bar(), px.bar(), 
                      px.scatter(), px.scatter(), px.bar(), px.bar()]
        return empty_figs + [[]]
    
    # 1. Rental Yield by Property Type
    yield_by_type = filtered_data.groupby('property_type')['rental_yield'].mean().reset_index()
    yield_property_type_chart = px.bar(
        yield_by_type,
        x='property_type',
        y='rental_yield',
        title='Average Rental Yield by Property Type',
        labels={'property_type': 'Property Type', 'rental_yield': 'Average Rental Yield (%)'},
        color='rental_yield',
        color_continuous_scale=px.colors.sequential.Viridis,
        text_auto='.2f'
    )
    
    yield_property_type_chart.update_traces(texttemplate='%{y:.2f}%', textposition='outside')
    
    # 2. Price Per Sqft by Area (Top 8)
    price_sqft_by_area = filtered_data.groupby('area_name')['price_per_sqft'].mean().reset_index()
    price_sqft_by_area = price_sqft_by_area.sort_values('price_per_sqft', ascending=False).head(8)
    
    price_sqft_area_chart = px.bar(
        price_sqft_by_area,
        y='area_name',
        x='price_per_sqft',
        title='Price Per Sqft by Area (Top 8)',
        labels={'area_name': 'Area', 'price_per_sqft': 'Price Per Sqft (AED)'},
        color='price_per_sqft',
        color_continuous_scale=px.colors.sequential.Plasma,
        orientation='h',
        text_auto='.0f'
    )
    
    price_sqft_area_chart.update_traces(texttemplate='%{x:.0f}', textposition='outside')
    
    # 3. Investment Score Components
    # Create a dataframe with average scores
    score_components = pd.DataFrame({
        'Component': ['Valuation', 'Rental Yield', 'Price to Rent', 'Location'],
        'Score': [
            filtered_data['valuation_score'].mean(),
            filtered_data['yield_score'].mean(),
            filtered_data['ptr_score'].mean(),
            filtered_data['location_score'].mean()
        ],
        'Max': [40, 30, 20, 10]  # Maximum possible score for each component
    })
    
    # Calculate percentage of maximum for each component
    score_components['Percentage'] = (score_components['Score'] / score_components['Max']) * 100
    
    investment_score_chart = px.bar(
        score_components,
        y='Component',
        x='Score',
        title=f'Investment Score Components (Avg: {filtered_data["investment_score"].mean():.1f}/100)',
        labels={'Component': 'Score Component', 'Score': 'Average Score'},
        color='Percentage',
        color_continuous_scale=px.colors.sequential.RdBu,
        orientation='h',
        text_auto='.1f'
    )
    
    investment_score_chart.update_traces(texttemplate='%{x:.1f}', textposition='outside')
    
    # Add line showing max possible score for each component
    for i, row in score_components.iterrows():
        investment_score_chart.add_shape(
            type="line",
            x0=0, x1=row['Max'],
            y0=i, y1=i,
            line=dict(color="rgba(0,0,0,0.3)", width=2, dash="dot")
        )
    
    # 4. Top Investment Opportunities
    # Sort by investment score and take top 10
    top_investments = filtered_data.sort_values('investment_score', ascending=False).head(10)
    
    top_investments_chart = px.bar(
        top_investments,
        y='area_name',
        x='investment_score',
        title='Top 10 Investment Opportunities',
        labels={'area_name': 'Area', 'investment_score': 'Investment Score'},
        color='investment_score',
        color_continuous_scale=px.colors.sequential.Greens,
        orientation='h',
        hover_data=['property_type', 'rental_yield', 'property_price', 'value_difference_percent'],
        text_auto='.1f'
    )
    
    top_investments_chart.update_traces(texttemplate='%{x:.1f}', textposition='outside')
    
    # 5. Price vs Size Scatter Plot
    price_vs_size_chart = px.scatter(
        filtered_data,
        x='size_sqft',
        y='property_price',
        title='Property Price vs Size',
        labels={'size_sqft': 'Size (sqft)', 'property_price': 'Property Price (AED)'},
        color='property_type',
        size='investment_score',  # Size points by investment score
        hover_data=['area_name', 'rental_yield', 'bed', 'bath']
    )
    
    # Add trendline
    price_vs_size_chart.update_layout(
        shapes=[
            {
                'type': 'line',
                'x0': filtered_data['size_sqft'].min(),
                'y0': filtered_data['property_price'].min(),
                'x1': filtered_data['size_sqft'].max(),
                'y1': filtered_data['property_price'].max(),
                'line': {'color': 'red', 'dash': 'dash'}
            }
        ]
    )
    
    # 6. Rental Yield vs Price Scatter
    yield_vs_price_chart = px.scatter(
        filtered_data,
        x='property_price',
        y='rental_yield',
        title='Rental Yield vs Property Price',
        labels={'property_price': 'Property Price (AED)', 'rental_yield': 'Rental Yield (%)'},
        color='property_type',
        size='investment_score',  # Size points by investment score
        hover_data=['area_name', 'bed', 'bath', 'price_to_rent_ratio']
    )
    
    # Add reference lines for average yield
    avg_yield = filtered_data['rental_yield'].mean()
    yield_vs_price_chart.add_shape(
        type="line",
        x0=filtered_data['property_price'].min(),
        y0=avg_yield,
        x1=filtered_data['property_price'].max(),
        y1=avg_yield,
        line=dict(color="red", width=2, dash="dash")
    )
    
    yield_vs_price_chart.add_annotation(
        x=filtered_data['property_price'].max(),
        y=avg_yield,
        text=f"Avg Yield: {avg_yield:.2f}%",
        showarrow=False,
        xshift=10,
        yshift=10
    )
    
    # 7. Price to Rent Ratio by Area
    ptr_by_area = filtered_data.groupby('area_name')['price_to_rent_ratio'].mean().reset_index()
    ptr_by_area = ptr_by_area.sort_values('price_to_rent_ratio')  # Lower is better
    
    price_rent_ratio_chart = px.bar(
        ptr_by_area,
        x='area_name',
        y='price_to_rent_ratio',
        title='Price to Rent Ratio by Area (Lower is Better for Investment)',
        labels={'area_name': 'Area', 'price_to_rent_ratio': 'Price to Rent Ratio'},
        color='price_to_rent_ratio',
        color_continuous_scale=px.colors.sequential.Viridis_r,  # Reversed so darker is better (lower)
        text_auto='.1f'
    )
    
    price_rent_ratio_chart.update_traces(texttemplate='%{y:.1f}', textposition='outside')
    price_rent_ratio_chart.update_layout(xaxis={'categoryorder': 'total ascending'})
    
    # Add reference line for average price to rent ratio
    avg_ptr = filtered_data['price_to_rent_ratio'].mean()
    price_rent_ratio_chart.add_shape(
        type="line",
        x0=-0.5, 
        x1=len(ptr_by_area)-0.5,
        y0=avg_ptr,
        y1=avg_ptr,
        line=dict(color="red", width=2, dash="dash")
    )
    
    price_rent_ratio_chart.add_annotation(
        x=0,
        y=avg_ptr,
        text=f"Avg: {avg_ptr:.1f}",
        showarrow=False,
        xshift=10,
        yshift=10
    )
    
    # 8. Valuation Analysis
    # Create a dataframe with counts of under/overvalued properties by area
    valuation_data = filtered_data.copy()
    valuation_data['valuation_status'] = np.where(valuation_data['value_difference'] > 0, 'Undervalued', 'Overvalued')
    
    valuation_counts = valuation_data.groupby(['area_name', 'valuation_status']).size().reset_index(name='count')
    
    valuation_analysis_chart = px.bar(
        valuation_counts,
        x='area_name',
        y='count',
        color='valuation_status',
        barmode='group',
        title='Valuation Analysis by Area',
        labels={'area_name': 'Area', 'count': 'Number of Properties', 'valuation_status': 'Valuation Status'},
        color_discrete_map={'Undervalued': 'green', 'Overvalued': 'red'}
    )
    
    # Add value labels
    valuation_analysis_chart.update_traces(texttemplate='%{y}', textposition='outside')
    
    # 9. Data Preview (filtered)
    data_preview = filtered_data.head(10).to_dict('records')
    
    return [
        yield_property_type_chart, 
        price_sqft_area_chart, 
        investment_score_chart, 
        top_investments_chart,
        price_vs_size_chart,
        yield_vs_price_chart,
        price_rent_ratio_chart,
        valuation_analysis_chart,
        data_preview
    ]

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
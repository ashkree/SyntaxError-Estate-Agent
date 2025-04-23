import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, html, dcc, Input, Output, dash_table
import dash_bootstrap_components as dbc

# Function to generate sample rental data (for testing)
def generate_sample_data(n_contracts=200):
    # Areas
    areas = ['Downtown', 'Marina', 'Suburbs', 'Business Bay', 'Palm Jumeirah', 
            'JBR', 'Silicon Oasis', 'Sports City', 'Discovery Gardens', 'JLT']
    
    # Property types
    property_types = ['Apartment', 'Villa', 'Townhouse', 'Office', 'Retail']
    
    # Property usage
    usages = ['Residential', 'Commercial', 'Mixed Use']
    
    # Generate random dates within the last 3 years
    years = [2022, 2023, 2024]
    months = list(range(1, 13))
    
    # Generate contract data
    data = {
        'area_name_en': np.random.choice(areas, n_contracts),
        'ejari_property_type_en': np.random.choice(property_types, n_contracts, 
                                                  p=[0.6, 0.2, 0.1, 0.07, 0.03]),  # More apartments
        'property_usage_en': np.random.choice(usages, n_contracts, p=[0.7, 0.25, 0.05]),
        'contract_year': np.random.choice(years, n_contracts),
        'contract_month': np.random.choice(months, n_contracts)
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Calculate quarter
    df['contract_quarter'] = df['contract_month'].apply(lambda x: (x-1)//3 + 1)
    
    # Create year-quarter field
    df['year_quarter'] = df['contract_year'].astype(str) + '-Q' + df['contract_quarter'].astype(str)
    
    # Base rent calculations with adjustments for area and property type
    df['base_rent'] = np.random.uniform(5000, 15000, n_contracts)
    
    # Premium areas have higher rent
    area_multipliers = {
        'Palm Jumeirah': 2.0,
        'Marina': 1.5,
        'Downtown': 1.5,
        'Business Bay': 1.3,
        'JBR': 1.4,
        'JLT': 1.2,
        'Sports City': 0.9,
        'Silicon Oasis': 0.8,
        'Discovery Gardens': 0.85,
        'Suburbs': 0.7
    }
    
    # Property type multipliers
    type_multipliers = {
        'Villa': 1.8,
        'Townhouse': 1.5,
        'Apartment': 1.0,
        'Office': 1.2,
        'Retail': 1.3
    }
    
    # Usage multipliers
    usage_multipliers = {
        'Residential': 1.0,
        'Commercial': 1.3,
        'Mixed Use': 1.2
    }
    
    # Seasonal adjustment - higher in winter months
    seasonal_multipliers = {
        1: 1.1,  # January
        2: 1.1,  # February
        3: 1.05, # March
        4: 1.0,  # April
        5: 0.95, # May
        6: 0.9,  # June
        7: 0.9,  # July
        8: 0.95, # August
        9: 1.0,  # September
        10: 1.05, # October
        11: 1.1,  # November
        12: 1.15  # December
    }
    
    # Apply multipliers
    for i, row in df.iterrows():
        area_mult = area_multipliers.get(row['area_name_en'], 1.0)
        type_mult = type_multipliers.get(row['ejari_property_type_en'], 1.0)
        usage_mult = usage_multipliers.get(row['property_usage_en'], 1.0)
        season_mult = seasonal_multipliers.get(row['contract_month'], 1.0)
        
        # Add slight randomization (±10%)
        random_factor = np.random.uniform(0.9, 1.1)
        
        df.at[i, 'monthly_rent'] = row['base_rent'] * area_mult * type_mult * usage_mult * season_mult * random_factor
    
    # Calculate annual rent
    df['annual_rent'] = df['monthly_rent'] * 12
    
    # Round the rent values
    df['monthly_rent'] = df['monthly_rent'].round(0)
    df['annual_rent'] = df['annual_rent'].round(0)
    
    return df

# Load or generate data
def load_data():
    # Try to load from CSV file
    try:
        # Replace with your actual file path
        file_path = '../data/training_data/rent_data.csv'
        df = pd.read_csv(file_path)
        print(f"Data loaded from {file_path}")
        return df
    except:
        # If file not found, generate sample data
        print("CSV file not found, generating sample data")
        return generate_sample_data()

# Load the data
rental_data = load_data()

# Create a Dash application
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define the layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Rental Data Dashboard", className="text-center my-4")
        ], width=12)
    ]),
    
    # Summary statistics cards
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Total Contracts", className="card-title"),
                    html.H2(f"{len(rental_data):,}", className="card-text text-primary")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Avg Monthly Rent", className="card-title"),
                    html.H2(f"{rental_data['monthly_rent'].mean():,.0f} AED", className="card-text text-success")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Highest Rent Area", className="card-title"),
                    html.H2(
                        rental_data.groupby('area_name_en')['monthly_rent'].mean().sort_values(ascending=False).index[0], 
                        className="card-text text-info"
                    )
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Most Common Property", className="card-title"),
                    html.H2(
                        rental_data['ejari_property_type_en'].value_counts().index[0], 
                        className="card-text text-warning"
                    )
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
                        options=[{'label': t, 'value': t} for t in sorted(rental_data['ejari_property_type_en'].unique())],
                        multi=True,
                        placeholder="Select property type(s)"
                    )
                ], width=6),
                dbc.Col([
                    html.Label("Area"),
                    dcc.Dropdown(
                        id='area-dropdown',
                        options=[{'label': a, 'value': a} for a in sorted(rental_data['area_name_en'].unique())],
                        multi=True,
                        placeholder="Select area(s)"
                    )
                ], width=6)
            ])
        ], width=12)
    ], className="mb-4"),
    
    # Charts - First row
    dbc.Row([
        # Property Type Chart
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Average Rent by Property Type"),
                dbc.CardBody([
                    dcc.Graph(id="property-type-chart")
                ])
            ])
        ], width=6),
        
        # Top Areas Chart
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Top 5 Areas by Average Rent"),
                dbc.CardBody([
                    dcc.Graph(id="top-areas-chart")
                ])
            ])
        ], width=6)
    ], className="mb-4"),
    
    # Charts - Second row
    dbc.Row([
        # Seasonal Trends Chart
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Seasonal Rent Trends by Month"),
                dbc.CardBody([
                    dcc.Graph(id="seasonal-trends-chart")
                ])
            ])
        ], width=6),
        
        # Yearly Trends Chart
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Yearly Rent Trends"),
                dbc.CardBody([
                    dcc.Graph(id="yearly-trends-chart")
                ])
            ])
        ], width=6)
    ], className="mb-4"),
    
    # Charts - Third row
    dbc.Row([
        # Property Usage Chart
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Rent by Property Usage"),
                dbc.CardBody([
                    dcc.Graph(id="property-usage-chart")
                ])
            ])
        ], width=6),
        
        # Quarterly Trends Chart
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Quarterly Rent Trends"),
                dbc.CardBody([
                    dcc.Graph(id="quarterly-trends-chart")
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
                        columns=[{"name": col, "id": col} for col in rental_data.columns],
                        data=rental_data.head(10).to_dict('records'),
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
        Output("property-type-chart", "figure"),
        Output("top-areas-chart", "figure"),
        Output("seasonal-trends-chart", "figure"),
        Output("yearly-trends-chart", "figure"),
        Output("property-usage-chart", "figure"),
        Output("quarterly-trends-chart", "figure"),
        Output("data-preview-table", "data")
    ],
    [
        Input("property-type-dropdown", "value"),
        Input("area-dropdown", "value")
    ]
)
def update_charts(selected_property_types, selected_areas):
    # Filter the data based on selections
    filtered_data = rental_data.copy()
    
    if selected_property_types and len(selected_property_types) > 0:
        filtered_data = filtered_data[filtered_data['ejari_property_type_en'].isin(selected_property_types)]
        
    if selected_areas and len(selected_areas) > 0:
        filtered_data = filtered_data[filtered_data['area_name_en'].isin(selected_areas)]
    
    # If no data after filtering, return empty charts
    if len(filtered_data) == 0:
        return [px.bar(), px.bar(), px.line(), px.line(), px.bar(), px.line(), []]
    
    # 1. Property Type Chart
    property_type_df = filtered_data.groupby('ejari_property_type_en')['monthly_rent'].mean().reset_index()
    property_type_chart = px.bar(
        property_type_df,
        x='ejari_property_type_en',
        y='monthly_rent',
        title='Average Monthly Rent by Property Type',
        labels={'ejari_property_type_en': 'Property Type', 'monthly_rent': 'Average Monthly Rent (AED)'},
        color='monthly_rent',
        color_continuous_scale=px.colors.sequential.Viridis
    )
    
    # 2. Top Areas Chart
    top_areas_df = filtered_data.groupby('area_name_en')['monthly_rent'].mean().reset_index().sort_values('monthly_rent', ascending=False).head(5)
    top_areas_chart = px.bar(
        top_areas_df,
        y='area_name_en',
        x='monthly_rent',
        title='Top 5 Areas by Average Rent',
        labels={'area_name_en': 'Area', 'monthly_rent': 'Average Monthly Rent (AED)'},
        color='monthly_rent',
        color_continuous_scale=px.colors.sequential.Plasma,
        orientation='h'
    )
    
    # 3. Seasonal Trends Chart
    monthly_df = filtered_data.groupby('contract_month')['monthly_rent'].mean().reset_index()
    monthly_df = monthly_df.sort_values('contract_month')
    
    # Ensure all months are represented (1-12)
    all_months = pd.DataFrame({'contract_month': range(1, 13)})
    monthly_df = pd.merge(all_months, monthly_df, on='contract_month', how='left').fillna(0)
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_df['month_name'] = monthly_df['contract_month'].apply(lambda x: month_names[int(x)-1])
    
    seasonal_chart = px.line(
        monthly_df,
        x='month_name',
        y='monthly_rent',
        title='Seasonal Rent Trends by Month',
        labels={'month_name': 'Month', 'monthly_rent': 'Average Monthly Rent (AED)'},
        markers=True
    )
    
    # Customize the x-axis to show month names in correct order
    seasonal_chart.update_layout(xaxis={'categoryorder': 'array', 'categoryarray': month_names})
    
    # 4. Yearly Trends Chart
    yearly_df = filtered_data.groupby('contract_year')['monthly_rent'].agg(['mean', 'count']).reset_index()
    
    yearly_chart = px.line(
        yearly_df,
        x='contract_year',
        y='mean',
        title='Yearly Rent Trends',
        labels={'contract_year': 'Year', 'mean': 'Average Monthly Rent (AED)'},
        markers=True
    )
    
    # Add count annotations
    # Note: No need for slicing here since we're adding annotations for all years,
    # but still converting to list for consistency
    for i, row in list(yearly_df.iterrows()):
        yearly_chart.add_annotation(
            x=row['contract_year'],
            y=row['mean'],
            text=f"n={row['count']}",
            showarrow=False,
            yshift=10
        )
    
    # 5. Property Usage Chart
    usage_df = filtered_data.groupby('property_usage_en')['monthly_rent'].mean().reset_index()
    usage_chart = px.bar(
        usage_df,
        x='property_usage_en',
        y='monthly_rent',
        title='Average Rent by Property Usage',
        labels={'property_usage_en': 'Property Usage', 'monthly_rent': 'Average Monthly Rent (AED)'},
        color='monthly_rent',
        color_continuous_scale=px.colors.sequential.Cividis
    )
    
    # 6. Quarterly Trends Chart
    quarterly_df = filtered_data.groupby('year_quarter')['monthly_rent'].agg(['mean', 'count']).reset_index()
    quarterly_df = quarterly_df.sort_values('year_quarter')
    
    quarterly_chart = px.line(
        quarterly_df,
        x='year_quarter',
        y='mean',
        title='Quarterly Rent Trends',
        labels={'year_quarter': 'Year-Quarter', 'mean': 'Average Monthly Rent (AED)'},
        markers=True
    )
    
    # Add count annotations to quarterly chart (skip some if too many)
    skip_factor = max(1, len(quarterly_df) // 8)  # Show at most ~8 labels
    
    # Convert iterrows generator to a list that we can slice
    rows_to_annotate = list(quarterly_df.iterrows())
    
    # Apply skip factor
    for i, row in rows_to_annotate[::skip_factor]:
        quarterly_chart.add_annotation(
            x=row['year_quarter'],
            y=row['mean'],
            text=f"n={row['count']}",
            showarrow=False,
            yshift=10
        )
        
    # 7. Data Preview (filtered)
    data_preview = filtered_data.head(10).to_dict('records')
    
    return [property_type_chart, top_areas_chart, seasonal_chart, yearly_chart, usage_chart, quarterly_chart, data_preview]

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
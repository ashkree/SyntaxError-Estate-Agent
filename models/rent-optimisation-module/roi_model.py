import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Assuming your dataframe is loaded as df
# df = pd.read_csv('your_data.csv')

# Let's visualize the relationship first
plt.figure(figsize=(10, 6))
plt.scatter(df['property_total_value'], df['annual_roi'], alpha=0.5)
plt.title('Property Value vs Annual ROI')
plt.xlabel('Property Total Value')
plt.ylabel('Annual ROI')
plt.show()

# Prepare features (X) and target (y)
X = df[['property_total_value']]  # Features in double brackets to keep as dataframe
y = df['annual_roi']  # Target

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Coefficient (slope): {model.coef_[0]:.6f}')
print(f'Intercept: {model.intercept_:.6f}')
print(f'Mean Squared Error: {mse:.6f}')
print(f'R² Score: {r2:.6f}')  # How much variance in ROI is explained by property value

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', alpha=0.5, label='Actual ROI')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted ROI')
plt.title('Linear Regression: Property Value vs Annual ROI')
plt.xlabel('Property Total Value')
plt.ylabel('Annual ROI')
plt.legend()
plt.show()

# Function to predict future ROI
def predict_roi(property_value):
    return model.predict([[property_value]])[0]

# Example usage
example_value = 500000  # Example property value
predicted_roi = predict_roi(example_value)
print(f'Predicted ROI for property value ${example_value}: {predicted_roi:.2f}%')
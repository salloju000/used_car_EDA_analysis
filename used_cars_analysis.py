# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Load dataset
data = pd.read_csv('used_cars.csv')

# Preview data and get basic info
print(data.head())
print("Dataset shape:", data.shape)
print(data.info())
print(data.describe())
print("Missing values per column:\n", data.isnull().sum())

# Clean column names by removing extra spaces
data.columns = data.columns.str.strip()

# Show sample unique values from key columns
print("Sample milage values:", data['milage'].unique()[:10])
print("Sample engine values:", data['engine'].unique()[:10])
print("Sample price values:", data['price'].unique()[:10])

# ---------------------------
# Data Cleaning & Transformation
# ---------------------------

# Convert 'milage' to numeric kilometers
data['milage'] = data['milage'].astype(str)                      # Make sure milage is string
data['milage'] = data['milage'].replace({',': ''}, regex=True)   # Remove commas if any
data['milage'] = data['milage'].str.extract('(\d+)').astype(int) # Extract numeric part
data['milage'] = (data['milage'] * 1.60934).astype(int)          # Convert miles to km

print("Converted milage:\n", data['milage'])

# Clean and convert price to float and add INR conversion
data['price'] = data['price'].replace({r'\$': '', r'\s+': '', ',': ''}, regex=True)
data['price'] = data['price'].astype(float)
data['price_inr'] = data['price'] * 85.10  # Convert USD to INR approx

print("Price in INR:\n", data['price_inr'])

# ---------------------------
# Extract horsepower and engine capacity
# ---------------------------

def extract_hp(engine_str):
    """Extract horsepower (HP) from engine description string."""
    match = re.search(r'(\d{2,4})\.?0?HP', engine_str)
    return int(match.group(1)) if match else None

def extract_liters(engine_str):
    """Extract engine capacity in liters from engine description string."""
    match = re.search(r'(\d\.\d)L', engine_str)
    return float(match.group(1)) if match else None

# Apply extraction functions to 'engine' column
data['engine_hp'] = data['engine'].apply(extract_hp)
data['engine_liters'] = data['engine'].apply(extract_liters)

print("Extracted engine horsepower:\n", data['engine_hp'])

# ---------------------------
# Handle missing values and categorical data cleaning
# ---------------------------

print("Fuel Type Values:", data['fuel_type'].unique())
print("Accident Values:", data['accident'].unique())

# Replace invalid fuel_type entries with NaN
data['fuel_type'].replace(['-', 'not supported'], np.nan, inplace=True)

# Fill missing fuel_type with the mode (most common value)
fuel_mode = data['fuel_type'].mode()[0]
data['fuel_type'].fillna(fuel_mode, inplace=True)

# Fill missing accident info with 'Unknown'
data['accident'].fillna('Unknown', inplace=True)

# Check counts after filling missing values
print("Fuel Type Counts:\n", data['fuel_type'].value_counts())
print("Accident Counts:\n", data['accident'].value_counts())

# ---------------------------
# Visualization Section
# ---------------------------

# Price distribution and boxplot (side-by-side)
plt.figure(figsize=(10,4))

plt.subplot(1, 2, 1)
sns.histplot(data['price'], kde=True, bins=30)
plt.title("Price Distribution")
plt.xlabel("Price")

plt.subplot(1, 2, 2)
sns.boxplot(x=data['price'])
plt.title("Price Boxplot")

plt.tight_layout()
plt.show()
# Price Distribution and Outlier Analysis
# The price distribution of used cars is right-skewed, showing that most vehicles are priced on the lower end with fewer high-priced options. 
# The boxplot highlights the presence of several high-value outliers, likely representing premium or luxury models.

# Scatterplot: Milage vs Price
plt.figure(figsize=(7,4))
sns.scatterplot(data=data, x='milage', y='price')
plt.title("Milage vs Price")
plt.xlabel("Milage (km)")
plt.ylabel("Price")
plt.tight_layout()
plt.show()
# Relationship Between Mileage and Price
# A strong negative correlation is observed between mileage and price. 
# As the mileage of a vehicle increases, its resale value tends to decline, emphasizing that lower mileage cars retain higher market value.

# Average price by car brand
plt.figure(figsize=(14,6))
brand_avg_price = data.groupby('brand')['price'].mean().sort_values(ascending=False)
sns.barplot(x=brand_avg_price.index, y=brand_avg_price.values)
plt.xticks(rotation=45)
plt.title("Average Price by Car Brand")
plt.xlabel("Car Brand")
plt.ylabel("Average Price")
plt.show()
# Average Price Across Different Car Brands
# Certain car brands consistently command higher average prices, reflecting brand reputation, perceived quality, and luxury status. 
# This insight can inform buyers about expected price variations across brands.

# Correlation heatmap
plt.figure(figsize=(10,4))
sns.heatmap(data[['price', 'milage', 'engine_hp', 'engine_liters']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
# Correlation Among Numerical Features
# The correlation heatmap reveals that price negatively correlates with mileage and positively correlates with horsepower and engine capacity. 
# This indicates that both vehicle usage and engine performance significantly influence car prices.

# Price vs Transmission boxplot
plt.figure(figsize=(10,4))
sns.boxplot(data=data, x='transmission', y='price')
plt.title("Transmission Type vs Price")
plt.xticks(rotation=45)
plt.show()
#Impact of Transmission Type on Price
# Vehicles with automatic transmission generally have higher prices compared to those with manual transmission. 
# This trend reflects buyer preferences for convenience and potentially better resale value of automatic cars.

 
# Price vs Fuel Type boxplot
plt.figure(figsize=(10,4))
sns.boxplot(data=data, x='fuel_type', y='price')
plt.title("Fuel Type vs Price")
plt.xticks(rotation=45)
plt.show()
# Price Variation by Fuel Type
# Hybrid and petrol vehicles tend to have higher prices relative to diesel and electric models in this dataset. 
# Fuel type appears to be an important factor affecting vehicle valuation and buyer preferences.

# Price vs Accident History boxplot
plt.figure(figsize=(7,5))
sns.boxplot(data=data, x='accident', y='price')
plt.title("Accident History vs Price")
plt.xticks(rotation=10)
plt.show()
# Effect of Accident History on Price
# Cars with a clean accident history show higher median prices, while those with recorded accidents or unknown histories tend to be priced lower. 
# This suggests that accident history strongly influences buyer trust and vehicle valuation.

# ---------------------------
# Summary Comments for report/resume
# ---------------------------

"""
Summary:
- Automatic and hybrid cars generally have higher prices.
- Cars with no accident history are valued higher.
- Fuel type and transmission strongly correlate with price.
- Lower mileage and higher engine performance (HP, liters) increase car value.
"""

# Additional correlation matrix plot for reporting
plt.figure(figsize=(8,6))
corr = data[['price', 'milage', 'engine_hp', 'engine_liters']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap")
plt.show()

"""
Correlation Summary:
- Price negatively correlates with mileage.
- Price positively correlates with horsepower and engine size.
- Suggests low-mileage, high-performance cars retain higher value.
"""

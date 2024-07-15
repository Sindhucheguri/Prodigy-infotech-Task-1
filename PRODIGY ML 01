import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer  # Import SimpleImputer
from sklearn.metrics import mean_squared_error
from scipy.stats import skew

# Read data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Define columns to keep
columns_to_keep = ['LotArea', 'BedroomAbvGr','BsmtFullBath', 'BsmtHalfBath', 'HalfBath', 'TotalBsmtSF',  'FullBath', 'SalePrice']

# Filter train and test datasets to keep only required columns
train = train[columns_to_keep]
test = test[['LotArea', 'BedroomAbvGr','BsmtFullBath', 'BsmtHalfBath', 'HalfBath', 'TotalBsmtSF',  'FullBath']]  # No SalePrice column in the test dataset

# Log transform the target variable
train['SalePrice'] = np.log1p(train['SalePrice'])# avoid potential errors when dealing with data that includes zero or negative values
new_skewness = skew(train['SalePrice'])

print("Skewness after logarithmic transformation:", new_skewness)

# Split data into features and target
X = train.drop('SalePrice', axis=1)
y = train['SalePrice']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.33)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Impute missing values in test data
imputer = SimpleImputer(strategy='mean')
X_test_scaled = imputer.fit_transform(X_test_scaled)

# Train Ridge regression model
ridge = Ridge(alpha=1.0)  # You can tune alpha
ridge.fit(X_train_scaled, y_train)

# Make predictions
predictions = ridge.predict(X_test_scaled)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)

# Plot actual vs. predicted values
plt.scatter(np.expm1(y_test), np.expm1(predictions))
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')
plt.title('Actual vs. Predicted Sale Price')
plt.show()

# Impute missing values in test data for final prediction
test_scaled = scaler.transform(test)
test_scaled = imputer.transform(test_scaled)

# Predict SalePrice for test data
test_predictions = ridge.predict(test_scaled)

# Convert SalePrice predictions back to original scale
predicted_sale_price = np.expm1(test_predictions)

# Plot LotArea vs. Predicted Sale Price
plt.scatter(test['LotArea'], predicted_sale_price)
plt.xlabel('Lot Area')
plt.ylabel('Predicted Sale Price')
plt.title('Lot Area vs. Predicted Sale Price')
plt.show()

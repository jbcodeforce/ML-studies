from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

np.random.seed(0)
n = 100
p = 10
X = np.random.randn(n, p)
true_coef = np.random.randn(p)
y = X.dot(true_coef) + np.random.randn(n)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define alpha values
alphas = [0.01, 0.1, 1, 10]

# Initialize lists to store Mean Square Error (MSE) and coefficients for Lasso and Ridge
lasso_mse = []
ridge_mse = []
lasso_coefs = []
ridge_coefs = []

# Fit models and compute MSE and coefficients for different alpha values
for alpha in alphas:
    lasso = Lasso(alpha=alpha)
    ridge = Ridge(alpha=alpha)
    lasso.fit(X_train, y_train)
    ridge.fit(X_train, y_train)
    
    # Compute predictions and MSE
    lasso_pred = lasso.predict(X_test)
    ridge_pred = ridge.predict(X_test)
    lasso_mse.append(mean_squared_error(y_test, lasso_pred))
    ridge_mse.append(mean_squared_error(y_test, ridge_pred))
    
    # Store coefficients
    lasso_coefs.append(lasso.coef_)
    ridge_coefs.append(ridge.coef_)

# Display MSE results
results_mse = pd.DataFrame({'Alpha': alphas, 'Lasso MSE': lasso_mse, 'Ridge MSE': ridge_mse})
print("MSE Results:")
print(results_mse)

# Display resulting equations
print("\nLasso Regression Coefficients:")
for i, alpha in enumerate(alphas):
    print(f"Alpha: {alpha}, Coefficients: {lasso_coefs[i]}")

print("\nRidge Regression Coefficients:")
for i, alpha in enumerate(alphas):
    print(f"Alpha: {alpha}, Coefficients: {ridge_coefs[i]}")
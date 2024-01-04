import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
data = np.loadtxt("X_data.csv", delimiter=',')
X = data[:, :10]
print(X.shape)
y = data[: , 10]
print(y.shape)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Define the parameter grid for the grid search
param_grid = {
    'colsample_bytree': [0.5, 0.7],
    'learning_rate': [1, 0.1, 0.01],
    'max_depth': [3, 4, 5],
    'reg_lambda': [1, 0.75, 0.5],
    'n_estimators': [500, 1000]
}

# Initialize XGBoost regressor
xgb_model = xgb.XGBRegressor()

# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best parameter combination
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Refit the model using the best parameters on all training instances
best_xgb_model = xgb.XGBRegressor(**best_params)
best_xgb_model.fit(X_train, y_train)

# Make predictions on the test set
test_preds = best_xgb_model.predict(X_test)

# Calculate RMSE and R-squared score on the test set
test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
test_r2 = r2_score(y_test, test_preds)

print(f"Test RMSE with best parameters: {test_rmse:.4f}")
print(f"Test R-squared with best parameters: {test_r2:.4f}")
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load your data
data = np.loadtxt("X_data.csv", delimiter=',')
X = data[:, :10]
print(X.shape)
y = data[: , 10]
print(y.shape)
# Split the data into training/val (80%) and hold-out test (20%) sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Split the data into training (90%) and hold-out validation (10%) sets
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1, random_state=42)

# Convert the data into DMatrix format
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test)

# Define the parameters for the XGBoost model
params = {
    'colsample_bytree': 0.5,
    'learning_rate': 0.1,
    'max_depth': 4,
    'reg_lambda': 1,
    'n_estimators': 500
}

# Monitor the training and validation RMSE during training
watchlist = [(dtrain, 'train'), (dval, 'eval')]
evals_result = {}
model = xgb.train(params, dtrain, num_boost_round=10000, evals=watchlist, evals_result=evals_result, early_stopping_rounds=10)

# Plot
train_rmse = evals_result['train']['rmse']
val_rmse = evals_result['eval']['rmse']
plt.figure(figsize=(10, 6))
plt.plot(train_rmse, label='Training RMSE')
plt.plot(val_rmse, label='Validation RMSE')
plt.xlabel('Boosting Iterations')
plt.ylabel('RMSE')
plt.legend()
plt.title('Training and Validation RMSE vs. Boosting Iterations')

# Evaluate 
test_preds = model.predict(dtest) # Run final model on test set
test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
test_r2 = r2_score(y_test, test_preds)
print(f"Test RMSE: {test_rmse:.4f}")
print(f"Test R-squared: {test_r2:.4f}")
plt.show()
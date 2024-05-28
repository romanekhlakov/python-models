from decision_tree_model import decision_tree_model
from evaluation_metrics import calculate_metrics
from gradient_boosting_model import gradient_boosting_model
from linear_regression_model import linear_regression_model
from random_forest_model import random_forest_model
from svr_model import svr_model
from termcolor import colored

file_path = '30_days.csv'

# # Decision Trees
# y_true, y_pred = decision_tree_model(file_path)
# metrics = calculate_metrics(y_true, y_pred)
# print(colored('Decision Tree:', 'green'))
# print(f"  MSE = {round(metrics['MSE'], 2)}")
# print(f"  MAE = {round(metrics['MAE'], 2)}")
# print(f"  RMSE = {round(metrics['RMSE'], 2)}")
# print(f"  R2 = {round(metrics['R2'], 2)}")
# print(f"  MAPE = {round(metrics['MAPE'], 2)}")
# print(f"  Accuracy = {round(metrics['Accuracy'], 2)}%\n")

# Gradient Boosting
y_true, y_pred = gradient_boosting_model(file_path)
metrics = calculate_metrics(y_true, y_pred)
print(colored('Gradient Boosting:', 'green'))
print(f"  MSE = {round(metrics['MSE'], 2)}")
print(f"  MAE = {round(metrics['MAE'], 2)}")
print(f"  RMSE = {round(metrics['RMSE'], 2)}")
print(f"  R2 = {round(metrics['R2'], 2)}")
print(f"  MAPE = {round(metrics['MAPE'], 2)}")
print(f"  Accuracy = {round(metrics['Accuracy'], 2)}%\n")

# # Linear Regression
# y_true, y_pred = linear_regression_model(file_path)
# metrics = calculate_metrics(y_true, y_pred)
# print(colored('Linear Regression:', 'green'))
# print(f"  MSE = {round(metrics['MSE'], 2)}")
# print(f"  MAE = {round(metrics['MAE'], 2)}")
# print(f"  RMSE = {round(metrics['RMSE'], 2)}")
# print(f"  R2 = {round(metrics['R2'], 2)}")
# print(f"  MAPE = {round(metrics['MAPE'], 2)}")
# print(f"  Accuracy = {round(metrics['Accuracy'], 2)}%\n")

# # Random Forest
# y_true, y_pred = random_forest_model(file_path)
# metrics = calculate_metrics(y_true, y_pred)
# print(colored('Random Forest:', 'green'))
# print(f"  MSE = {round(metrics['MSE'], 2)}")
# print(f"  MAE = {round(metrics['MAE'], 2)}")
# print(f"  RMSE = {round(metrics['RMSE'], 2)}")
# print(f"  R2 = {round(metrics['R2'], 2)}")
# print(f"  MAPE = {round(metrics['MAPE'], 2)}")
# print(f"  Accuracy = {round(metrics['Accuracy'], 2)}%\n")

# # Support Vector Regression
# y_true, y_pred = svr_model(file_path)
# metrics = calculate_metrics(y_true, y_pred)
# print(colored('SVR:', 'green'))
# print(f"  MSE = {round(metrics['MSE'], 2)}")
# print(f"  MAE = {round(metrics['MAE'], 2)}")
# print(f"  RMSE = {round(metrics['RMSE'], 2)}")
# print(f"  R2 = {round(metrics['R2'], 2)}")
# print(f"  MAPE = {round(metrics['MAPE'], 2)}")
# print(f"  Accuracy = {round(metrics['Accuracy'], 2)}%\n")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score

data = pd.read_csv("cleaned_train.csv")

# 分割数据
x = data.drop(columns=['monthly_rent'])
y = data['monthly_rent']
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.25, random_state=33)

# Regressor = RandomForestClassifier(n_estimators=300, random_state=42)
# Regressor = KNeighborsRegressor()
# Regressor = AdaBoostRegressor(base_estimator=None, n_estimators=50,
#                               learning_rate=1.0, loss='linear', random_state=None)
Regressor = XGBRegressor()
Regressor.fit(train_x, train_y)
pred_y = Regressor.predict(test_x)
mse = mean_squared_error(test_y, pred_y)
print("Predicted price:", pred_y)
print("R2 =", r2_score(test_y, pred_y))
print("MSE =", mse)
print("RMSE =", sqrt(round(mse, 2)))
print('-----------------------------------------------------')
# Create residual plot
residuals = [true - pred for true, pred in zip(test_y, pred_y)]
plt.figure(figsize=(8, 6), facecolor='lightblue')
ax = plt.gca()
ax.set_facecolor('lightblue')
plt.scatter(test_y, residuals, c='blue', label='Residuals', s=10)
plt.axhline(y=0, color='red', linestyle='--', lw=2)
plt.xlabel('True Values')
plt.ylabel('Residuals')
# plt.title('RandomForest')
# plt.title('KNN')
# plt.title('AdaBoost')
plt.title('XGBoost')
plt.legend()
plt.grid(True)
plt.show()
# # Draw feature importance chart
# feature_names = x.columns
# feature_importance = Regressor.feature_importances_
# plt.figure(figsize=(10, 6))
# plt.barh(feature_names, feature_importance)
# plt.xlabel('Feature Importance')
# plt.title('Random Forest Feature Importance')
# plt.gca().invert_yaxis()
# plt.yticks(rotation=45)
# plt.show()

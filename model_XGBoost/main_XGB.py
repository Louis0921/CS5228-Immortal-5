import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

train_data = pd.read_csv("cleaned_train.csv")
test_data = pd.read_csv("cleaned_test.csv")
# Feature normalization
x = RobustScaler().fit_transform(train_data.drop(['monthly_rent'], axis=1).values)
test_data = RobustScaler().fit_transform(test_data.values)

# Extract tags
y = train_data['monthly_rent'].values

# main_model
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.25, random_state=33)
xgboost_regressor = XGBRegressor(colsample_bytree=0.38, gamma=0.045,
                                 learning_rate=0.05, max_depth=3,
                                 min_child_weight=1.98, n_estimators=2600,
                                 reg_alpha=0.48, reg_lambda=0.85,
                                 subsample=0.73, random_state=1, nthread=-1)

# Validation set verification
xgboost_regressor.fit(train_x, train_y)
pred_y = xgboost_regressor.predict(test_x)
mse = mean_squared_error(test_y, pred_y)
print("XGB:")
print("Predicted price:", pred_y)
print("RMSE =", sqrt(round(mse, 2)))

# Get feature importance
importance = xgboost_regressor.feature_importances_

# Extract feature names and importance scores
feature_names = train_data.drop(['monthly_rent'], axis=1).columns

# Feature importance
plt.figure(figsize=(8, 15))
ax = plt.gca()
ax.set_facecolor('lightblue')
plt.barh(range(len(feature_names)), importance, tick_label=feature_names)
plt.xlabel('Feature Importance Score')
plt.title('XGBoost Feature Importance')
plt.gca().invert_yaxis()
plt.subplots_adjust(left=0.4)
plt.show()

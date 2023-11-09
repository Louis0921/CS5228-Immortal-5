import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import catboost as cb
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor

train_data = pd.read_csv("cleaned_train.csv")
test_data = pd.read_csv("cleaned_test.csv")

# Extract tags
cat_features = [0, 1, 2, 5]
x = train_data.drop(['monthly_rent'], axis=1).values
y = train_data['monthly_rent'].values

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.25, random_state=33)
params = {
    'iterations': 1060,
    'learning_rate': 0.03,
    'depth': 6,
    'loss_function': 'RMSE'
}
clf = CatBoostRegressor(**params, cat_features=cat_features)

# # Adjust parameters
# cv_params = {'depth': [8, 9, 10]}
#
# other_params = {
#     'iterations': 1500,
#     'learning_rate': 0.03,
#     # 'depth': 8,
#     'loss_function': 'RMSE'
# }
#
# cat_model_ = CatBoostRegressor(**other_params)
# cat_search = GridSearchCV(cat_model_,
#                           param_grid=cv_params,
#                           scoring='neg_mean_squared_error',
#                           n_jobs=-1,
#                           cv=5)
#
# cat_search.fit(train_x, train_y)
#
# means = cat_search.cv_results_['mean_test_score']
# params = cat_search.cv_results_['params']
#
# print(means)
# print(params)
# print(cat_search.best_params_)
# print(cat_search.best_score_)

# Validation set verification
clf.fit(train_x, train_y, verbose=100, plot=True)
pred_y = clf.predict(test_x)
mse = mean_squared_error(test_y, pred_y)
print("CB:")
print("Predicted price:", pred_y)
print("RMSE =", sqrt(round(mse, 2)))
print("R2 =", r2_score(test_y, pred_y))
print("MSE =", mse)


iterations = 1060
rmse_values = []

for i in range(1, iterations + 1):
    pred_y = clf.predict(test_x, ntree_end=i)
    rmse = sqrt(mean_squared_error(test_y, pred_y))
    rmse_values.append(rmse)

# RMSE vs number of iterations
plt.figure(figsize=(10, 6))
ax = plt.gca()
ax.set_facecolor('lightblue')
plt.plot(range(1, iterations + 1), rmse_values, linewidth=2, markersize=2)
plt.xlabel('Iterations')
plt.ylabel('RMSE')
plt.title('RMSE vs. Iterations')
plt.grid(True)
plt.show()

# Draw feature contribution
feature_importance = clf.get_feature_importance(cb.Pool(train_x, label=train_y, cat_features=cat_features), type="PredictionValuesChange")
feature_name = train_data.drop(['monthly_rent'], axis=1).columns
feature_importance = list(zip(feature_name, feature_importance))
feature_importance.sort(key=lambda x: x[1], reverse=False)
sorted_feature_names, sorted_feature_contributions = zip(*feature_importance)
plt.figure(figsize=(10, 6))
ax = plt.gca()
ax.set_facecolor('lightblue')
plt.barh(sorted_feature_names, sorted_feature_contributions)
plt.xlabel('Feature Contribution')
plt.title('Feature Contribution Rankings')
plt.show()

# Residual plot
residuals = [true - pred for true, pred in zip(test_y, pred_y)]
plt.figure(figsize=(8, 6), facecolor='lightblue')
ax = plt.gca()
ax.set_facecolor('lightblue')
plt.scatter(test_y, residuals, c='blue', label='Residuals', s=10)
plt.axhline(y=0, color='red', linestyle='--', lw=2)
plt.xlabel('True Values')
plt.ylabel('Residuals')
plt.title('Catboost')
plt.legend()
plt.grid(True)
plt.show()

# # Test set result prediction
# clf.fit(x, y)
# ans = clf.predict(test_data)
# result = pd.DataFrame(ans)
# result.index.name = 'Id'
# result.columns = ['Predicted']
# result.to_csv("result.csv")
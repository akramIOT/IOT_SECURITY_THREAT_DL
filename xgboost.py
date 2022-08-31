import numpy as np
import xgboost

# just a simple threshold
X = np.linspace(0, 1, 50).reshape(-1, 1)
y = np.array([0 if x < 0.5 else 1 for x in X])

# a non-sklearn model
model = xgboost.train({'objective': 'binary:logistic',
                       'max_depth': 2},
                      xgboost.DMatrix(X, label=y),
                      1000)

xgboost.plot_tree(model)
print('Labels:\n', y)
print('Predictions:\n', model.predict(xgboost.DMatrix(X)))

# the sklearn friendly model
sk_model = xgboost.XGBRegressor(objective='binary:logistic',
                                max_depth=2)
sk_model.fit(X, y)

print('Predictions (sklearn):\n', sk_model.predict(X))
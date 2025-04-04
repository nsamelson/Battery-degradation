import numpy as np
from sklearn.linear_model import LinearRegression




#TODO: change with RMSE or MSE
def compute_bic(y_true, y_pred, num_params):
    n = len(y_true)
    residuals = y_true - y_pred
    rss = np.sum(residuals**2)
    bic = num_params * np.log(n) + n * np.log(rss / n)
    return bic




# Suppose X, Y are your data
# model = LinearRegression()
# model.fit(X, Y)
# y_pred = model.predict(X)

# k = X.shape[1] + 1  # +1 for the intercept
# bic_value = compute_bic(Y, y_pred, k)
# print("BIC:", bic_value)
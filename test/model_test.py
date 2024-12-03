import pandas as pd
import cloudpickle
from sklearn.metrics import r2_score, mean_squared_error as MSE

df_test = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_test.csv')

y_test = df_test['selling_price']
X_test = df_test.drop('selling_price', axis=1)

with open('../model/ridge_pipeline.pkl', 'rb') as file:
    pipeline_from_cpickle = cloudpickle.load(file)

pred_test = pipeline_from_cpickle.predict(X_test)

assert MSE(y_test, pred_test) == 175755929580.7268
assert r2_score(y_test, pred_test) == 0.6942463490083228
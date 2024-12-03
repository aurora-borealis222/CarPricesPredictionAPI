import pandas as pd
import numpy as np
import random

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, MinMaxScaler

from category_encoders.leave_one_out import LeaveOneOutEncoder

from sklearn.metrics import r2_score, mean_squared_error as MSE

import cloudpickle

random.seed(42)
np.random.seed(42)

def preprocess_numeric_values(X: pd.DataFrame) -> pd.DataFrame:
    X['mileage'] = X['mileage'].str.split(expand=True)[0].astype(float)
    X.rename(columns={'mileage': 'mileage_kmpl_km/kg'}, inplace=True)

    X['engine'] = X['engine'].str.split(expand=True)[0].astype(float)
    X.rename(columns={'engine': 'engine_CC'}, inplace=True)

    # X[X['max_power'].str.strip() == 'bhp']['max_power'] = np.nan
    X.loc[X['max_power'].str.strip() == 'bhp', ['max_power']] = np.nan
    X['max_power'] = X['max_power'].str.split(expand=True)[0].astype(float)

    X.rename(columns={'max_power': 'max_power_bhp'}, inplace=True)

    return X.drop('torque', axis=1)

def convert_types(X: pd.DataFrame) -> pd.DataFrame:
    return X.astype({'engine_CC': 'int64', 'seats': 'int32'})

def remove_duplicates(X: pd.DataFrame) -> pd.DataFrame:
    feature_columns = ['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner',
                       'mileage', 'engine', 'max_power', 'seats', 'torque']

    X = X.drop_duplicates(subset=feature_columns)
    X.reset_index(drop=True, inplace=True)

    return X

def preprocess_name_column(X: pd.DataFrame) -> pd.DataFrame:
    X[['brand', 'model', 'variant']] = X['name'].str.split(n=2, expand=True)
    return X.drop('name', axis=1)


numerical_preprocessor = Pipeline(steps=[
    ("values_preprocessor", FunctionTransformer(preprocess_numeric_values)),
    ("nan_imputer", SimpleImputer(strategy='median')),
    ("zero_imputer", SimpleImputer(strategy='median', missing_values=0)),
    ("types_converter", FunctionTransformer(convert_types))
]).set_output(transform='pandas')

loo_features = ['brand', 'model', 'variant']

categorical_preprocessor2 = Pipeline(steps=[
    ("name_preprocessor", FunctionTransformer(preprocess_name_column)),
    ("loo_enc", LeaveOneOutEncoder(sigma=1., cols=loo_features))
])

numerical_features = ['mileage', 'engine', 'max_power', 'seats', 'torque']

categorical_features = ['fuel', 'seller_type', 'transmission', 'owner', 'seats']

all_features = ['mileage', 'engine', 'max_power', 'seats', 'torque', 'name', 'fuel', 'seller_type', 'transmission', 'owner', 'seats']

numerical_transformer = ColumnTransformer(transformers=[
    ("numerical", numerical_preprocessor, numerical_features)],
    remainder='passthrough',
    verbose_feature_names_out=False)

categorical_transformer1 = ColumnTransformer(transformers=[
    ("loo_enc", LeaveOneOutEncoder(sigma=1., cols=loo_features), loo_features)],
    remainder='passthrough',
    verbose_feature_names_out=False)

categorical_transformer2 = ColumnTransformer(transformers=[
    ("onehot", OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), categorical_features)],
    remainder='passthrough',
    verbose_feature_names_out=False)

preprocessor = Pipeline(steps=[
    ("numerical_transformer", numerical_transformer),
    ("name_preprocessor", FunctionTransformer(preprocess_name_column)),
    ("categorical_transformer1", categorical_transformer1),
    ("categorical_transformer2", categorical_transformer2),
    ("scaler", MinMaxScaler()),
]).set_output(transform='pandas')

ridge_pipeline = Pipeline(
    steps=[("preprocessor", preprocessor),
           ("lr_model", Ridge(alpha=2.76))])

df_train = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv')
df_test = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_test.csv')

df_train = remove_duplicates(df_train)
print(df_train.shape)

y_train = df_train['selling_price']
X_train = df_train.drop('selling_price', axis=1)

y_test = df_test['selling_price']
X_test = df_test.drop('selling_price', axis=1)

print('Pipeline start')

ridge_pipeline.fit(X_train, y_train)

pred_test = ridge_pipeline.predict(X_test)
print(pred_test)

print(MSE(y_test, pred_test))

print(r2_score(y_test, pred_test))


with open('../model/ridge_pipeline.pkl', 'wb') as file:
    cloudpickle.dump(ridge_pipeline, file)

with open('../model/ridge_pipeline.pkl', 'rb') as file:
    pipeline_from_cpickle = cloudpickle.load(file)

print(all(ridge_pipeline.predict(X_test) == pipeline_from_cpickle.predict(X_test)))

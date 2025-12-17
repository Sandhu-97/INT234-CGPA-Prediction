import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


data = pd.read_csv('cleaned.csv')


print('=====EXPLORATORY DATA ANALYSIS=====')
print('Shape:', data.shape)
print('Preview:\n', data.head())
print("Info:\n", data.info())
print("Describe:\n", data.describe())


print('Missing values check:')
print(data.isnull().sum())


scaler = StandardScaler()

scaled_cols = [
    "study_hours", "sleep_hours", "attendance",
    "screen_time", "activities", "stress", "prev_gpa"
]

data_scaled = data.copy()
data_scaled[scaled_cols] = scaler.fit_transform(data_scaled[scaled_cols])
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, precision_recall_fscore_support
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

df = pd. read_csv("C:/Users/kavya/OneDrive/Desktop/Spring2023_890/LoanFHLBData/original/LoanData_preprocessed.csv")


target=["BoRace"]
X=df.loc[:,~df.columns.isin(target)]
y=df.loc[:,df.columns.isin(target)]
feature_names=X.columns

scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
scaler = MinMaxScaler(feature_range=(0, 1))

# Create a random forest classifier model
model = RandomForestClassifier()

# Use SelectFromModel to select the top two features based on feature importance
selector = SelectFromModel(model, max_features=2)
X_new = selector.fit_transform(X, y)


selector = SelectKBest(chi2, k=10)
X_new = selector.fit_transform(X_scaled_df, y)
# print selected features
selected_features=selector.get_support(indices=True)
print([feature_names[i] for i in selected_features])


import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, precision_recall_fscore_support, confusion_matrix
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# df = pd. read_csv("C:/Users/kavya/OneDrive/Desktop/Spring2023_890/LoanFHLBData/original/LoanData.csv")
# df.drop(['LoanNumber', 'AssignedID','Coop','Program','AcquDate','MortDate','Bed1','Bed2','Bed3','Bed4','Aff1','Aff2','Aff3','Aff4','Occup','NumUnits','RentUt1','RentUt2','RentUt3','Product','RentUt4','Rent1','Rent2','Rent3','Rent4','FeatureID','SellType','Seller','SellCity','SellSt','CICA','LienStatus','FedFinStbltyPlan','GSEREO','FHFBID','Geog','SpcHsgGoals','AcqTyp','PrepayP'], axis=1, inplace=True)

df = pd. read_csv("C:/Users/isayal/Desktop/LoanData.csv")

dk = pd.DataFrame()
dk = df[['FHLBankID','FIPSStateCode']]
fill_values = dict(dk.groupby('FIPSStateCode')['FHLBankID'].apply(lambda x: x.dropna().mode()[0]))
dk['FHLBankID'] = dk.apply(lambda x: fill_values[x['FIPSStateCode']] if pd.isnull(x['FHLBankID']) else x['FHLBankID'], axis=1)
df['FHLBankID'] = dk['FHLBankID']

label_encoder = LabelEncoder()
# df[['FHLBankID', 'PropType']] = df[['FHLBankID', 'PropType']].apply(label_encoder.fit_transform)
fhlbankid_encoder = LabelEncoder()
df['FHLBankID'] = fhlbankid_encoder.fit_transform(df['FHLBankID'])

# create label encoder for PropType column
proptype_encoder = LabelEncoder()
df['PropType'] = proptype_encoder.fit_transform(df['PropType'])
# df['PrepayP'] = pd.to_numeric(pd.to_datetime(df['PrepayP']).astype(int))
# df['PrepayP'] = pd.to_datetime(df['PrepayP'], errors='coerce').fillna(0)# df['PrepayP'] = pd.to_numeric(pd.to_datetime(df['PrepayP'], errors='coerce').fillna(0))
# def date_to_numeric(date):
#     if pd.isna(date):
#         return 0
#     else:
#         try:
#             date_obj = pd.to_datetime(date)
#             if date_obj.year < 1970 or date_obj.year > 2100:
#                 return 0
#             else:
#                 return int(date_obj.timestamp())
#         except ValueError:
#             return 0
# df['PrepayP'] = df['PrepayP'].apply(date_to_numeric)
# Convert the date column to an integer format
# df['FHLBankID'] = label_encoder.fit_transform(df['FHLBankID','PropType'])

cols_with_missing = df.columns[df.isnull().any()].tolist()
testdata = df[df.isnull().any(axis=1)]
df.dropna(inplace=True)
target=["Tractrat","IncRat"]
X_train=df.loc[:,~df.columns.isin(target)]
y_train=df.loc[:,df.columns.isin(target)]
X_test=testdata.loc[:,~df.columns.isin(target)]
y_test=testdata.loc[:,df.columns.isin(target)]
clf = MultiOutputRegressor(RandomForestRegressor(max_depth=2, random_state=0))
clf.fit(X_train, y_train)
y_train.isnull().sum()
y_pred = clf.predict(X_test)
y_test=y_pred
X_test[target]=y_pred
X_train[target]=y_train
df=X_train.append(X_test)

# df[['FHLBankID', 'PropType']] = df[['FHLBankID', 'PropType']].apply(label_encoder.inverse_transform)
df.isnull().any()
target=["Year"]
X=df.loc[:,~df.columns.isin(target)]
y=df.loc[:,df.columns.isin(target)]

# Define the class labels
class_labels = {2009: 0, 2010: 1, 2011: 2, 2012: 3, 2013: 4, 2014: 5, 2015: 6, 2016: 7, 2017: 8, 2018: 9, 2019: 10, 2020: 11, 2021: 12}

# Map the year values to class labels
y['Year'] = y['Year'].map(class_labels)
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.30, random_state=0)

# train= xgb.DMatrix(X_train,label=y_train)
# test= xgb.DMatrix(X_test,label=y_test)
#
#param={
#     'max_depth' : 4,
#     'eta': 0.3,
#     'objective':'multi:softmax',
#     'num_class':13# }
# epochs=10# model=xgb.train(param,train,epochs)
# df.info()
# label_values = (df['Year'].unique())
xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=13, learning_rate=0.1, max_depth=6, n_estimators=100, colsample_bytree=0.8)

# Train the XGBoost model on the training data
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
accuracy = accuracy_score(y_test, y_pred)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1_score)
print("Accuracy:", accuracy)

# Assuming y_pred and y_true are the predicted and true labels respectively

cm = confusion_matrix(y_test, y_pred)
classes= (df['Year'].unique())
type(cm)
# Print the confusion matrix
print(cm)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.annotate(str(cm[i][j]), xy=(j, i), ha='center', va='center')
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.show()

importances = xgb_model.feature_importances_
feature_importances = pd.DataFrame({'feature': X.columns, 'importance': importances})
# sort features by importance in descending order
feature_importances = feature_importances.sort_values(by='importance', ascending=False)

# plot feature importances
plt.barh(feature_importances['feature'], feature_importances['importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances')
plt.show()

similarity_matrix = cm / cm.sum(axis=1, keepdims=True)
plt.figure(figsize=(8, 6))
sns.heatmap(similarity_matrix, annot=True, cmap='Blues')
plt.title('Similarity Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
df['FHLBankID'] = fhlbankid_encoder.inverse_transform(df['FHLBankID'])
df['PropType'] = proptype_encoder.inverse_transform(df['PropType'])

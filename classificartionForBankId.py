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


df = pd. read_csv("C:/Users/isayal/Desktop/LoanData.csv")

# Code to drop unwanted columns
df.drop(['LoanNumber', 'AssignedID','Coop','Program','AcquDate','MortDate','Bed1','Bed2','Bed3','Bed4','Aff1','Aff2','Aff3','Aff4','Occup','NumUnits','RentUt1','RentUt2','RentUt3','Product','RentUt4','Rent1','Rent2','Rent3','Rent4','FeatureID','SellType','Seller','SellCity','SellSt','CICA','LienStatus','FedFinStbltyPlan','GSEREO','FHFBID','Geog','SpcHsgGoals','AcqTyp','PrepayP'], axis=1, inplace=True)


# code to fill missing values for FHLBankID Column based on FIPSStateCode
dk = pd.DataFrame()
dk = df[['FHLBankID','FIPSStateCode']]
fill_values = dict(dk.groupby('FIPSStateCode')['FHLBankID'].apply(lambda x: x.dropna().mode()[0]))
dk['FHLBankID'] = dk.apply(lambda x: fill_values[x['FIPSStateCode']] if pd.isnull(x['FHLBankID']) else x['FHLBankID'], axis=1)
df['FHLBankID'] = dk['FHLBankID']

label_encoder = LabelEncoder()
fhlbankid_encoder = LabelEncoder()
df['FHLBankID'] = fhlbankid_encoder.fit_transform(df['FHLBankID'])

# create label encoder for PropType column
proptype_encoder = LabelEncoder()
df['PropType'] = proptype_encoder.fit_transform(df['PropType'])

cols_with_missing = df.columns[df.isnull().any()].tolist()
testdata = df[df.isnull().any(axis=1)]
df.dropna(inplace=True)
target=["Tractrat", "IncRat"]
X_train = df.loc[:, ~df.columns.isin(target)]
y_train = df.loc[:, df.columns.isin(target)]
X_test = testdata.loc[:, ~df.columns.isin(target)]
y_test = testdata.loc[:, df.columns.isin(target)]
clf = MultiOutputRegressor(RandomForestRegressor(max_depth=2, random_state=0))
clf.fit(X_train, y_train)
y_train.isnull().sum()
y_pred = clf.predict(X_test)
y_test = y_pred
X_test[target] = y_pred
X_train[target] = y_train
df = X_train.append(X_test)

# code for predictions
df["FHLBankID"].unique()
dk["FHLBankID"].unique()

df.isnull().any()
# removing columns FIPSStateCode and FIPSCountyCode

df.drop(['FIPSStateCode', 'FIPSCountyCode'], axis=1, inplace=True)
target = ["FHLBankID"]
X = df.loc[:, ~df.columns.isin(target)]
y = df.loc[:, df.columns.isin(target)]

# Define the class labels
class_labels = {3: 0, 6: 1, 9: 2, 8: 3, 2: 4, 10: 5, 5: 6, 1: 7, 7: 8, 4: 9, 0: 10}
#class_labels = {3, 6, 9, 8, 2, 10, 5, 1, 7, 4, 0}

# Map the year values to class labels
y['FHLBankID'] = y['FHLBankID'].map(class_labels)
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
classes = (df['FHLBankID'].unique())
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

# correlation code
df = pd.DataFrame(X_test, columns=X_test.columns)
df["predicted_value"] = y_pred
corr_matrix = df.corr()
print(corr_matrix)
sns.set(font_scale=1)
sns.set(rc={"figure.figsize":(12,8)})
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')

# alternative code for correlation matrix
df = pd.DataFrame(X_test, columns=X_test.columns)
#df["predicted_value"] = y_pred
df["FHLBankID"] = y_pred
# Calculate the correlation matrix
corr_matrix = df.corr()

# Plot the correlation matrix as a heatmap
plt.imshow(corr_matrix, cmap='hot', interpolation='nearest')
plt.xticks(range(len(df.columns)), df.columns, rotation=90)
plt.yticks(range(len(df.columns)), df.columns)
plt.colorbar()
plt.show()








""" 
#correlatio code for a case where we can select features instead of taking the entire dataset
import pandas as pd
import seaborn as sns
import numpy as np

# create a sample dataframe with 60 features
df = pd.DataFrame(np.random.rand(100, 60), columns=[f"Feature_{i}" for i in range(60)])

# select a subset of features
selected_features = ["Feature_1", "Feature_3", "Feature_5", "Feature_7", "Feature_9", "Feature_11", "Feature_13", "Feature_15"]

# calculate the correlation matrix for the selected features
corr_matrix = df[selected_features].corr()

# increase the size of the heatmap figure
sns.set(font_scale=1.2)
sns.set(rc={"figure.figsize":(12,8)})

# plot the correlation matrix as a heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', center=0, vmin=-1, vmax=1, xticklabels=True, yticklabels=True)
"""



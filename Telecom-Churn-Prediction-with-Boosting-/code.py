# --------------
import pandas as pd
from sklearn.model_selection import train_test_split
#path - Path of file 

# Code starts here
df = pd.read_csv(path)
print(df.shape)
X = df.drop(['customerID','Churn'],axis=1)
print(X.shape)
y = df['Churn']
print(y.shape)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)




# --------------
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Code starts here
X_train['TotalCharges'] = X_train['TotalCharges'].replace(r'^\s*$', np.nan, regex=True).astype(float)
X_test['TotalCharges'] = X_test['TotalCharges'].replace(r'^\s*$', np.nan, regex=True).astype(float)
X_train['TotalCharges'] = X_train['TotalCharges'].fillna(X_train['TotalCharges'].mean())
X_test['TotalCharges'] = X_test['TotalCharges'].fillna(X_test['TotalCharges'].mean())
print(X_train.isnull().sum())
print(X_test.isnull().sum())

cat_cols = X_train.select_dtypes(include='O').columns.tolist()
#Label encoding train data
for x in cat_cols:
    le = LabelEncoder()
    X_train[x] = le.fit_transform(X_train[x])
    X_test[x] = le.fit_transform(X_test[x])
y_train = y_train.replace({'No':0, 'Yes':1})
y_test = y_test.replace({'No':0, 'Yes':1})


# --------------
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

# Code starts here
print(X_train.head())
print(X_test.head())
print(y_train.head())
print(y_test.head())
ada_model = AdaBoostClassifier(random_state=0)
ada_model.fit(X_train,y_train)
y_pred = ada_model.predict(X_test)
ada_score = accuracy_score(y_test,y_pred)
print("AdaBoost Score : {}".format(ada_score))
ada_cm = confusion_matrix(y_test,y_pred)
print("Confusion Matrix For AdaBoost Classifier: {}".format(ada_cm))
ada_cr = classification_report(y_test,y_pred)
print("Classification Report For AdaBoost Classifier: {}".format(ada_cr))


# --------------
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

#Parameter list
parameters={'learning_rate':[0.1,0.15,0.2,0.25,0.3],
            'max_depth':range(1,3)}

# Code starts here
xgb_model = XGBClassifier(random_state=0)
xgb_model.fit(X_train,y_train)
y_pred = xgb_model.predict(X_test)
xgb_score = accuracy_score(y_pred,y_test)
print("XGB Score: {}".format(xgb_score))
xgb_cm = confusion_matrix(y_test,y_pred)
print("Confusion Matrix For XGB Classifier: {}".format(xgb_cm))
xgb_cr = classification_report(y_test,y_pred)
print("Classification Report For XGB Classifier: {}".format(xgb_cr))

clf_model = GridSearchCV(estimator = xgb_model,param_grid = parameters)
clf_model.fit(X_train,y_train)
y_pred = clf_model.predict(X_test)
clf_score = accuracy_score(y_pred,y_test)
print("GridSearchCV Score: {}".format(clf_score))
clf_cm = confusion_matrix(y_test,y_pred)
print("Confusion Matrix For GridSearchCV Classifier: {}".format(clf_cm))
clf_cr = classification_report(y_test,y_pred)
print("Classification Report For GridSearchCV Classifier: {}".format(clf_cr))



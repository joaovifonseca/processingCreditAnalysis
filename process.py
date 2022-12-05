import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report, accuracy_score,precision_score,recall_score,f1_score,matthews_corrcoef
from sklearn.metrics import confusion_matrix
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


cust_df=pd.read_csv(r"./customer_data.csv")
paymt_df=pd.read_csv(r"./payment_data.csv")

print(cust_df.head())
print("")
print(cust_df.info())
print("")
print(cust_df.shape)
print("")
print(cust_df.describe())
print("")
# Checking the missing values values for customer data
print(cust_df.isnull().sum())
#print("**************payment************************")
#print(paymt_df.isnull().sum())
cust_df["label"].value_counts()

low_risk=cust_df[cust_df["label"]==0]
high_risk=cust_df[cust_df["label"]==1]
frac=len(high_risk)/float(len(low_risk))
frac



#visualising the "label" column 
plt.pie(cust_df["label"].value_counts(),labels = ["Not Risk","Risk"],colors = ["g","r"],shadow = True)
plt.legend(title ="Credit Risk")
plt.show() 
#This shows that the dataset is Imbalanced


#Correlation Matrix

correlation_matrix = cust_df.corr()
fig = plt.figure(figsize=(12,9))
sns.heatmap(correlation_matrix,annot=True,square=True, linewidths=.5,cmap=plt.cm.Reds)
plt.show()




y=cust_df["label"]
x=cust_df.copy()
x.drop(columns=["label"],inplace=True)
x.head()


#handeling missing data (replace all missing values with mean value of that column)
x["fea_2"].fillna(x["fea_2"].mean(),inplace=True)
print(x.isnull().sum())
print(x.shape,y.shape)







os =  RandomOverSampler(0.7)
X_train_res, y_train_res = os.fit_resample(x, y)
print(" New 'x' has",X_train_res.shape,"        New 'Y' has",y_train_res.shape)
print('Original dataset shape {}'.format(Counter(y)))
print('Resampled dataset shape {}'.format(Counter(y_train_res)))








#np.random.seed(123)
X_train, X_test, y_train, y_test = train_test_split(X_train_res, y_train_res, train_size = 0.70, test_size = 0.30, random_state = 1)
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)




forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(X_train, y_train)
# predictions
y_pred =forest_model.predict(X_test)
print(y_pred)






#printing the confusion matrix
#n_outliers = len(high_risk)
n_errors = (y_pred != y_test).sum()
LABELS = ['GOOD', 'BAD']
conf_matrix = confusion_matrix(y_test, y_pred.round())
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()




# Run classification metrics
plt.figure(figsize=(9, 7))
print('{}: {}'.format("Random Forest", n_errors))
print(accuracy_score(y_test, y_pred.round()))
print(classification_report(y_test, y_pred.round()))
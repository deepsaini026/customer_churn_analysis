import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report,confusion_matrix
import joblib
import openpyxl
from openpyxl import load_workbook

file_path=r"C:\Users\HP\Downloads\prediction_data.xlsx"
sheet_name='vw_churnData'
data=pd.read_excel(file_path,sheet_name)

print(data.head())
data=data.drop(['Customer_ID','Churn_Category','Churn_Reason'],axis=1)

columns_to_encode=['Gender','Married','State','Value_Deal','Phone_Service','Multiple_Lines','Internet_Service',
                   'Internet_Type','Online_Security','Online_Backup','Device_Protection_Plan',
                   'Premium_Support','Streaming_TV','Streaming_Movies','Streaming_Music','Unlimited_Data',
                   'Contract','Paperless_Billing','Payment_Method']
 
label_encoders={}
for column in columns_to_encode:
    label_encoders[column]=LabelEncoder()
    data[column]=label_encoders[column].fit_transform(data[column])

data['Customer_Status']= data['Customer_Status'].map({'Stayed':0,'Churned':1})

#Split data into feature and target
x=data.drop('Customer_Status',axis=1)
y=data['Customer_Status']

# split data into training and testing sets
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#train random forest model
# initialize the random forest classifier
rf_model=RandomForestClassifier(n_estimators=100,random_state=42)
rf_model.fit(x_train,y_train)

rf_model.predict(x_test)

#evaluate model
#make prediction
y_pred=rf_model.predict(x_test)

#evaluate the model
print('confusion matrix :')
print(confusion_matrix(y_test,y_pred))
print('\n classification report :')
print(classification_report(y_test,y_pred))

#feature selection using feature importance
importances=rf_model.feature_importances_
indices=np.argsort(importances)[::-1]

#plot the feature importances
plt.figure(figsize=(15,6))
sns.barplot(x=importances[indices],y=x.columns[indices])
plt.title('feature importances')
plt.xlabel('relative importance')
plt.ylabel('feature names')
#plt.show()

#model for vw_joindata
file_pathh=r"C:\Users\HP\Downloads\prediction_data.xlsx"
sheet_namee='joindate'
new_data=pd.read_excel(file_pathh,sheet_namee)

#retain the original dataframe to preserve unencoded columns 
orignal_data=new_data.copy()

#retain customer_id column
customer_ids=new_data['Customer_ID']

new_data=new_data.drop(['Customer_ID','Customer_Status','Churn_Category','Churn_Reason'],axis=1)

#encode categorical variables using the saved label encoders
for column in new_data.select_dtypes(include=['object']).columns:
    new_data[column]=label_encoders[column].transform(new_data[column])

#make prediction
new_predictions=rf_model.predict(new_data)

#add prediction to the original dataFrame
orignal_data['Customer_Status_Predicted']=new_predictions

#filter the DataFrame to include only recprds predicted as 'Churned'
original_data=orignal_data[orignal_data['Customer_Status_Predicted']==1]

#save results 
original_data.to_excel(r"C:\Users\HP\Downloads\prediction_data.xlsx",engine='openpyxl')

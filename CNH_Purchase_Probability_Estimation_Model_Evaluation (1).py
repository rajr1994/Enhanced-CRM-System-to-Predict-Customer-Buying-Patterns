# # Prediction of Customer Buying Patterns (+ Evaluation)


# ## Importing libraries
import zipfile
import pandas as pd
import os
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from datetime import datetime
import time

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier, PassiveAggressiveClassifier, SGDClassifier, Perceptron, LogisticRegressionCV, RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis    
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, auc, balanced_accuracy_score, cohen_kappa_score, log_loss, roc_auc_score, precision_score, recall_score
from sklearn import preprocessing

from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import calibration_curve
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample


from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals())

import pymysql as mysql
import pandas as pd
from sqlalchemy import create_engine

import statistics
from getpass import getpass
from sshtunnel import SSHTunnelForwarder
from tqdm import tqdm_notebook
from tqdm import tqdm

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms

# ## Uploading data from database
#Establishing connection to Datamine
#PLEASE UPDATE THE inputs for HOST, DATABASE, USER and PASSWORD

try:
    conn = mysql.connect(host='hostname', port=portno, database='dbname',user='dbuser', password='dbpassword')

except:
    conn.close()
    
#Pushing data (pandas dataframe) into a new/existing table in MySQLServer
engine = create_engine("mysql+pymysql://dbuser:dbpassword@datamine.rcac.purdue.edu/dbname".format
                       (user="dbuser",pw="dbpassword",db="dbname"))

# Loading the dataset
purchases = pd.read_sql('select * from ppa_purchases_2009_2019', conn) #Time Consuming large Dataset
product_grouping = pd.read_sql('select * from ProductGrouping', conn)
webactivity = pd.read_sql('select * from WebActivity', conn)


# ## Data Filtering
# USER INPUT: Change 'x' to select frequency filter

groups=purchases['dmkey'].value_counts()
groups=pd.DataFrame({'dmkey':groups.index, 'count':groups.values})

x=4   # Define the min. number of purchases by customer to qualify for modeling 

print("Selecting the customers with over "+ str(x-1) + " purchases...")

groups=groups[groups['count']>=x]
tags = groups['dmkey'].unique()

print("Selecting "+str(len(tags))+" customers with over "+ str(x-1) + " purchases...")
purchases = purchases[purchases['dmkey'].isin(tags)]

# USER INPUT: Change y to select value filter
dmkeyVal=purchases.groupby('dmkey')['equipment_value'].sum()
dmkeyVal=pd.DataFrame({'dmkey':dmkeyVal.index, 'eqpVal':dmkeyVal.values})
y=1000000   # Define the min. market value of the customer to qualify for modeling (min. 100,000 p.a. recommended)
print("Selecting the customers with over $"+ str(y) + " in net purchase value...")
dmkeyVal=dmkeyVal[dmkeyVal['eqpVal']>=y]
tags2 = dmkeyVal['dmkey'].unique()
print("Selecting "+str(len(tags2))+" customers with over $"+ str(y) + " in net purchase value...")
purchases = purchases[purchases['dmkey'].isin(tags2)]


# ## Data Cleaning and Pre-Processing
#Joining ppa_purchases with product_grouping on SubCatID
data_agg=pd.merge(purchases, product_grouping, how='left',left_on='subcatID',right_on='Subcat')
data_agg1 = data_agg[['ppa_purchase_id','dmkey','equipment_value', 'brand_affiliation', 'newused', 'uccstatus', 'PPM', 'uccdate','state','dealernumber']]

#Adding a quarter column
data_agg1['uccdate'] = pd.to_datetime(data_agg1['uccdate'].astype(str), format='%m/%d/%Y')
data_agg1['quarter'] = pd.PeriodIndex(data_agg1.uccdate, freq='Q')

#Creating a feature to measure company loyalty
data_agg1['company_affiliation'] = np.where(data_agg1['brand_affiliation'] == 'company', 'company', 'Other')


# Removing customers who purchased only "Others"
#Filtering out customers who have only purchased PPM marked as 'other' - insignificant by itself
data_agg2= data_agg1.loc[data_agg1['PPM'] != 'other']
temp_dmkey=data_agg2['dmkey'].unique()
data_agg2 = data_agg1[data_agg1['dmkey'].isin(temp_dmkey)].copy()

#Taking only the columns required for feature engineering
data_agg2 = data_agg2[['ppa_purchase_id','dmkey', 'quarter', 'equipment_value', 'company_affiliation', 'newused', 'uccstatus', 'PPM', 'uccdate','state']]
data_agg2 = data_agg2.fillna(0)


# ## Feature Engineering - Creating RFM variables specific to Customer
#Sorting by customer, quarter and date of transaction
data_agg2 = data_agg2.sort_values(by=['dmkey','quarter', 'uccdate'])
data_agg2 = data_agg2.reset_index(drop=True)
mem_id = data_agg2['dmkey'].unique().tolist()
count_list = []
quarter_average = []
equipment_value = []
new_used = []
ucc_status = []
company_aff = []
dmkey_average_days = []

# LOOP TO CREATE CUSTOMER-SPECIFC RFM FEATURES
# for ppm_id in tqdm_notebook(mem_id):  #alternative loop to track progress/time taken
for member_id in mem_id:

    temp_df = data_agg2.loc[(data_agg2['dmkey'] == member_id)].reset_index(drop=True)
    between_days = []
    
    for index, row in temp_df.iterrows():
        
        count = temp_df.loc[(temp_df['quarter'] <= row['quarter']) & (temp_df['quarter'] >= (row['quarter'] - 20))].shape[0]
        count_list.append(count)
        
        quarter_average.append(temp_df.loc[(temp_df['quarter'] == row['quarter'])].shape[0])
        equipment_value.append(temp_df.loc[(temp_df['quarter'] <= row['quarter']) & (temp_df['quarter'] >= (row['quarter'] - 20))]['equipment_value'].sum())
        
        new = temp_df.loc[(temp_df['quarter'] <= row['quarter']) & (temp_df['quarter'] >= (row['quarter'] - 20)) & (temp_df['newused'] == "NEW")].shape[0]
        new_used.append(new/count)
        
        sale = temp_df.loc[(temp_df['quarter'] <= row['quarter']) & (temp_df['quarter'] >= (row['quarter'] - 20)) & (temp_df['uccstatus'] == "SALE")].shape[0]
        ucc_status.append(sale/count)
        
        company = temp_df.loc[(temp_df['quarter'] <= row['quarter']) & (temp_df['quarter'] >= (row['quarter'] - 20)) & (temp_df['company_affiliation'] == "company")].shape[0]
        company_aff.append(company/count)
        

        if index == 0:
            ucc_date_1 = row['uccdate']
            between_days.append(0)
        else:
            temp_days = row['uccdate'] - ucc_date_1
            between_days.append(temp_days.days)
            ucc_date_1 = row['uccdate']
            
    temp_df['dmkey_between_days'] = between_days
    
    for index, row in temp_df.iterrows():

        temp_df_2 = temp_df.loc[(temp_df['quarter'] <= row['quarter']) & (temp_df['quarter'] >= (row['quarter'] -20))]
        sum_of_days = temp_df_2['dmkey_between_days'].sum()
        if temp_df_2.shape[0] > 1:
            dmkey_average_days.append(sum_of_days/(temp_df_2.shape[0] -1 ))
        else:
            dmkey_average_days.append(0)

print('completed! \n now assigning values to dataframe...')
            
data_agg2['count_list'] = count_list
data_agg2['quarter_average'] = quarter_average
data_agg2['equipment_value_agg'] = equipment_value
data_agg2['new_used'] = new_used
data_agg2['ucc_status'] = ucc_status
data_agg2['company_aff'] = company_aff
data_agg2['dmkey_average_days'] = dmkey_average_days

dmkey_sd_average_days = []

for member_id in mem_id:
    temp_df = data_agg2.loc[(data_agg2['dmkey'] == member_id)].reset_index(drop=True)
    
    for index, row in temp_df.iterrows():
        sd_list = temp_df.loc[(temp_df['quarter'] <= row['quarter']) & (temp_df['quarter'] >= (row['quarter'] -20))]['dmkey_average_days']
        if len(sd_list[1:]) <= 1:
            dmkey_sd_average_days.append(0)
        else:
            dmkey_sd_average_days.append(statistics.stdev(sd_list[1:]))
    

data_agg2['dmkey_sd_average_days'] = dmkey_sd_average_days


# ## Feature Engineering - Creating RFM variables specifc to PPM
data_agg2 = data_agg2.sort_values(by=['PPM','quarter', 'uccdate' ])
data_agg2 = data_agg2.reset_index(drop=True)
PPM_id_list  = data_agg2['PPM'].unique().tolist()
df2 = data_agg2

average_days = []
total_transaction_ppm = []
sd_average_days = []
diff_days = []

# for ppm_id in tqdm_notebook(PPM_id_list):  #alternative loop to track progress/time taken
for ppm_id in PPM_id_list:

    between_days = []
    temp_df = df2.loc[(df2['PPM'] == ppm_id)].reset_index(drop=True)
    rows = temp_df.shape[0]
    days = 0    
    
    for index, row in temp_df.iterrows():
        if index == 0:
            ucc_date_1 = row['uccdate']
            between_days.append(0)
        else:
            temp_days = row['uccdate'] - ucc_date_1
            between_days.append(temp_days.days)
            ucc_date_1 = row['uccdate']
            
    temp_df['PPM_between_days'] = between_days
    

    for index, row in temp_df.iterrows():
        temp_df_2 = temp_df.loc[(temp_df['quarter'] <= row['quarter']) & (temp_df['quarter'] >= (row['quarter'] -20))]
        sum_of_days = temp_df_2['PPM_between_days'].sum()
        if temp_df_2.shape[0] > 1:
            average_days.append(sum_of_days/(temp_df_2.shape[0] -1 ))
        else:
            average_days.append(0)
        total_transaction_ppm.append(temp_df_2.shape[0])

df2['average_days'] = average_days
df2['total_transaction_PPM'] = total_transaction_ppm

# # Sd_average_days this feature was removed due to high computational time

# sd_average_days = []

# for ppm_id in tqdm_notebook(PPM_id_list):
#     temp_df = df2.loc[(df2['PPM'] == ppm_id)].reset_index(drop=True)
    
#     for index, row in temp_df.iterrows():
#         if ppm_id!='other':
#             sd_list = temp_df.loc[(temp_df['quarter'] <= row['quarter']) & (temp_df['quarter'] >= (row['quarter'] -20))]['average_days']
#             if len(sd_list[1:]) <= 1:
#                 sd_average_days.append(0)
#             else:
#                 sd_average_days.append(statistics.stdev(sd_list[1:]))
#         else:
#             sd_average_days.append(0.03) # Appended the mean SD to improve computational speed

# df2['sd_average_days'] = sd_average_days


# ## Generating Y Variable
df2 = df2.sort_values(by=['dmkey', 'quarter','PPM', 'uccdate'])
model = df2[['dmkey', 'PPM', 'quarter','uccdate']]
model = model.reset_index(drop=True)


#Purchase in 3 months
Y_3M = []
quarter_diff = 1
x = 0
start=time.time() #To track the time involved in this process

# for index, row in tqdm(model.iterrows()): alternative loop to track progress/time taken
for index, row in model.iterrows():
    
    #creating 3 month y variable
    size_3m = model.loc[(model['dmkey'] == row['dmkey']) & (model['PPM'] == row['PPM']) &(model['quarter'] == (row['quarter'] +quarter_diff))].shape[0]    
    if size_3m == 0:
        Y_3M.append(0)
    else:
        Y_3M.append(1)

end=time.time() #To track the time involved in this process

print((end-start)/3600) #To print the time taken for this process (in hours)


#Purchase in 6 months
Y_6M = []
quarter_diff = 2

for index, row in model.iterrows():

    #creating 6 month y variable
    size_6m = model.loc[(model['dmkey'] == row['dmkey']) & (model['PPM'] == row['PPM']) &(model['quarter'] == (row['quarter'] +quarter_diff))].shape[0]    
    if size_6m == 0:
        Y_6M.append(0)
    else:
        Y_6M.append(1)
        

#Purchase in 9 months
Y_9M = []
quarter_diff = 3

for index, row in model.iterrows():

    #creating 9 month y variable
    size_9m = model.loc[(model['dmkey'] == row['dmkey']) & (model['PPM'] == row['PPM']) &(model['quarter'] == (row['quarter'] +quarter_diff))].shape[0]    
    if size_9m == 0:
        Y_9M.append(0)
    else:
        Y_9M.append(1)


model['Y_3M'] = Y_3M
model['Y_3M_6M'] = Y_6M
model['Y_6M_9M'] = Y_9M

#Create rolling 6-month and 9-month variables

model['Y_6Mr'] = model[['Y_3M','Y_3M_6M']].where(model[['Y_3M','Y_3M_6M']] ==1).sum(axis=1)
model['Y_6M'] = model['Y_6Mr'].apply(lambda x: 1 if x >= 1 else 0)

model['Y_9Mr'] = model[['Y_3M','Y_3M_6M','Y_6M_9M']].where(model[['Y_3M','Y_3M_6M','Y_6M_9M']] ==1).sum(axis=1)
model['Y_9M'] = model['Y_9Mr'].apply(lambda x: 1 if x >= 1 else 0)

# Sort & Concat - To join the 'Y' variables to the main dataframe
model2 = pd.concat([df2.reset_index(drop=True), model[['Y_3M','Y_6M','Y_9M','Y_3M_6M','Y_6M_9M']]], axis=1)

model3 = model2[['ppa_purchase_id','quarter','count_list', 'quarter_average', 'new_used','ucc_status',
                    'dmkey_average_days','average_days', 'total_transaction_PPM', 'Y_3M','Y_6M','Y_9M','Y_3M_6M','Y_6M_9M']]


# ## Database Upload Point #2
#Establishing connection to Datamine
try:
    conn = mysql.connect(host='datamine.rcac.purdue.edu', port=3306, database='dbname',user='dbuser', password='dbpassword')

except:
    conn.close()
    
#Pushing data (pandas dataframe) into a new/existing table in MySQLServer
engine = create_engine("mysql+pymysql://dbuser:dbpassword@datamine.rcac.purdue.edu/dbname".format(user="dbuser",pw="dbpassword",db="dbname"))

model3 = pd.read_sql('select * from model3', conn)


#Data preprocessing required to convert from data format to time format

model3['uccdate'] = pd.to_datetime(model['uccdate'].astype(str), format='%Y/%m/%d')
model3['quarter'] = pd.PeriodIndex(model.uccdate, freq='Q')
model3=model3.drop(columns='Unnamed: 0')


# ## Data Preparation: Upsampling & Scaling 
data = model3

# Split Data
train = data.loc[data['quarter'] < '2018Q4'] 
test = data.loc[data['quarter'] == '2018Q4']

xtrain = train.iloc[:, 2:9].values
ytrain_3m = train.iloc[:, 9:10].values
ytrain_6m = train.iloc[:, 10:11].values
ytrain_9m = train.iloc[:, 11:12].values

xtest = test.iloc[:, 2:9].values
ytest_3m = test.iloc[:, 9:10].values
ytest_6m = test.iloc[:, 10:11].values
ytest_9m = test.iloc[:, 11:12].values

#Upsampling done to balance the dataset

sm = SMOTE(random_state=12, sampling_strategy = 1)  
xtrain_3m, ytrain_3m = sm.fit_sample(xtrain, ytrain_3m)
xtrain_6m, ytrain_6m = sm.fit_sample(xtrain, ytrain_6m)
xtrain_9m, ytrain_9m = sm.fit_sample(xtrain, ytrain_9m)


sc_x = MinMaxScaler() 
#sc_x = StandardScaler() 
xtrain = sc_x.fit_transform(xtrain)
xtrain_6m = sc_x.fit_transform(xtrain_6m)
xtrain_9m = sc_x.fit_transform(xtrain_9m)
xtest = sc_x.transform(xtest)

# ## Model Training
#1 RANDOM FOREST CLASSIFIER
rfc=RandomForestClassifier(class_weight='balanced')

# 3 MONTH FIT

start= time.time()
rfc.fit(xtrain_3m, ytrain_3m)
print("Training on RFC...")
y_pred_rfc_3m = (rfc.predict(xtest)) 
y_prob_rfc_3m = (rfc.predict_proba(xtest)) #new addition
end = time.time()
t=end-start
print("Time Taken: "+str(t))
rfc_y_3m, rfc_x_3m = calibration_curve(ytest_3m, y_prob_rfc_3m[:,1], n_bins=10) #new addition
f_rfc_3m = f1_score(ytest_3m, y_pred_rfc_3m)
rc_rfc_3m = recall_score(ytest_3m,y_pred_rfc_3m)
cm_rfc_3m = confusion_matrix(ytest_3m,y_pred_rfc_3m)


# 6 MONTH FIT

start= time.time()
rfc.fit(xtrain_6m, ytrain_6m)
print("Training on RFC...")
y_pred_rfc_6m = (rfc.predict(xtest)) 
y_prob_rfc_6m = (rfc.predict_proba(xtest)) #new addition
end = time.time()
t=end-start
print("Time Taken: "+str(t))

rfc_y_6m, rfc_x_6m = calibration_curve(ytest_6m, y_prob_rfc_6m[:,1], n_bins=10) #new addition
f_rfc_6m = f1_score(ytest_6m, y_pred_rfc_6m)
rc_rfc_6m = recall_score(ytest_6m,y_pred_rfc_6m)
cm_rfc_6m = confusion_matrix(ytest_6m,y_pred_rfc_6m)


# 9 MONTH FIT

start= time.time()
rfc.fit(xtrain_9m, ytrain_9m)
print("Training on RFC...")
y_pred_rfc_9m = (rfc.predict(xtest)) 
y_prob_rfc_9m = (rfc.predict_proba(xtest)) #new addition
end = time.time()
t=end-start
print("Time Taken: "+str(t))

rfc_y_9m, rfc_x_9m = calibration_curve(ytest_9m, y_prob_rfc_9m[:,1], n_bins=10) 
f_rfc_9m = f1_score(ytest_9m, y_pred_rfc_9m)
rc_rfc_9m = recall_score(ytest_9m,y_pred_rfc_9m)
cm_rfc_9m = confusion_matrix(ytest_9m,y_pred_rfc_9m)


#2 BAGGING CLASSIFIER
bag=BaggingClassifier()

# 3 MONTH FIT

start= time.time()
bag.fit(xtrain_3m, ytrain_3m)
print("Training on Bagging...")
y_pred_bag_3m = (bag.predict(xtest)) 
y_prob_bag_3m=(bag.predict_proba(xtest))
end = time.time()
t=end-start
print("Time Taken: "+str(t))

bag_y_3m, bag_x_3m = calibration_curve(ytest_3m, y_prob_bag_3m[:,1], n_bins=10)
f_bag_3m = f1_score(ytest_3m, y_pred_bag_3m)
rc_bag_3m = recall_score(ytest_3m,y_pred_bag_3m)
cm_bag_3m = confusion_matrix(ytest_3m,y_pred_bag_3m)


# 6 MONTH FIT

start= time.time()
bag.fit(xtrain_6m, ytrain_6m)
print("Training on Bagging...")
y_pred_bag_6m = (bag.predict(xtest)) 
y_prob_bag_6m=(bag.predict_proba(xtest))
end = time.time()
t=end-start
print("Time Taken: "+str(t))

bag_y_6m, bag_x_6m = calibration_curve(ytest_6m, y_prob_bag_6m[:,1], n_bins=10)
f_bag_6m = f1_score(ytest_6m, y_pred_bag_6m)
rc_bag_6m = recall_score(ytest_6m,y_pred_bag_6m)
cm_bag_6m = confusion_matrix(ytest_6m,y_pred_bag_6m)


# 9 MONTH FIT

start= time.time()
bag.fit(xtrain_9m, ytrain_9m)
print("Training on Bagging...")
y_pred_bag_9m = (bag.predict(xtest)) 
y_prob_bag_9m=(bag.predict_proba(xtest))
end = time.time()
t=end-start
print("Time Taken: "+str(t))

bag_y_9m, bag_x_9m = calibration_curve(ytest_9m, y_prob_bag_9m[:,1], n_bins=10)
f_bag_9m = f1_score(ytest_9m, y_pred_bag_9m)
rc_bag_9m = recall_score(ytest_9m,y_pred_bag_9m)
cm_bag_9m = confusion_matrix(ytest_9m,y_pred_bag_9m)


#3 LOGISTIC REGRESSION
log=LogisticRegression()

# 3 MONTH FIT

start= time.time()
log.fit(xtrain_3m, ytrain_3m)
print("Training on Logistic...")
y_pred_log_3m = (log.predict(xtest)) 
y_prob_log_3m =(log.predict_proba(xtest)) #new addition
end = time.time()
t=end-start
print("Time Taken: "+str(t))

logreg_y_3m, logreg_x_3m = calibration_curve(ytest_3m, y_prob_log_3m[:,1], n_bins=10) #new addition
f_log_3m = f1_score(ytest_3m, y_pred_log_3m)
rc_log_3m = recall_score(ytest_3m,y_pred_log_3m)
cm_log_3m = confusion_matrix(ytest_3m,y_pred_log_3m)


# 6 MONTH FIT

start= time.time()
log.fit(xtrain_6m, ytrain_6m)
print("Training on Logistic...")
y_pred_log_6m = (log.predict(xtest)) 
y_prob_log_6m =(log.predict_proba(xtest)) #new addition
end = time.time()
t=end-start
print("Time Taken: "+str(t))

logreg_y_6m, logreg_x_6m = calibration_curve(ytest_6m, y_prob_log_6m[:,1], n_bins=10) #new addition
f_log_6m = f1_score(ytest_6m, y_pred_log_6m)
rc_log_6m = recall_score(ytest_6m,y_pred_log_6m)
cm_log_6m = confusion_matrix(ytest_6m,y_pred_log_6m)


# 9 MONTH FIT

start= time.time()
log.fit(xtrain_9m, ytrain_9m)
print("Training on Logistic...")
y_pred_log_9m = (log.predict(xtest)) 
y_prob_log_9m =(log.predict_proba(xtest)) #new addition
end = time.time()
t=end-start
print("Time Taken: "+str(t))

logreg_y_9m, logreg_x_9m = calibration_curve(ytest_9m, y_prob_log_9m[:,1], n_bins=10) #new addition
f_log_9m = f1_score(ytest_9m, y_pred_log_9m)
rc_log_9m = recall_score(ytest_9m,y_pred_log_9m)
cm_log_9m = confusion_matrix(ytest_9m,y_pred_log_9m)


#4 LOGISTIC REGRESSION with CROSS VALIDATION
cv = StratifiedKFold(n_splits=5, random_state=10, shuffle=True)    
logcv=LogisticRegressionCV(cv = cv, n_jobs=-1, scoring = 'f1')

# 3 MONTH FIT

start= time.time()
logcv.fit(xtrain_3m, ytrain_3m)
print("Training on Logistic with Cross Validation ...")
y_pred_logcv_3m = (logcv.predict(xtest)) 
y_prob_logcv_3m =(logcv.predict_proba(xtest)) #new addition
end = time.time()
t=end-start
print("Time Taken: "+str(t))

logcvreg_y_3m, logcvreg_x_3m = calibration_curve(ytest_3m, y_prob_logcv_3m[:,1], n_bins=10) #new addition
f_logcv_3m = f1_score(ytest_3m, y_pred_logcv_3m)
rc_logcv_3m = recall_score(ytest_3m,y_pred_logcv_3m)
cm_logcv_3m = confusion_matrix(ytest_3m,y_pred_logcv_3m)


# 6 MONTH FIT

start= time.time()
logcv.fit(xtrain_6m, ytrain_6m)
print("Training on Logistic with Cross Validation ...")
y_pred_logcv_6m = (logcv.predict(xtest)) 
y_prob_logcv_6m =(logcv.predict_proba(xtest)) #new addition
end = time.time()
t=end-start
print("Time Taken: "+str(t))

logcvreg_y_6m, logcvreg_x_6m = calibration_curve(ytest_6m, y_prob_logcv_6m[:,1], n_bins=10) #new addition
f_logcv_6m = f1_score(ytest_6m, y_pred_logcv_6m)
rc_logcv_6m = recall_score(ytest_6m,y_pred_logcv_6m)
cm_logcv_6m = confusion_matrix(ytest_6m,y_pred_logcv_6m)


# 9 MONTH FIT

start= time.time()
logcv.fit(xtrain_9m, ytrain_9m)
print("Training on Logistic with Cross Validation ...")
y_pred_logcv_9m = (logcv.predict(xtest)) 
y_prob_logcv_9m =(logcv.predict_proba(xtest)) #new addition
end = time.time()
t=end-start
print("Time Taken: "+str(t))

logcvreg_y_9m, logcvreg_x_9m = calibration_curve(ytest_9m, y_prob_logcv_9m[:,1], n_bins=10) #new addition
f_logcv_9m = f1_score(ytest_9m, y_pred_logcv_9m)
rc_logcv_9m = recall_score(ytest_9m,y_pred_logcv_9m)
cm_logcv_9m = confusion_matrix(ytest_9m,y_pred_logcv_9m)


# ## Evaluate the models
#Dataframe for the PCP Plot

pcp = pd.DataFrame({'model':"Logistic",'interval':"3M",'predicted': logreg_x_3m, 'actual': logreg_y_3m})
pcp = pcp.append(pd.DataFrame({'model':"Random Forest",'interval':"3M",'predicted': rfc_x_3m, 'actual': rfc_y_3m}), ignore_index=True)
pcp = pcp.append(pd.DataFrame({'model':"Logistic CV",'interval':"3M",'predicted': logcvreg_x_3m, 'actual': logcvreg_y_3m}), ignore_index=True)
pcp = pcp.append(pd.DataFrame({'model':"Random Forest",'interval':"3M",'predicted': bag_x_3m, 'actual': bag_y_3m}), ignore_index=True)

pcp = pcp.append(pd.DataFrame({'model':"Logistic",'interval':"6M",'predicted': logreg_x_6m, 'actual': logreg_y_6m}), ignore_index=True)
pcp = pcp.append(pd.DataFrame({'model':"Random Forest",'interval':"6M",'predicted': rfc_x_6m, 'actual': rfc_y_6m}), ignore_index=True)
pcp = pcp.append(pd.DataFrame({'model':"Logistic CV",'interval':"6M",'predicted': logcvreg_x_6m, 'actual': logcvreg_y_6m}), ignore_index=True)
pcp = pcp.append(pd.DataFrame({'model':"Random Forest",'interval':"6M",'predicted': bag_x_6m, 'actual': bag_y_6m}), ignore_index=True)

pcp = pcp.append(pd.DataFrame({'model':"Logistic",'interval':"9M",'predicted': logreg_x_9m, 'actual': logreg_y_9m}), ignore_index=True)
pcp = pcp.append(pd.DataFrame({'model':"Random Forest",'interval':"9M",'predicted': rfc_x_9m, 'actual': rfc_y_9m}), ignore_index=True)
pcp = pcp.append(pd.DataFrame({'model':"Logistic CV",'interval':"9M",'predicted': logcvreg_x_9m, 'actual': logcvreg_y_9m}), ignore_index=True)
pcp = pcp.append(pd.DataFrame({'model':"Random Forest",'interval':"9M",'predicted': bag_x_9m, 'actual': bag_y_9m}), ignore_index=True)
pcp.to_csv('pcp.csv')


#Print the F1 Scores
#Change 9m to 3m/6m to get the scores for respective timeframes

print("F1 score Logistic: "+"{0:.2%}".format(f_log_9m))
print("F1 score Random Forest: "+"{0:.2%}".format(f_rfc_9m))
print("F1 score LogCV: "+"{0:.2%}".format(f_logcv_9m))
print("F1 score Bagging: "+"{0:.2%}".format(f_bag_9m))


#Print the Confusion matrix
#Change 9m to 3m/6m to get the scores for respective timeframes

print("Confusion Matrix Logistic\n:"+str(cm_log_9m))
print("Confusion Matrix Random Forest\n:"+str(cm_rfc_9m))
print("Confusion Matrix LogCV:\n"+str(cm_logcv_9m))
print("Confusion Matrix Bagging:\n"+str(cm_bag_9m))

#Print the Recall Score
#Change 9m to 3m/6m to get the scores for respective timeframes

print("Recall score Logistic: "+"{0:.2%}".format(rc_log_9m))
print("Recall score Random Forest: "+"{0:.2%}".format(rc_rfc_9m))
print("Recall score LogCV: "+"{0:.2%}".format(rc_logcv_9m))
print("Recall score Bagging: "+"{0:.2%}".format(rc_bag_9m))


# ### Feature Importance
# Calculate feature importances (Change 'rfc' to other classifiers to check for feature importance)

rfc=RandomForestClassifier(class_weight='balanced')

mod_fi3=rfc.fit(xtrain_3m, ytrain_3m)
mod_fi6=rfc.fit(xtrain_6m, ytrain_6m)
mod_fi9=rfc.fit(xtrain_9m, ytrain_9m)

# Assigning feature importances
importances_3 = mod_fi3.feature_importances_
importances_6 = mod_fi6.feature_importances_
importances_9 = mod_fi6.feature_importances_

features=train.columns[2:9]

# Appending Feature importance to a dataframe

importance_3=pd.DataFrame(importances_3,features,columns=['3_month'])
importance_6=pd.DataFrame(importances_6,features,columns=['6_month'])
importance_9=pd.DataFrame(importances_6,features,columns=['9_month'])

importance_n=pd.concat([importance_3,importance_6,importance_9],axis=1)

# Print Feature importance
importance_n


# ## Merge Output with Test Dataset
#Creating an intermediary dataframe with predictions
int_df = pd.concat((test.reset_index(drop=True),pd.DataFrame(y_prob_rfc_3m)[1].rename('Y_3M_Pred'),
                pd.DataFrame(y_prob_rfc_6m)[1].rename('Y_6M_Pred'),pd.DataFrame(y_prob_rfc_9m)[1].rename('Y_9M_Pred')), axis=1)

#Merging the "dmkey & PPM" to the intermediary output dataframe

final_df=pd.merge(int_df, model2[['ppa_purchase_id','dmkey','PPM']], how='left',
                  left_on='ppa_purchase_id',right_on='ppa_purchase_id')


#Merging the "company_aff & state" to the final output dataframe

final_df2=pd.merge(final_df,df2[['ppa_purchase_id','company_aff','state']], how='left',
                   left_on='ppa_purchase_id',right_on='ppa_purchase_id')

#Merging website activity with the output dataframe

final_df3=pd.merge(final_df2,web[['leadId']], how='left',left_on='dmkey',right_on='leadId')
final_df3['WebActivity'] = np.where(final_df3['dmkey']==final_df3['leadId'], 1, 0)


# ## Export Output to Database
#Establishing connection
try:
    conn = mysql.connect(host='datamine.rcac.purdue.edu', port=3306, database='dbname',user='dbuser', password='dbpassword')

except:
    conn.close()
    
#Pushing data (pandas dataframe) into a new/existing table in MySQLServer
engine = create_engine("mysql+pymysql://dbuser:dbpassword@datamine.rcac.purdue.edu/dbname".format(user="dbuser",pw="dbpassword",db="dbname"))

final_df4.to_sql('predictions', con = engine, chunksize = 10000)

pcp.to_sql('pcp', con = engine, chunksize = 1000)
importance_n.to_sql('feature_imp', con = engine, chunksize = 1000)
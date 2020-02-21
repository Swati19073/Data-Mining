#!/usr/bin/env python
# coding: utf-8

# In[191]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,roc_curve
from matplotlib import rcParams
from sklearn.metrics import roc_auc_score,roc_curve
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import AdaBoostClassifier
import seaborn as sns
import seaborn as sns; sns.set(style="ticks", color_codes=True)
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
heart_df=pd.read_csv("heart.csv")
#understanding data
heart_df.describe


# In[192]:


#data description
#frahmingham data, cleveland data base
# age: Age of the person
# sex: sex of the person ( 0 = female,1 = male)
# cp: Chest pain experienced by the person ( 1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic)
# typical angina is the discomfort that is noted when the heart does not get enough blood or oxygen.
# trestbps: The person's resting blood pressure (mm Hg on admission to the hospital)
# chol: The person's cholesterol measurement in mg/dl
# fbs: fasting blood sugar of a person(> 120 mg/dl, 1 = true; 0 = false)
# restecg: Resting electrocardiographic measurement (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy by Estes' criteria)
#restecg is slope measured by ecg machine
# thalach: maximum heart rate achieved
# exang: Exercise induced angina (value 1 = yes;value 0 = no)
# oldpeak: ST depression induced by exercise relative to rest ('ST' relates to positions on the ECG plot)
# ST depression refers to a finding on an electrocardiogram, wherein the trace in the ST segment is abnormally low below the baseline.
# slope: the slope of the peak exercise ST segment (1: upsloping,2: flat,3: downsloping)
# ca: The number of major vessels (0:if all vessels are fine,1:one vessel is not fine,2:2 vessels are not fine,3: no vessel is fine)
# thal: A blood disorder called thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect)
#Thalassemia is a blood disorder passed down through families (inherited) in which the body makes an abnormal form or inadequate amount of hemoglobin.
# target: Heart disease (0 = no, 1 = yes)


# In[ ]:





# In[193]:


#data analysis 
heart_df.head()


# In[194]:


heart_df.shape
#means our data has 1025 rows and 14 columns


# In[195]:


heart_df.dtypes


# In[196]:


#finding null values or missing values
print(heart_df.info())
heart_df.isnull().sum()
heart_df.isnull().values.any()
#there are no null values in our dataset


# In[197]:


#finding zero values 
heart_df.isin([0]).sum()


# In[198]:


#checking class is balanced or not

heart_df['target'].value_counts()


# In[199]:


heart_df['target'].value_counts().plot.pie(autopct='%2.2f%%', shadow=True)
plt.title("")
#no class imbalance problem


# In[200]:




sns.pairplot(heart_df,vars=["age"], diag_kind="hist", palette = 'hus1')
sns.pairplot(heart_df,vars=["sex"], diag_kind="hist", palette = 'hus1')
sns.pairplot(heart_df,vars=["cp"], diag_kind="hist", palette = 'hus1')
sns.pairplot(heart_df,vars=["ca"], diag_kind="hist", palette = 'hus1')
sns.pairplot(heart_df,vars=["trestbps"], diag_kind="hist", palette = 'hus1')
sns.pairplot(heart_df,vars=["chol"], diag_kind="hist", palette = 'hus1')
sns.pairplot(heart_df,vars=["fbs"], diag_kind="hist", palette = 'hus1')
sns.pairplot(heart_df,vars=["restecg"], diag_kind="hist", palette = 'hus1')
sns.pairplot(heart_df,vars=["thalach"], diag_kind="hist", palette = 'hus1')
sns.pairplot(heart_df,vars=["exang"], diag_kind="hist", palette = 'hus1')
sns.pairplot(heart_df,vars=["oldpeak"], diag_kind="hist", palette = 'hus1')
sns.pairplot(heart_df,vars=["slope"], diag_kind="hist", palette = 'hus1')
sns.pairplot(heart_df,vars=["thal"], diag_kind="hist", palette = 'pink')


# In[201]:


#https://github.com/joelmatt/Heart-Disease-Prediction/blob/master/Files/heart.ipynb

rcParams['figure.figsize'] = 15,15
a=heart_df.corr()
plt.matshow(heart_df.corr(),cmap="Spectral")
plt.yticks(np.arange(heart_df.shape[1]), heart_df.columns)
plt.xticks(np.arange(heart_df.shape[1]), heart_df.columns)
plt.colorbar()

#very less correlation so we can say that no data is related with each other
#by value of one feature we cannot predict any other feature


# In[202]:


#finding attributes whose correlation with target is greater than .3
corr_with_target=(a["target"])
higher_corr_features=corr_with_target[corr_with_target>0.30]
higher_corr_features


# In[203]:


boxplot = heart_df.boxplot(column=['age', 'chol', 'thalach'])


# In[204]:


indexN = heart_df[heart_df['chol'] >400].index
heart_df.drop(indexN, inplace=True)
heart_df.shape


# In[205]:


boxplot = heart_df.boxplot(column=[ 'sex', 'cp','fbs','restecg','exang','oldpeak','slope','ca','thal'])


# In[206]:


indexN1 = heart_df[heart_df['ca'] >3].index
heart_df.drop(indexN1, inplace=True)
heart_df.shape

indexN2 = heart_df[heart_df['oldpeak'] >5].index
heart_df.drop(indexN2, inplace=True)
heart_df.shape


# In[207]:


#https://machinelearningmastery.com/feature-selection-machine-learning-python/
acc_list=[]
heart_df.col=["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","target"]
column_values=["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]
val = heart_df.values
x_ax = val[:,0:13]
y_ax = val[:,13]
test = SelectKBest(score_func=chi2, k=5)
fit = test.fit(x_ax, y_ax)
np.set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(x_ax)
#we can see that thalach ,thal and slope are most relavant features of our data 


# In[ ]:





# In[208]:


#feature extraction of data using pca
#https://www.geeksforgeeks.org/ml-principal-component-analysispca/

#heart_df1=heart_df.drop(columns=["exang","oldpeak"])

  
scalar = StandardScaler() 
  
# fitting 
scalar.fit(heart_df) 
scaled_df = scalar.transform(heart_df) 

from sklearn.decomposition import PCA
p = PCA(n_components = 2) 
p.fit(scaled_df) 
x_axis = p.transform(scaled_df) 
x_std=p.fit_transform(scaled_df)
x_std.shape
print(x_std)
#x_axis.shape
#p.explained_variance_ratio_
#only .221+.129 % of the information is retained other information are lost.
#component1 is contributing only .221 and component2 is contributing .129 of the variance


# In[209]:


#https://stackoverflow.com/questions/50654620/add-legend-to-scatter-plot-pca


plt.figure(figsize =(8, 6)) 
scatter=plt.scatter(x_axis[:, 0], x_axis[:, 1], c = heart_df['target'], cmap ='cool') 
labels=np.unique(heart_df['target'])
handles = [plt.Line2D([],[],color=scatter.cmap(scatter.norm(yi))) for yi in labels]
plt.legend(handles, labels)
  
# labeling x and y axes 
plt.xlabel('First Principal Component') 

plt.ylabel('Second Principal Component')

plt.show()
#our data is linearly seperable
# As good as it seems like even a linear classifier could do very well to identify a class from the test set..


# In[210]:


#https://www.geeksforgeeks.org/classifying-data-using-support-vector-machinessvms-in-python/
#as data is linealy separable so we can use support vector machine as our classification model. It will give the best result
# x=heart_df.drop(["target"],axis=1)
y=heart_df.target.values
x=x_std
xtrain, xtest, ytrain, ytest = train_test_split(x_std,y,test_size=0.2)


# In[211]:


#take training data from heart_df also
SVM=SVC(kernel='rbf')
SVM.fit(xtrain,ytrain)
print ("Accuracy for training data :", SVM.score(xtrain,ytrain))
SVMscore = SVM.score(xtest,ytest)
print("Accuracy for testing data :" ,SVMscore)
from sklearn.metrics import confusion_matrix
y_pred_svm = SVM.fit(xtrain, ytrain).predict(xtest)
cm_svm = confusion_matrix(ytest, y_pred_svm)
print(cm_svm)

fig, axes = plt.subplots(figsize=(6,6))
sns.heatmap(cm_svm,annot = True, linewidths=2,fmt=".0f",axes=axes)
plt.xlabel("Predicted values")
plt.ylabel("True values")
plt.title("Confusion matrix for SVM")
plt.show()



# SVM = SVC(kernel='rbf')
scores_svm = cross_val_score(SVM, x, y, cv=5)
print(scores_svm) 
print("average accuracy after applying cross validation is:" ,scores_svm.mean())
acc_list.append(scores_svm.mean())


cv=[1,2,3,4,5]
x_pos = [i for i, _ in enumerate(cv)]
# plt.barh(cv, scores_logReg, align='center', alpha=0.5, color='green')
fig, ax = plt.subplots()
rects1 = ax.bar(cv,scores_svm,align='center', color='green')
plt.xlabel("cross validation steps")
plt.ylabel("accuracies")
plt.title('Accuracies after cross validation in SVM')
plt.xticks(cv)
plt.show()


#plotting ROC curve

SVM = CalibratedClassifierCV(SVM)
SVM.fit(xtrain,ytrain)
y_probabilities = SVM.predict_proba(xtest)[:,1]
FPR_svm,TPR_svm,threshold_svm = roc_curve(ytest,y_probabilities)
plt.figure(figsize=(5,5))
plt.title('Receiver Operating Characterstic for SVM')
plt.plot(FPR_svm,TPR_svm)
plt.plot([0,1],ls='--')
plt.plot([0,0],[1,0],c='.3')
plt.plot([1,1],c='.3')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()


# In[ ]:





# In[212]:


#https://www.kaggle.com/amanajmera1/logistic-regression-model-on-framingham-dataset
#https://github.com/ShubhankarRawat/Heart-Disease-Prediction/blob/master/heart_code.py
clf_logReg = LogisticRegression()
clf_logReg.fit(xtrain, ytrain)

# Predicting the Test set results
y_pred_logReg = clf_logReg.predict(xtest)

from sklearn.metrics import confusion_matrix
cm_test_lr = confusion_matrix(y_pred_logReg, ytest)

y_pred_train = clf_logReg.predict(xtrain)
cm_train_lr = confusion_matrix(y_pred_train, ytrain)
acc_train=((cm_train_lr[0][0] + cm_train_lr[1][1])/len(ytrain))
acc_test=((cm_test_lr[0][0] + cm_test_lr[1][1])/len(ytest))
# print()
print('Accuracy of training set for Logistic Regression:',acc_train)
print('Accuracy of test set for Logistic Regression :', acc_test)
from sklearn.metrics import confusion_matrix
y_pred_log = clf_logReg.fit(xtrain, ytrain).predict(xtest)
cm_logReg = confusion_matrix(ytest, y_pred_log)
print(cm_logReg)


#with cross validation
clf_logReg_cv = LogisticRegression()
scores_logReg = cross_val_score(clf_logReg_cv, x, y, cv=5)
print(scores_logReg) 
print("average accuracy after applying cross validation is:" ,scores_logReg.mean())
acc_list.append(scores_logReg.mean())






# In[213]:


cv=[1,2,3,4,5]
x_pos = [i for i, _ in enumerate(cv)]
# plt.barh(cv, scores_logReg, align='center', alpha=0.5, color='green')
fig, ax = plt.subplots()
rects1 = ax.bar(cv,scores_logReg,align='center', color='green')
plt.xlabel("cross validation steps")
plt.ylabel("accuracies")
plt.title('Accuracies after cross validation in Logistic Regression')
plt.xticks(cv)
plt.show()


# In[214]:


#plotting ROC curve

clf_logReg = CalibratedClassifierCV(clf_logReg)
clf_logReg.fit(xtrain,ytrain)
y_probabilities_LR = clf_logReg.predict_proba(xtest)[:,1]
FPR_LR,TPR_LR,threshold_LR = roc_curve(ytest,y_probabilities_LR)
plt.figure(figsize=(5,5))
plt.title('Receiver Operating Characterstic of Logistic Regression')
plt.plot(FPR_LR,TPR_LR)
plt.plot([0,1],ls='--')
plt.plot([0,0],[1,0],c='.3')
plt.plot([1,1],c='.3')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()


# In[215]:



clf_random=RandomForestClassifier(n_estimators=100, max_depth=3)
clf_random.fit(xtrain,ytrain)
y_pred_random=clf_random.predict(xtest)



rf_train=clf_random.score(xtrain, ytrain)
rf_test= clf_random.score(xtest, ytest)
print('Random Forest accuracy for training data', rf_train)
print('Random Forest accuracy for testing data',rf_test)

# confusion matrix
cm_random = confusion_matrix(ytest, y_pred_random)
print(cm_random)


clf_random_cv =RandomForestClassifier(n_estimators=100, max_depth=3)
scores_random = cross_val_score(clf_random_cv, x, y, cv=5)
print(scores_random) 
print("average accuracy after applying cross validation is:" ,scores_random.mean())
acc_list.append(scores_random.mean())
cv=[1,2,3,4,5]
x_pos = [i for i, _ in enumerate(cv)]
# plt.barh(cv, scores_logReg, align='center', alpha=0.5, color='green')
fig, ax = plt.subplots()
rects1 = ax.bar(cv,scores_random,align='center', color='green')
plt.xlabel("cross validation steps")
plt.ylabel("accuracies")
plt.title('Accuracies after cross validation in Random Forest')
plt.xticks(cv)
plt.show()


# In[216]:


fig, axes = plt.subplots(figsize=(6,6))
sns.heatmap(cm_random,annot = True, linewidths=2,fmt=".0f",axes=axes)
plt.xlabel("Predicted values")
plt.ylabel("True values")
plt.title("Confusion matrix for Ensembled Random Forest Classifier")
plt.show()

#https://medium.com/machine-learning-101/chapter-5-random-forest-classifier-56dc7425c3e1
#plotting ROC curve
clf_random = CalibratedClassifierCV(clf_random)
clf_random.fit(xtrain,ytrain)
y_probabilities_RF = clf_random.predict_proba(xtest)[:,1]
FPR_RF,TPR_RF,threshold_RF = roc_curve(ytest,y_probabilities_RF)
plt.figure(figsize=(5,5))
plt.title('Receiver Operating Characterstic of Randon Forest')
plt.plot(FPR_RF,TPR_RF)
plt.plot([0,1],ls='--')
plt.plot([0,0],[1,0],c='.3')
plt.plot([1,1],c='.3')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()


# In[217]:


#https://jtsulliv.github.io/perceptron/
#https://www.kaggle.com/excalibur7/predicting-heart-disease-plotly



# In[218]:



#https://www.kaggle.com/excalibur7/predicting-heart-disease-plotly
clf_ada = AdaBoostClassifier()
clf_ada.fit(xtrain, ytrain)
ada_train=clf_ada.score(xtrain, ytrain)
ada_test= clf_ada.score(xtest, ytest)
y_pred_ada=clf_ada.predict(xtest)
print('AdaBoost accuracy for training data', ada_train)
print('AdaBoost accuracy for testing data',ada_test)

#confusion matrix
cm_ada = confusion_matrix(ytest, y_pred_ada)
print(cm_ada)


# In[219]:


clf_ada_cv =AdaBoostClassifier()
scores_ada = cross_val_score(clf_ada_cv, x, y, cv=5)
print(scores_ada) 
print("average accuracy after applying cross validation is:" ,scores_ada.mean())
acc_list.append(scores_ada.mean())
cv=[1,2,3,4,5]
x_pos = [i for i, _ in enumerate(cv)]
# plt.barh(cv, scores_logReg, align='center', alpha=0.5, color='green')
fig, ax = plt.subplots()
rects1 = ax.bar(cv,scores_ada,align='center', color='green')
plt.xlabel("cross validation steps")
plt.ylabel("accuracies")
plt.title('Accuracies after cross validation in Adaboosting')
plt.xticks(cv)
plt.show()

fig, axes = plt.subplots(figsize=(6,6))
sns.heatmap(cm_ada,annot = True, linewidths=2,fmt=".0f",axes=axes)
plt.xlabel("Predicted values")
plt.ylabel("True values")
plt.title("Confusion matrix for Adaboost Classifier")
plt.show()


# In[220]:


#plotting ROC curve
clf_ada = CalibratedClassifierCV(clf_ada)
clf_ada.fit(xtrain,ytrain)
y_probabilities_ADA = clf_ada.predict_proba(xtest)[:,1]
FPR_ADA,TPR_ADA,threshold_ADA = roc_curve(ytest,y_probabilities_ADA)
plt.figure(figsize=(5,5))
plt.title('Receiver Operating Characterstic for adaboosting')
plt.plot(FPR_ADA,TPR_ADA)
plt.plot([0,1],ls='--')
plt.plot([0,0],[1,0],c='.3')
plt.plot([1,1],c='.3')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()


# In[221]:



z=[scores_svm.mean(),scores_logReg.mean(),scores_random.mean(),scores_ada.mean()]
classifiers=["SVM","Logistic Regression","Random Forest","AdaBoosting"]
fig = go.Figure(data=[go.Bar(x=classifiers, y=z,text=z,textposition='auto',)])
fig.show()


# In[ ]:





# In[ ]:





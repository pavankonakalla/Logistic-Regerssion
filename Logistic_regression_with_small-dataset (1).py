#!/usr/bin/env python
# coding: utf-8

# Logistic Regression:
# 
# Logistic regression is one such regression algorithm which can be used for performing classification problems.
# It calculates the probability that a given value belongs to a specific class. If the probability is more than 50%,
# it assigns the value in that particular class else if the probability is less than 50%, the value is assigned to the other
# class. Therefore, we can say that logistic regression acts as a binary classifier.

# Sigmoid function
# We use the sigmoid function as the underlying function in Logistic regression.
# Why do we use the Sigmoid Function?
# 
# 1) The sigmoid function’s range is bounded between 0 and 1. Thus it’s useful in calculating the probability for the Logistic function. 
# 2) It’s derivative is easy to calculate than other functions which is useful during gradient descent calculation. 
# 3) It is a simple way of introducing non-linearity to the model.
# 
# Although there are other functions as well, which can be used, but sigmoid is the most common function used for logistic regression. We will talk about the rest of the functions in the neural network section.

# Multiple Logistic Function
# We can generalise the simple logistic function for multiple features as:
# 
# And the logit function can be written as:
# 
# 
# The coefficients are calculated the same we did for simple logistic function, by passing the above equation in the
# cost function.Just like we did in multilinear regression, we will check for correlation between different features for
# Multi logistic as well.
# We will see how we implement all the above concept through a practical example.

# Evaluation of a Classification Model:
# 
# In machine learning, once we have a result of the classification problem, how do we measure how accurate our classification is? For a regression problem, we have different metrics like R Squared score, Mean Squared Error etc. what are the metrics to measure the credibility of a classification model?
# 
# Metrics In a regression problem, the accuracy is generally measured in terms of the difference in the actual values and the predicted values. In a classification problem, the credibility of the model is measured using the confusion matrix generated, i.e., how accurately the true positives and true negatives were predicted. The different metrics used for this purpose are:
# 
# Accuracy
# Recall
# Precision
# F1 Score
# Specifity
# AUC( Area Under the Curve)
# RUC(Receiver Operator Characteristic)
# Confusion Matrix
# A typical confusion matrix looks like the figure shown.
# 
# 
# Where the terms have the meaning:
# 
#  True Positive(TP): A result that was predicted as positive by the classification model and also is positive
# 
#  True Negative(TN): A result that was predicted as negative by the classification model and also is negative
# 
#  False Positive(FP): A result that was predicted as positive by the classification model but actually is negative
# 
#  False Negative(FN): A result that was predicted as negative by the classification model but actually is positive.
# 
# The Credibility of the model is based on how many correct predictions did the model do.
# 
# Accuracy
# The mathematical formula is :
# 
# Accuracy= (𝑇𝑃+𝑇𝑁)(𝑇𝑃+𝑇𝑁+𝐹𝑃+𝐹𝑁)
# Or, it can be said that it’s defined as the total number of correct classifications divided by the total number of classifications.
# 
# Recall or Sensitivity
# The mathematical formula is:
# 
# Recall= 𝑇𝑃(𝑇𝑃+𝐹𝑁)
# Or, as the name suggests, it is a measure of: from the total number of positive results how many positives were correctly predicted by the model.
# 
# It shows how relevant the model is, in terms of positive results only.
# 
# Let’s suppose in the previous model, the model gave 50 correct predictions(TP) but failed to identify 200 cancer patients(FN). Recall in that case will be:
# 
# Recall=50(50+200)
# = 0.2 (The model was able to recall only 20% of the cancer patients)
# 
# Precision
# Precision is a measure of amongst all the positive predictions, how many of them were actually positive. Mathematically,
# 
# Precision= 𝑇𝑃(𝑇𝑃+𝐹𝑃)
#  
# Let’s suppose in the previous example, the model identified 50 people as cancer patients(TP) but also raised a false alarm for 100 patients(FP). Hence,
# 
# Precision= 50(50+100)
#  =0.33 (The model only has a precision of 33%)

# F1 Score
# From the previous examples, it is clear that we need a metric that considers both Precision and Recall for evaluating a model. One such metric is the F1 score.
# 
# F1 score is defined as the harmonic mean of Precision and Recall.
# 
# The mathematical formula is: F1 score=  2∗((𝑃𝑟𝑒𝑐𝑖𝑠𝑖𝑜𝑛∗𝑅𝑒𝑐𝑎𝑙𝑙)(𝑃𝑟𝑒𝑐𝑖𝑠𝑖𝑜𝑛+𝑅𝑒𝑐𝑎𝑙𝑙))
#  
# Specificity or True Negative Rate
# This represents how specific is the model while predicting the True Negatives. Mathematically,
# 
# Specificity=𝑇𝑁(𝑇𝑁+𝐹𝑃)
#  Or, it can be said that it quantifies the total number of negatives predicted by the model with respect to the total number of actual negative or non favorable outcomes.
# 
# Similarly, False Positive rate can be defined as: (1- specificity) Or, 𝐹𝑃(𝑇𝑁+𝐹𝑃)

# # Python Implementation

# In[1]:


#Let's start with importing necessary libraries

import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler 
from statsmodels.stats.outliers_influence import variance_inflation_factor 
from sklearn.model_selection import train_test_split
from sklearn.linear_model  import Ridge,Lasso,RidgeCV, LassoCV, ElasticNet, ElasticNetCV, LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
import scikitplot as skl
sns.set()


# In[6]:


# Reading the Data
data=pd.read_csv("diabetes.csv")


# In[7]:


df.head()


# In[8]:


data.describe()


# Seems to be no missing as the count is same

# In[13]:


# let's see how data is distributed for every column
plt.figure(figsize=(20,25), facecolor='white')
plotnumber = 1

for column in data:
    if plotnumber<=9 :     # as there are 9 columns in the data
        ax = plt.subplot(3,3,plotnumber)
        sns.histplot(data[column])
        plt.xlabel(column,fontsize=20)
        #plt.ylabel('Salary',fontsize=20)
    plotnumber+=1
plt.show()


# We can see there is some skewness in the data, let's deal with data.
# 
# Also, we can see there few data for columns Glucose, Insulin, skin thickness, BMI and Blood Pressure which have value as 0. 
# That's not possible. You can do a quick search to see that one cannot have 0 values for these. Let's deal with that.
# we can either remove such data or simply replace it with their respective mean values. 
# Let's do the latter.

# In[14]:


data['BMI'] = data['BMI'].replace(0,data['BMI'].mean())
data['Glucose']=data['Glucose'].replace(0,data['Glucose'].mean())
data['BloodPressure']=data['BloodPressure'].replace(0,data['BloodPressure'].mean())
data['Insulin']=data['Insulin'].replace(0,data['Insulin'].mean())
data['SkinThickness']=data['SkinThickness'].replace(0,data['SkinThickness'].mean())


# In[16]:


# let's see how data is distributed for every column
plt.figure(figsize=(20,25), facecolor='white')
plotnumber = 1

for column in data:
    if plotnumber<=9 :
        ax = plt.subplot(3,3,plotnumber)
        sns.histplot(data[column])
        plt.xlabel(column,fontsize=20)
        #plt.ylabel('Salary',fontsize=20)
    plotnumber+=1
plt.show()


# Now we have dealt with the 0 values and data looks better. But, there still are outliers present in some columns. Let's deal with them.
# 
# ​

# In[17]:


fig, ax = plt.subplots(figsize=(15,10))
sns.boxplot(data=data, width= 0.5,ax=ax,  fliersize=3)


# In[20]:


q = data['Pregnancies'].quantile(0.98)
# we are removing the top 2% data from the Pregnancies column
data_cleaned = data[data['Pregnancies']<q]
q = data_cleaned['BMI'].quantile(0.99)
# we are removing the top 1% data from the BMI column
data_cleaned  = data_cleaned[data_cleaned['BMI']<q]
q = data_cleaned['SkinThickness'].quantile(0.99)
# we are removing the top 1% data from the SkinThickness column
data_cleaned  = data_cleaned[data_cleaned['SkinThickness']<q]
q = data_cleaned['Insulin'].quantile(0.95)
# we are removing the top 5% data from the Insulin column
data_cleaned  = data_cleaned[data_cleaned['Insulin']<q]
q = data_cleaned['DiabetesPedigreeFunction'].quantile(0.99)
# we are removing the top 1% data from the DiabetesPedigreeFunction column
data_cleaned  = data_cleaned[data_cleaned['DiabetesPedigreeFunction']<q]
q = data_cleaned['Age'].quantile(0.99)
# we are removing the top 1% data from the Age column
data_cleaned  = data_cleaned[data_cleaned['Age']<q]


# In[22]:


# let's see how data is distributed for every column
plt.figure(figsize=(20,25), facecolor='white')
plotnumber = 1

for column in data_cleaned:
    if plotnumber<=9 :
        ax = plt.subplot(3,3,plotnumber)
        sns.histplot(data_cleaned[column])
        plt.xlabel(column,fontsize=20)
        #plt.ylabel('Salary',fontsize=20)
    plotnumber+=1
plt.show()


# The data looks much better now than before. We will start our analysis with this data now as we don't want to lose important information. If our model doesn't work with accuracy,
# we will come back for more preprocessing.

# In[23]:


X = data.drop(columns = ['Outcome'])
y = data['Outcome']


# Before we fit our data to a model, let's visualize the relationship between our independent variables and the categories.

# In[26]:


# let's see how data is distributed for every column
plt.figure(figsize=(20,25), facecolor='white')
plotnumber = 1

for column in X:
    if plotnumber<=9 :
        ax = plt.subplot(3,3,plotnumber)
        sns.stripplot(y,X[column])
    plotnumber+=1
plt.tight_layout()


# Great!! Let's proceed by checking multicollinearity in the dependent variables.
# Before that, we should scale our data. Let's use the standard scaler for that.
# 

# In[27]:


scalar = StandardScaler()
X_scaled = scalar.fit_transform(X)


# In[28]:


X_scaled


# In[29]:


vif = pd.DataFrame()
vif["vif"] = [variance_inflation_factor(X_scaled,i) for i in range(X_scaled.shape[1])]
vif["Features"] = X.columns

#let's check the values
vif


# All the VIF values are less than 5 and are very low. That means no multicollinearity. 
# Now, we can go ahead with fitting our data to the model.
# Before that, let's split our data in test and training set.

# In[30]:


x_train,x_test,y_train,y_test = train_test_split(X_scaled,y, test_size= 0.25, random_state = 355)


# In[31]:


log_reg = LogisticRegression()

log_reg.fit(x_train,y_train)


# In[32]:


import pickle
# Writing different model files to file
with open( 'modelForPrediction.sav', 'wb') as f:
    pickle.dump(log_reg,f)
    
with open('sandardScalar.sav', 'wb') as f:
    pickle.dump(scalar,f)


# Let's see how well our model performs on the test data set.

# In[33]:


y_pred = log_reg.predict(x_test)


# In[34]:


accuracy = accuracy_score(y_test,y_pred)
accuracy


# In[35]:


# Confusion Matrix
conf_mat = confusion_matrix(y_test,y_pred)
conf_mat


# In[36]:


true_positive = conf_mat[0][0]
false_positive = conf_mat[0][1]
false_negative = conf_mat[1][0]
true_negative = conf_mat[1][1]


# In[37]:


# Precison
Precision = true_positive/(true_positive+false_positive)
Precision


# In[38]:


# Recall
Recall = true_positive/(true_positive+false_negative)
Recall


# In[39]:


# F1 Score
F1_Score = 2*(Recall * Precision) / (Recall + Precision)
F1_Score


# In[40]:


# Area Under Curve
auc = roc_auc_score(y_test, y_pred)
auc


# In[41]:


fpr, tpr, thresholds = roc_curve(y_test, y_pred)


# In[42]:


plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--',label='ROC curve (area = %0.2f)' % auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()


# What is the significance of Roc curve and AUC?
# In real life, we create various models using different algorithms that we can use for classification purpose. We use AUC to determine which model is the best one to use for a given dataset. Suppose we have created Logistic regression, SVM as well as a clustering model for classification purpose. We will calculate AUC for all the models seperately. The model with highest AUC value will be the best model to use.
# 
# Advantages of Logisitic Regression:
# It is very simple and easy to implement.
# The output is more informative than other classification algorithms
# It expresses the relationship between independent and dependent variables
# Very effective with linearly seperable data
# 
# Disadvantages of Logisitic Regression:
# Not effective with data which are not linearly seperable
# Not as powerful as other classification models
# Multiclass classifications are much easier to do with other algorithms than logisitic regression
# It can only predict categorical outcomes

# In[ ]:





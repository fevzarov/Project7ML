
# coding: utf-8

# ### Enron Fraud Case: Identifying Person of Interest (POI) using Machine Learning Tools
# 
# For this project, we utilize machine learning skills to choose a algorithm to identify Enron Employees who may have committed fraud based on the public Enron financial and email datasets. The approach to arrive at a best model include:
# 
# * handling real-world dataset: Since such dataset tends to be imperfect, we may have to clean and remove the data plus make some assumptions about the data inadequacies; 
# * validating machine learning results using test data;
# * evaluate machine learning result using quantitative metrics such as accuracy, precision, and recall scores;
# * dealing with features: This includes creating, selecting, and transforming variables;
# * comparing the performance of various machine learning algorithms
# * fine tuning machine learning algorithms for maximum performance; and
# * communicating your machine learning algorithm results clearly.
# 
# At the minimum, the best machine learning algorithm chose will have precision and recall scores of at least 0.3.
# 
# 
# ###### Project Overview
# 
# In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud due to misleading accounting practices. In the resulting Federal investigation, a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for top executives. 
# 
# Using Machine Learning tools, we build a best model for a person of interest identifier based on financial and email data made public as a result of the Enron scandal. The dataset provided by Udacity has the persons of interest in the fraud case tagged, which is defined by individuals who were indicted, reached a settlement or plea deal with the government, or testified in exchange for prosecution immunity.
# 
# ##### Analyses
# 

# In[1]:

# Base Packages & Path

import sys
import pickle
import matplotlib.pyplot
import tester
sys.path.append("../tools/")

import numpy as np
import pandas as pd

from feature_format import featureFormat, targetFeatureSplit

from tester import dump_classifier_and_data

from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn import tree
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from tester import test_classifier
from time import time


# ***
# __Task 1: Select what features you'll use__

# In[2]:

### Task 1: Select what features you'll use. ###
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
### features_list = ['poi','salary'] # You will need to use more features
# data_label = 'poi'
# features_list = enron_tools.get_features(data_dict)

# features_list = ['poi','salary', 'expenses', 'total_stock_value', 'bonus', 'from_poi_to_this_person', 'shared_receipt_with_poi'] # You will need to use more features

features_financial = ['salary', 'deferral_payments', 'total_payments',
                      'loan_advances', 'bonus', 'restricted_stock_deferred', 
                      'deferred_income', 'total_stock_value', 'expenses', 
                      'exercised_stock_options', 'other', 
                      'long_term_incentive', 'restricted_stock', 
                      'director_fees']
                      
features_email = ['to_messages', 'email_address', 'from_poi_to_this_person', 
                  'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

# Removing email address - not likely to be useful
# due to missing emails, already have name, and categorical nature.
features_email.remove('email_address')

# Initial Fueature List for Analysis
features_list = ['poi'] + features_financial + features_email

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


# ###### Data Overview
# 
# *Missing Values*
# 
# In research, missing values may not be the first thing we look into. After conducting a few analyses with the data, I noticed that the significant numbers of "NaN" values may pose a problem for the analysis. 
# 
# In this Machine Learning course, we did not go through the algorithm on how we arrived at the best fitting algorithm. However, based on what I know about regression, "NaN" or too many zero values would not be the most useful in arriving at a fitting model. So, my hypothesis is that variables with many NaN values would not be useful for precision and recall. Hopefully, *SelectKBest* will tell us if this is true. Besides *SelectKBest*, we need to be cognizant of not including variables that tend to be highly correlated - either via definition or correlation finding.
# 
# 

# In[3]:

### Missing Value count helps to determine less useful variables
# Dataframe Table Format

enron_df = pd.DataFrame.from_dict(data_dict, orient = "index")
enron_df.replace(to_replace='NaN', value=np.nan, inplace=True)

enron_df.count().sort_values()

# Missing data is an issue if we want to use the data for analyses
# Are these NaNs missing values or non-inputted values, thus, they should be zero?


# The features are sorted by the number of valid response with the least numberof valied response listed the first. A question we need to address here is whether the "not a number" response here are actually zero or missing values. This is actually a huge portion of a project in real life, especially for a case of this magnitude. This approach is time consuming because we extensive research requirement. A simpler approach would be to impute these zero-or-missing values but things can easily get messy here too. Imputations introduce errors to the data if not performed well. For this study, we assume that these values are actually **zeroes**.

# In[4]:

# Dataframe Table Format
enron_df = pd.DataFrame.from_dict(data_dict, orient = "index")

# Setting index name of pandas data frame to POI's name
enron_df.index.name = "Name"

# Removing Less Useful Column - Already Have Unique Name
enron_df.drop(["email_address"], inplace=True,axis=1)

#convert columns from objects to float
for c in enron_df.columns:
    if enron_df[c].dtype == object:
        enron_df[c] = enron_df[c].astype("float64")


# **Which observations have the many missing values?**

# In[5]:

### Missing Variables
enron_df["nmiss"] = enron_df.isnull().sum(axis = 1)
print(enron_df["nmiss"].describe())
print "Median=", (enron_df["nmiss"].median())


# In[6]:

# Let's Analyze Observation with so Many Missing Data
enron_df[enron_df["nmiss"]>15]


# We have one person with no meaningful data, so, we'll expunge **"LOCKHART EUGENE E"** from the dataset. I contemplated on removing the other observations with too many "NaN" values. 
# 
# I noticed that all of these individuals have NaN (zero?) income. They either had either  stocks or received director fees. Both of them may be useful in poi algorithm. So, we've decided to keep these observations. 

# In[7]:

pd.crosstab(index = enron_df["poi"],columns="count") 


# In[8]:

### Since describe produces freq and top=NaN, 
### convert 'NaN' from strings to a real 'NaN'
enron_df.replace(to_replace='NaN', value=np.nan, inplace=True)

print "Original Dataset Dimension:",enron_df.shape
print "Original Dataset Observations:",enron_df.shape[0]
print "Original Dataset Variables:",enron_df.shape[1]

print "\nUNIVARIATE STATISTICS"
enron_df.describe().transpose()
# enron_df.dtypes


# Univariate statistical analyses may tell us about bad data too. For example, we look into the data range. Data points thta fall outside an expected range shoudld be investigated. An obvious example is salary, which should be positive. 
# 
# My three immediate concerns pertain to *restricted stocks*, *restricted stock deferred*, and *total stocks* because these financial assets can take negative and positive values. Based on my limited knowledge on corporate finance, I can see how specific stocks may have negative values. People may have purchased stocks when their values were really high and sold them when their values were low - most likely at the *peak* of the scandal. In the end, I take all these values as accpetable and they are included in the analyses.
# 
# ##### Other Data Explorations

# In[9]:

# Typical Data Checks
# print "Dataframe dimension is", enron_df.shape
# print "\nName is the Index: \n",enron_df.index
# print "\nColumn Names: \n",enron_df.columns
# print "\nColumn Data Type: \n",enron_df.dtypes


# ***
# __Task 2: Remove outliers and Other Bad Data__

# ##### Data Cleaning
# 

# In[10]:

### read in data dictionary, convert to numpy array
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)

### Add these lines ... to make your scatterplot: 

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()


# In[11]:

# Outlier Identification
enron_df.loc[enron_df.salary>5000000]
# enron_df.loc['THE TRAVEL AGENCY IN THE PARK', :]


# In[12]:

# Based on concerns about missing data, I looked into their believability
# Wheh filter to NaN salary, I discovered a "THE TRAVEL AGENCY IN THE PARK"
enron_df[pd.isnull(enron_df.salary)]


# In[13]:

### Removing "TOTAL" & "THE TRAVEL AGENCY IN THE PARK"
data_dict.pop('TOTAL', 0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)
data_dict.pop('LOCKHART EUGENE E', 0)

data = featureFormat(data_dict, features)

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )
    
matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()


# In[14]:

### Keep the remaining outliers?
enron_df[(enron_df.salary>500000) | (enron_df.bonus>3000000)] 


# *Is it plausible that all these individuals got huge salary and/or bonus?* **Yes!**
# 
# * ALLEN PHILLIP K: He "received a $4.4 million bonus" (Barboza, 2002).
# 
# * BELDEN TIMOTHY N: Mr. Belden, "an executive on the power desk in Portland, Ore., got a $5.2 million bonus" (Barboza, 2002).
# 
# * FREVERT MARK A: Enron vice chairman.
# 
# * KITCHEN LOUISE: A "young British trader spearheading Enron's entry into Europe's energy markets"
# * LAVORATO JOHN J: Top Executive
# * LAY KENNETH L: Kenneth L. Lay, the company's former chairman and chief executive.
# * PICKERING MARK R: Chief Technology Officer
# * SKILLING JEFFREY K: Former Enron president and CEO
# * TOTAL: Removed from *data_dict*
# * WHALLEY LAWRENCE G: Enron president and chief operating officer
# 
# _References:_
# 
# Officials Got a Windfall Before Enron's Collapse, By David Barboza. June 18, 2002: https://www.nytimes.com/2002/06/18/business/officials-got-a-windfall-before-enron-s-collapse.html
# 
# The Guardian. Enron Key Players. January 12, 2002: https://www.theguardian.com/business/2002/jan/13/corporatefraud.enron
# 
# Los Angeles Times. Enron's Run Tripped by Arrogance, Greed. By David Streitfel and Lee Romney. January 27, 2002: http://articles.latimes.com/2002/jan/27/news/mn-25002

# In[15]:

# Count, by POI
print "Allocation Across Classes (POI/non-POI)"
enron_df.pivot_table(index='poi',aggfunc='count').transpose()


# ***
# **Task 3: Create new feature(s)**
# _plus Feature Reduction_

# In[16]:

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Created Features ...

def fraction_calc( numerator, denominator ):
    if numerator == 'NaN' or denominator == 'NaN':
        fraction = 0
    else:
        fraction = float(numerator)/float(denominator)    
    return round(fraction, 2)
    
    
def fraction_to_dict(dictionary, numerator, denominator, new_feature_name):
    num = dictionary[numerator]
    den = dictionary[denominator]
    fraction = fraction_calc(num, den)
    dictionary[new_feature_name] = fraction
    return dictionary
    
for p in my_dataset:      
    # New Feature: Total Stocks to Salary Ratio
    my_dataset[p] = fraction_to_dict(my_dataset[p],
                                        'total_stock_value',
                                        'salary',
                                        'stock_to_salary_ratio') 
                                        
    
features_financial = features_financial + ['stock_to_salary_ratio']

features_list = ['poi'] + features_financial + features_email


# In[17]:

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

print "Numpy Array Length:", len(data), "(v. 146 in the original data)"
print "\nFeatures List:", features_list
#print "Labels:", labels
#print "Features:", features


# _Feature Selction: Selecting Best Features for Analyses_
# 
# We used SelectKBest to select features to be utilized in our analyses, as shown below. We selected:
# 
# **['salary', 'bonus', 'total_stock_value','expenses', 'shared_receipt_with_poi', 'from_poi_to_this_person']**
# 
# These three were chosen because they have have larger scores compared to the others. If given more time, we would explore how many variables or principal components would be adequate in explaining the outcome variable. We have seen some analyses at work where programmers compared number of features vs. percentage of variance explained. Analysts were supposed to select a number of principal components that explains anywhere from 85% to 95% of variance. 

# In[18]:

# Univariate Feature Selection
import sklearn.feature_selection

features_financial = features_financial + ['stock_to_salary_ratio']
features_list = ['poi'] + features_financial + features_email

knum=7

k_best = sklearn.feature_selection.SelectKBest(k=knum)
k_best.fit(features, labels)
scores = k_best.scores_
unsorted_pairs = zip(features_list[1:], scores)
sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
k_best_features = dict(sorted_pairs[:knum])
print knum,"best features: {0}\n".format(k_best_features.keys())
k_best_features

import operator
#sorted(k_best_features.keys())
#sorted(x.items(), key=operator.itemgetter(1))
sorted(k_best_features.items(), key=operator.itemgetter(1), reverse=True)


# 
# ***
# 
# **Task 4: Try a varity of classifiers**

# In[19]:

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point... 


# I considered using the word bags aka text learning. Not sure if this would be useful because crooks may encrypt their wordings and there are definite other means of communication that are not examined here. e.g. phone, F2F.
# 
# I would have tried a regression analysis. However, obvioiusly a linear regression model is not appropiate for a 'POI vs. non-POI' analysis, i.e. the outcome variable is a binary variable. If I was given more time, I would explore a Logistic Regression model, which was not covered in this course. Personally, I am not clear on why Logistic Regression analysis was not covered in the course given that such analysis would have taken a 1-2 hours of lecture time. Besides the time constraint, I am happy to limit myself to the basic unsupervised machine learning methods.     
# 
# Thus, I resort to these machine learning algorithms:
# 1. Naive Bayes
# 2. Support vector machine
# 3. Decision Tree Classifier
# 

# In[20]:

### 1. Naive Bayes (default parameters)

features_list = ['poi', 'exercised_stock_options','bonus','salary'] 

# Test Size
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.30, random_state=42)

clf = GaussianNB()
clf.fit(features_train, labels_train)
predict = clf.predict(features_test)
accuracy = accuracy_score(predict, labels_test)
acc = accuracy_score(labels_test, predict)
prec = precision_score(labels_test, predict)
recall = recall_score(labels_test, predict)

print "Prediction Totals:", sum(predict) 
print "Accuracy =",("{0:.3f}".format(accuracy))
print "Acc =",("{0:.3f}".format(acc))
print "Prec =",("{0:.3f}".format(prec))
print "Rec =",("{0:.3f}".format(recall))

print "\nConfusion Matrix [0 v. 1]: \n",confusion_matrix(labels_test, predict)

print "\n-----------------------------------------------" 
print "\nTester Classification\n" 
test_classifier(clf, my_dataset, features_list)


# In[21]:

### 2. SVM (default parameters)

features_list = ['poi', 'exercised_stock_options','bonus','salary'] 

# Test Size
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.30, random_state=42)

clf = SVC()
clf.fit(features_train, labels_train)
predict = clf.predict(features_test)
accuracy = accuracy_score(predict, labels_test)
acc = accuracy_score(labels_test, predict)
prec = precision_score(labels_test, predict)
recall = recall_score(labels_test, predict)

print "Prediction Totals:", sum(predict) 
print "Accuracy =",("{0:.3f}".format(accuracy))
print "Acc =",("{0:.3f}".format(acc))
print "Prec =",("{0:.3f}".format(prec))
print "Rec =",("{0:.3f}".format(recall))

print "\nConfusion Matrix [0 v. 1]: \n",confusion_matrix(labels_test, predict)

print "\n-----------------------------------------------" 
print "\nTester Classification\n" 
test_classifier(clf, my_dataset, features_list)


# In[22]:

### 3. Decision Tree Classifier (default parameters)

features_list = ['poi', 'exercised_stock_options','bonus','salary'] 

# Test Size
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.30, random_state=42)

clf = tree.DecisionTreeClassifier()
clf.fit(features_train, labels_train)
predict = clf.predict(features_test)
accuracy = accuracy_score(predict, labels_test)
acc = accuracy_score(labels_test, predict)
prec = precision_score(labels_test, predict)
recall = recall_score(labels_test, predict)

print "Prediction Totals:", sum(predict) 
print "Accuracy =",("{0:.3f}".format(accuracy))
print "Acc =",("{0:.3f}".format(acc))
print "Prec =",("{0:.3f}".format(prec))
print "Rec =",("{0:.3f}".format(recall))

print "\nConfusion Matrix [0 v. 1]: \n",confusion_matrix(labels_test, predict)

print "\n-----------------------------------------------" 
print "\nTester Classification\n" 
test_classifier(clf, my_dataset, features_list)


# With the default parameters, Naive Bayes actually met the to meeting the 0.3 Precision and Recall standards and Decision Tree Classifier were close. We expected that algorithms' performance improve with fine tunings and scaling. 
# 
# ***
# 
# **Task 5: Tune your classifier**
# 

# In[23]:

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

#1. Naive Bayes 
# No tuning available with the Naive Bayes method

features_list = ['poi', 'exercised_stock_options','bonus','salary'] 
# features_list = ['poi', 'exercised_stock_options','bonus','salary','deferred_income','long_term_incentive','restricted_stock'] 

# Test Size
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.30, random_state=42)

clf = GaussianNB()
clf.fit(features_train, labels_train)
predict = clf.predict(features_test)
accuracy = accuracy_score(predict, labels_test)
acc = accuracy_score(labels_test, predict)
prec = precision_score(labels_test, predict)
recall = recall_score(labels_test, predict)

print "Prediction Totals:", sum(predict) 
print "Accuracy =",("{0:.3f}".format(accuracy))
print "Acc =",("{0:.3f}".format(acc))
print "Prec =",("{0:.3f}".format(prec))
print "Rec =",("{0:.3f}".format(recall))

print "\nConfusion Matrix [0 v. 1]: \n",confusion_matrix(labels_test, predict)

print "-----------------------------------------------" 

print "\nTester Classification\n" 
test_classifier(clf, my_dataset, features_list)


# ##### SVM Tuned Up
# 
# We have four cells dealing with SVM. Its short story:
# 1. SVM does not do a good job when tune ups include altering C, gamma, and SVM parameters. All our results' precision and recall metrics were 0.000.
# 2. SVM performs good when MinMaxSclaler() scaler transformation was applied.
# 
# _a. SVM without a Scaler Transformation_

# In[24]:

### 2. SVM (Tuned Up, but Unscaled)

features_list = ['poi', 'exercised_stock_options','bonus','salary'] 
# features_list = ['poi', 'exercised_stock_options','bonus','salary','deferred_income','long_term_incentive','restricted_stock'] 

C = [0.01, 0.1, 1, 10, 100]
gamma_list = [1, 0.1, 0.01, 0.001, 0.0001]
svm_param = ['rbf', 'sigmoid'] # kernel; rbf the fastest
# n_sample = [0.3, 0.35, 0.4, 0.5] # kernel; rbf the fastest

# for ccc in C
print "C | gamma | svmpar | n_pred | acc | prec | recall "
print "---------------------------------------------------------" 

# a more hands on approach for my understanding 
# especially on which parameters having the largest impact
for ccc in C:
    for ggg in gamma_list:
        for svmpar in svm_param:
            # for nnn in n_sample:
                # Test Size
            features_train, features_test, labels_train, labels_test             = train_test_split(features, labels, test_size=0.3, random_state=42)

            clf = SVC(kernel=svmpar, C = ccc, gamma=ggg)
            clf.fit(features_train, labels_train)
            predict = clf.predict(features_test)
            acc = accuracy_score(labels_test, predict)
            prec = precision_score(labels_test, predict)
            recall = recall_score(labels_test, predict)

            print ccc, ggg, svmpar,sum(predict),("{0:.3f}".format(acc)),             ("{0:.3f}".format(prec)), ("{0:.3f}".format(recall)),"\n"


# In[25]:

# SVM doesn't do a great job in predicting
# various sets of parameters result in 0.000 precision and recall scores
# The unsclaes SVM also suffers from "lack of true positive predicitons"

features_list = ['poi', 'exercised_stock_options','bonus','salary'] 

# Test Size
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

ccc = 100 # C
ggg = 0.1  # gamma_list
svmpar = 'sigmoid' # svm_param = ['rbf', 'sigmoid']; kernel; rbf the fastest

clf = SVC(kernel=svmpar, C = ccc, gamma=ggg)
clf.fit(features_train, labels_train)
predict = clf.predict(features_test)
acc = accuracy_score(labels_test, predict)
prec = precision_score(labels_test, predict)
recall = recall_score(labels_test, predict)

print ccc, ggg, svmpar,sum(predict),("{0:.3f}".format(acc)), ("{0:.3f}".format(prec)), ("{0:.3f}".format(recall)),"\n"
print "\nConfusion Matrix [0 v. 1]: \n",confusion_matrix(labels_test, predict)
print "\nTester Classification\n" 
test_classifier(clf, my_dataset, features_list)


# _b. SVM with a Scaler Transformation_

# In[26]:

# WRONG?: Scaler Transform
# SVM doesn't do a great job in predicting
# various sets of parameters result in 0.000 precision and recall scores

# # Apply standardization to SVM
# features_trainX = scaler.fit(features_train).transform(features_train)
# features_testX = scaler.fit(features_test).transform(features_test)

# # Test Size
# features_trainX, features_testX, labels_train, labels_test \
# = train_test_split(features, labels, test_size=0.3, random_state=42)


# In[27]:

## SVM Model (Tuned Up and Scaled)
from sklearn import svm

### Min-Max Scaler ###
# Very likely to be necessary due to wide ranges among variables
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)

features_train, features_test, labels_train, labels_test             = train_test_split(features, labels, test_size=0.3, random_state=42)

svm = Pipeline([('scaler',preprocessing.MinMaxScaler()),('svm',svm.SVC())])
param_grid = ([{'svm__C': [0.1, 1, 50, 100],
                'svm__gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                'svm__degree':[1,2],
                'svm__kernel': ['rbf','poly']}])

svm_grid_search = GridSearchCV(svm, param_grid, scoring='recall').fit(features, labels).best_estimator_
print "---------------------------------------------------------" 
print "\nTester Classification\n" 
tester.test_classifier(svm_grid_search, my_dataset, features_list)


# In[28]:

### 2. SVM Tested (Tuned Up and Scaled)

from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)

features_list = ['poi', 'exercised_stock_options', 'bonus', 'salary'] 

# Apply standardization to SVM
features_trainX = scaler.fit(features_train).transform(features_train)
features_testX = scaler.fit(features_test).transform(features_test)

# Test Size
features_trainX, features_testX, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

ccc = 10 # C
ggg = 1  # gamma_list
svmpar = 'sigmoid' # svm_param = ['rbf', 'sigmoid']; kernel; rbf the fastest

clf = SVC(kernel=svmpar, C = ccc, gamma=ggg)
clf.fit(features_trainX, labels_train)
predict = clf.predict(features_testX)
acc = accuracy_score(labels_test, predict)
prec = precision_score(labels_test, predict)
recall = recall_score(labels_test, predict)


print "C | gamma | svmpar | n_pred | acc | prec | recall \n"
print ccc, ggg, svmpar,sum(predict),("{0:.3f}".format(acc)), ("{0:.3f}".format(prec)), ("{0:.3f}".format(recall)),"\n"

print "\nConfusion Matrix [0 v. 1]: \n",confusion_matrix(labels_test, predict)
print "---------------------------------------------------------" 
print "\nTester Classification\n" 
tester.test_classifier(svm_grid_search, my_dataset, features_list)


# In[29]:

### 3. Decision Tree (Tuned Up)

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn import tree

features_list = ['poi', 'exercised_stock_options','bonus','salary'] 
# features_list = ['poi', 'exercised_stock_options','bonus','salary','deferred_income','long_term_incentive','restricted_stock'] 

# Test Size
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.30, random_state=42)

data = featureFormat(my_dataset, features_list, sort_keys = True)  
labels, features = targetFeatureSplit(data)

dt = tree.DecisionTreeClassifier()

parameters = {'criterion': ('gini','entropy'),
              'min_samples_split':[2, 3, 5, 10],
                'max_depth':[5,7,10,15,20,25],
                'max_leaf_nodes':[10,30,40],
             'splitter':('best','random')}


cv_strata = StratifiedShuffleSplit(labels, 100, random_state = 42)
grid_search = GridSearchCV(dt, parameters, cv=cv_strata, scoring='f1')

grid_search.fit(features, labels)
clf = grid_search.best_estimator_

from tester import test_classifier

print "Tester Classification\n" 
test_classifier(clf, my_dataset, features_list)


# In[30]:

clf = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=15,
            max_features=None, max_leaf_nodes=10, min_impurity_split=1e-07,
            min_samples_leaf=1, min_samples_split=10,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='best')
clf.fit(features_train, labels_train)
predict = clf.predict(features_test)
acc = accuracy_score(labels_test, predict)
prec = precision_score(labels_test, predict)
recall = recall_score(labels_test, predict)

print "\nConfusion Matrix [0 v. 1]: \n",confusion_matrix(labels_test, predict)
print "---------------------------------------------------------" 
print "\nTester Classification\n" 
test_classifier(clf, my_dataset, features_list)


# In[31]:

pd.DataFrame([[0.84277, 0.48281,  0.30900, 0.37683], 
              [0.85485, 0.63420, 0.13350, 0.22057],
              [0.81738, 0.38789, 0.32350, 0.35278]],
             columns = ['Accuracy','Precision', 'Recall', 'F1'], 
             index = ['Gaussian Naive Bayes','SVM','Decision Tree Classifier'])


# The tuned up Decision Tree Classifier model is the only model that satisfies the 0.3 precision and recal recall requirements. So, we select it to be the final model. 
# 
# ***
# 
# **Task 6: Dump your classifier, dataset, and features_list **
# 

# In[32]:

### Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)


# ##### Main References
# 
# 
# Decision Tree Classifier in Python using Scikit-learn: http://benalexkeen.com/decision-tree-classifier-in-python-using-scikit-learn/
# 
# Helpful Python Code Snippets for Data Exploration in Pandas (Michael Salmon): https://medium.com/@msalmon00/helpful-python-code-snippets-for-data-exploration-in-pandas-b7c5aed5ecb9
# 
# How does SelectKBest work?: https://datascience.stackexchange.com/questions/10773/how-does-selectkbest-work
# 
# How to apply standardization to SVMs in scikit-learn?: https://stackoverflow.com/questions/14688391/how-to-apply-standardization-to-svms-in-scikit-learn
# 
# How to compute precision, recall, accuracy and f1-score for the multiclass case with scikit learn?: https://stackoverflow.com/questions/31421413/how-to-compute-precision-recall-accuracy-and-f1-score-for-the-multiclass-case
# 
# How to filter in NaN (pandas)?: https://stackoverflow.com/questions/25050141/how-to-filter-in-nan-pandas
# 
# How to frame two for loops in list comprehension python: https://stackoverflow.com/questions/18551458/how-to-frame-two-for-loops-in-list-comprehension-python
# 
# How to get a classifier's confidence score for a prediction in sklearn?: https://stackoverflow.com/questions/31129592/how-to-get-a-classifiers-confidence-score-for-a-prediction-in-sklearn
# 
# Identify Fraud From Enron Data (Arjan Hada): https://arjan-hada.github.io/pages/about-me.html
# 
# Los Angeles Times. Enron's Run Tripped by Arrogance, Greed. By David Streitfel and Lee Romney. January 27, 2002: http://articles.latimes.com/2002/jan/27/news/mn-25002
# 
# Machine Learning with Python on the Enron Dataset - Investigating Fraud using Scikit-learn(William Koehrsen): https://medium.com/@williamkoehrsen/machine-learning-with-python-on-the-enron-dataset-8d71015be26d
# 
# Officials Got a Windfall Before Enron's Collapse, By David Barboza. June 18, 2002: https://www.nytimes.com/2002/06/18/business/officials-got-a-windfall-before-enron-s-collapse.html
# 
# Pandas: make pivot table with percentage: https://stackoverflow.com/questions/40301973/pandas-make-pivot-table-with-percentage
# 
# The Guardian. Enron Key Players. January 12, 2002: https://www.theguardian.com/business/2002/jan/13/corporatefraud.enron
# 
# Using iloc, loc, & ix to select rows and columns in Pandas DataFrames (Shane Lynn): https://www.shanelynn.ie/select-pandas-dataframe-rows-and-columns-using-iloc-loc-and-ix/
# 
# Visualize feature selection in descending order with SelectKBest: https://stackoverflow.com/questions/40245277/visualize-feature-selection-in-descending-order-with-selectkbest
# 
# Your First Machine Learning Project in Python Step-By-Step (Jason Brownlee): https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
# 

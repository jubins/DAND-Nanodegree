#!/usr/bin/python2
#@author: jubinsoni

import sys, os
import pickle

sys.path.append(os.getcwd()+"\\tools\\")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'exercised_stock_options', 'expenses', 'total_payments', 'total_stock_value']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
#Importing pandas and numpy
import pandas as pd
import numpy as np

enron_data = pd.DataFrame.from_dict(data_dict, orient='index')
print ("There are total {} people in the dataset.".format(enron_data.shape[0]))
print("Out of which {} are POI and {} are Non-POI.".format(enron_data.poi.value_counts()[True],
                                                          enron_data.poi.value_counts()[False]))
print("Total number of email plus financial features are {}.".format(enron_data.columns.shape[0]-1))
print("Label is 'poi' column.")

#Replacing the NaN values with zeroes
enron_data.replace(to_replace='NaN', value=0, inplace=True)

#Importing plotly
from plotly import tools
from plotly import plotly
from plotly import graph_objs

#Setting plotly API credentials
tools.set_credentials_file(username='jubinsoni', api_key='yKCkLUthlyqn7oXWf4U2')

#Making scatterplot before the 'TOTAL' outlier removal
with_outlier_total = graph_objs.Scatter(x = enron_data['salary'],
                           y = enron_data['bonus'],
                           text = enron_data.index,
                           mode = 'markers')

#Removing the outlier
enron_data.drop(labels=['TOTAL'], inplace=True)

#Making scatterplot after the 'TOTAL' outlier removal
without_outlier_total = graph_objs.Scatter(x = enron_data['salary'],
                                    y = enron_data['bonus'],
                                    text = enron_data.index,
                                    mode = 'markers')


#Layout the plots together side by side
fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('Before outlier TOTAL removal', 'After outlier TOTAL removal'))
fig.append_trace(with_outlier_total, 1, 1)
fig.append_trace(without_outlier_total, 1, 2)
fig['layout']['xaxis1'].update(title='salary')
fig['layout']['xaxis2'].update(title='salary')
fig['layout']['yaxis1'].update(title='bonus')
fig['layout']['yaxis2'].update(title='bonus')
plotly.iplot(fig)

#Making scatterplot before the 'TRAVEL AGENCY' outlier removal
with_outlier_travel = graph_objs.Scatter(x = enron_data['salary'],
                                  y = enron_data['bonus'],
                                  text = enron_data.index,
                                  mode = 'markers')
#drop
enron_data.drop(labels=['THE TRAVEL AGENCY IN THE PARK'], axis=0, inplace=True)

#Making scatterplot after the 'TRAVEL AGENCY' outlier removal
without_outlier_travel = graph_objs.Scatter(x = enron_data['salary'],
                                  y = enron_data['bonus'],
                                  text = enron_data.index,
                                  mode = 'markers')


#Layout the plots together side by side
fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('Before outlier TRAVEL AGENCY removal', 'After outlier TRAVEL AGENCY removal'))
fig.append_trace(with_outlier_travel, 1, 1)
fig.append_trace(without_outlier_travel, 1, 2)
fig['layout']['xaxis1'].update(title='salary')
fig['layout']['xaxis2'].update(title='salary')
fig['layout']['yaxis1'].update(title='bonus')
fig['layout']['yaxis2'].update(title='bonus')
plotly.iplot(fig)

### Task 3: Create new feature(s)
#Creating new feature(s)
enron_data['fraction_from_poi'] = enron_data['from_poi_to_this_person'].divide(enron_data['to_messages'], fill_value=0)
enron_data['fraction_to_poi'] = enron_data['from_this_person_to_poi'].divide(enron_data['to_messages'], fill_value=0)

#Replacing NaN in new features with 0
enron_data['fraction_from_poi'].fillna(value=0, inplace=True)
enron_data['fraction_to_poi'].fillna(value=0, inplace=True)

### Store to my_dataset for easy export below.
my_dataset = enron_data.to_dict('index')

from sklearn.feature_selection import SelectKBest, f_classif

features_list = ['poi', 'bonus', 'exercised_stock_options', 'expenses', 'other', 'restricted_stock', 'salary',
                'shared_receipt_with_poi', 'total_payments', 'total_stock_value', 'fraction_to_poi', 'fraction_from_poi']

data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

#Perform feature selection
selector = SelectKBest(f_classif, k=5)
selector.fit(features, labels)

#Get the raw p-values for each feature, and transform from p-values into scores
scores = -np.log10(selector.pvalues_)

#Bokeh Barplots
from bokeh.charts import Bar, show

data = {'scores': scores, 'features': features_list[1:]}

bar = Bar(data, label='features', values='scores', title='Select K Best',
         legend = None, plot_width=450, plot_height=450)

show(bar)

### Task 4: Try a varity of classifiers
### Extract features and labels from dataset for local testing
features_list = ['poi', 'bonus', 'exercised_stock_options', 'salary', 'total_stock_value',
                 'shared_receipt_with_poi', 'total_payments', 'fraction_to_poi', 'fraction_from_poi']
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

#Separating training and test dataset
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)
from sklearn.metrics import accuracy_score, classification_report

#Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(criterion='entropy', min_samples_leaf=1, min_samples_split=2)
rf = rf.fit(features_train, labels_train)
rf_labels_predicted = rf.predict(features_test)

rf_accuracy = accuracy_score(labels_test, rf_labels_predicted)
rf_classification_report = classification_report(labels_test, rf_labels_predicted)

print("Random Forest accuracy score: {}.".format(rf_accuracy))
print("Random Forest classification report:\n{}.".format(rf_classification_report))

pd.options.display.max_colwidth = 0

data = {'Algorithms':['GaussianNaiveBayes Classifier',
                      'SupportVectorMachines Classifier',
                      'AdaBoost Classifier',
                      'RandomForst Classifier'],
       'Parameters': ["Default",
                     "kernel='rbf', C=0.1, degree=3",
                      "learning_rate=0.1, algorithm='SAMME.R",
                      "criterion='entropy', min_samples_leaf=1, min_samples_split=2"
                     ],
       'Accuracy': [0.8604, 0.8837, 0.8139, 0.8604],
       'Precision': [0.86, 0.78, 0.77, 0.78],
        'Recall': [0.86, 0.88, 0.81, 0.86],
        'F1':[0.86, 0.83, 0.79, 0.82]
       }

algorithms = pd.DataFrame(data, columns=['Algorithms', 'Parameters', 'Accuracy', 'Precision', 'Recall', 'F1'])
print(algorithms)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
#Removing 'total_stock_value' from the features list
features_list = ['poi', 'bonus', 'exercised_stock_options', 'total_payments', 'salary',
                 'shared_receipt_with_poi', 'fraction_to_poi', 'fraction_from_poi']

#Converting the features into vectors
data = featureFormat(my_dataset, features_list, sort_keys=True)

#Splitting the features and labels
labels, features = targetFeatureSplit(data)

#Creating separate training and test sets
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

#Initializing the RandomForest Classifier on the tuned parameters
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_split=1, min_samples_leaf=5,
            min_samples_split=10, min_weight_fraction_leaf=0.0,
            n_estimators=10, n_jobs=1, oob_score=False, random_state=None,
            verbose=0, warm_start=False)

#Training the classifier
rf = rf.fit(features_train, labels_train)

#Predicting the labels
rf_labels_predicted = rf.predict(features_test)

#Calculating the accuracy, precision, recall and f1 scores
rf_accuracy = accuracy_score(labels_test, rf_labels_predicted)
rf_classification_report = classification_report(labels_test, rf_labels_predicted)

print("After Tuning:")
print("Random Forest accuracy score: {}.".format(rf_accuracy))
print("Random Forest classification report:\n{}.".format(rf_classification_report))


### Task 6: Dump your classifier, dataset, and features_list
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(criterion='entropy',
                            max_depth=2,
                            n_estimators=5,
                            min_samples_leaf=10,
                            min_samples_split=50)

dump_classifier_and_data(clf=rf, dataset=my_dataset, feature_list=features_list)

from tester import test_classifier
test_classifier(clf=rf, dataset=my_dataset, feature_list=features_list)
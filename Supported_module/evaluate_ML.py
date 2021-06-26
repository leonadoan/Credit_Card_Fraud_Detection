# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 20:52:59 2021

@author: DOAN, LE MINH THAO - A0213039
"""

# Loading packages

import matplotlib.pyplot as plt

import numpy as np
import seaborn as sns
import time


from sklearn.metrics import average_precision_score, roc_auc_score, classification_report, confusion_matrix, roc_curve, precision_recall_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.model_selection import learning_curve

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


import warnings
warnings.filterwarnings("ignore")



# Design ML

RANDOM_STATE = 38

classifiers = {
    "LogisiticRegression": LogisticRegression(random_state=RANDOM_STATE),
    "KNearest": KNeighborsClassifier(),
    "DecisionTreeClassifier": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest Classifier": RandomForestClassifier(random_state=RANDOM_STATE)
}

# For reference, before applying Gridsearch CV to find best parameter
def cross_validate(X_train, y_train, cv=5):
    for key, classifier in classifiers.items():
        classifier.fit(X_train, y_train)
        training_score = cross_val_score(classifier, X_train, y_train, cv=cv)
        print("Classifiers: ", classifier.__class__.__name__, 
              "has a training accuracy score of", round(training_score.mean(),2) * 100, "%")


# Use GridSearchCV to find the best parameters:
def model_best_estimator(x_train, y_train, class_weight=None, RANDOM_STATE=38, cv=5):
    
    # Logistic Regression 
    t0 = time.time()
    log_params_grid = {"solver": ["liblinear", "sag", "lbfgs"], "penalty":['l2'],
                       'C': [0.01, 0.1, 1, 100]}

    grid_log_reg = GridSearchCV(LogisticRegression(random_state=RANDOM_STATE, 
                                                   class_weight=class_weight, max_iter=10000),
                                log_params_grid, cv=cv, n_jobs=4)
    grid_log_reg.fit(x_train, y_train)

    # get the logistic regression with the best parameters.
    log_reg = grid_log_reg.best_estimator_
    t1 = time.time()

    print("Best fit parameter for Log regression", log_reg)
    print("Elapsed time {:.2f} s".format(t1 - t0))

    
    # KNN
    t2 = time.time()
    knears_params_grid = {"n_neighbors": list(range(2,8,1)), 
                          "metric": ('minkowski', 'euclidean', 'manhattan')}
    
    grid_knears = GridSearchCV(KNeighborsClassifier(), knears_params_grid, cv=cv)
    grid_knears.fit(x_train, y_train)
   
    # KNN best estimator
    knn = grid_knears.best_estimator_
    t3 = time.time()
    print("\nBest fit parameter for KNN", knn)
    print("Effective metric:", knn.effective_metric_)
    print("Elapsed time {:.2f} s".format(t3 - t2))
    
    
    # DecisionTree Classifier:
    t4 = time.time()
    tree_params_grid = {"criterion": ["gini", "entropy"], "max_depth": list(range(2,6,1)),
                        "min_samples_leaf": list(range(2,7,1))}
    grid_tree = GridSearchCV(DecisionTreeClassifier(random_state=RANDOM_STATE,
                                                    class_weight=class_weight),
                             tree_params_grid, cv=cv)
    grid_tree.fit(x_train, y_train)
    
    # tree best estimator
    tree_clf = grid_tree.best_estimator_
    t5 = time.time()
    
    print("\nBest fit parameter for Decision Tree:", tree_clf)
    print("Elapsed time {:.2f} s".format(t5 - t4))

    # Random Forest Classifier
    t6 = time.time()
    rf_params_grid = {"criterion": ["gini", "entropy"], "max_depth": list(range(2,6,1)), 
                "min_samples_leaf": list(range(2,7,1))}

    grid_rf = GridSearchCV(RandomForestClassifier(random_state=RANDOM_STATE,
                                                  class_weight=class_weight), 
                           rf_params_grid, cv=cv)
    grid_rf.fit(x_train, y_train)

    # random forest best estimator
    rf = grid_rf.best_estimator_
    t7 = time.time()

    print("\nBest fit parameter for Random Forest:", rf)
    print("Elapsed time {:.2f} s".format(t7 - t6))
    
    return [log_reg, knn, tree_clf, rf]   


# Evaluate model by using cross validation
def evaluate_model(classifier, x_train, y_train, cv=5):
    classifier.fit(x_train, y_train)
    score = cross_val_score(classifier, x_train, y_train, cv=cv)
    return score

# Get training model results
def train_model(classifier, x_train, y_train, cv=5):
    y_train_pre = cross_val_predict(classifier, x_train, y_train, cv=cv)
    print(classification_report(y_train, y_train_pre, labels=[1,0]))  
  

# Get testing model results
def predict_model(classifier, x_test, y_test):
    y_pre = classifier.predict(x_test)
    print(classification_report(y_test, y_pre, labels=[1,0]))
    
    # Confusion Matrix
    print('Confusion matrix:', classifier)
    cf_matrix = confusion_matrix(y_test, y_pre, labels=[1,0])
    ax =sns.heatmap(cf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=['Fraud', 'Normal'],
                yticklabels=['Fraud', 'Normal'])
    ax.set(xlabel="Predicted outputs", ylabel = "Actual outputs")
    plt.show()
    
    
# Plot ROC
def plot_result(log_reg, knn, tree_clf, rf, x_train, y_train, cv=5):
    # Get probability of y train predict:
    log_reg_pred = cross_val_predict(log_reg, x_train, y_train, cv=cv,
                             method="decision_function")
    knn_pred = cross_val_predict(knn, x_train, y_train, 
                                method='predict_proba', cv=cv)[:,1]
    tree_pred = cross_val_predict(tree_clf, x_train, y_train, 
                                method='predict_proba', cv=cv)[:,1]
    rf_pred = cross_val_predict(rf, x_train, y_train, 
                                method='predict_proba', cv=cv)[:,1]
    
    # calculate fpr and tpr and threshold
    log_fpr, log_tpr, log_thresold = roc_curve(y_train, log_reg_pred, pos_label=1)
    knn_fpr, knn_tpr, knn_threshold = roc_curve(y_train, knn_pred, pos_label=1)
    tree_fpr, tree_tpr, tree_threshold = roc_curve(y_train, tree_pred, pos_label=1)
    rf_fpr, rf_tpr, rf_threshold = roc_curve(y_train, rf_pred, pos_label=1)

    # Plot ROC
    
    f, (ax1, ax2) = plt.subplots(1,2, figsize=(16,8))
    
    ax2.plot(log_fpr, log_tpr, 
             label='Logistic Regression Classifier Score: {:.3f}'.format(roc_auc_score(y_train, log_reg_pred, labels=[1,0])))
    ax2.plot(knn_fpr, knn_tpr, 
             label='KNears Neighbors Classifier Score: {:.3f}'.format(roc_auc_score(y_train, knn_pred, labels=[1,0])))
    ax2.plot(tree_fpr, tree_tpr, 
             label='Decision Tree Classifier Score: {:.3f}'.format(roc_auc_score(y_train, tree_pred, labels=[1,0])))
    ax2.plot(rf_fpr, rf_tpr, 
             label='Random Forest Classifier Score: {:.3f}'.format(roc_auc_score(y_train, rf_pred, labels=[1,0])))
    ax2.plot([0, 1], [0, 1], 'k--')
    #ax2.axis([-0.01, 1, 0, 1])
    ax2.set_xlabel('False Positive Rate', fontsize=16)
    ax2.set_ylabel('True Positive Rate', fontsize=16)
    ax2.set_title('ROC Curve', fontsize=18)
    ax2.legend(loc = 'best')
    
    
    # calc precision, recall and thresholds
    log_precision, log_recall, log_thres_pr = precision_recall_curve(y_train, log_reg_pred, pos_label=1)
    knn_precision, knn_recall, knn_thres_pr = precision_recall_curve(y_train, knn_pred,  pos_label=1)
    tree_precision, tree_recall, tree_thres_pr = precision_recall_curve(y_train, tree_pred,  pos_label=1)
    rf_precision, rf_recall, rf_thres_pr = precision_recall_curve(y_train, rf_pred, pos_label=1)
    
    # Plot precision-recall curve
    ax1.plot(log_precision, log_recall, 
             label="Logistic Regression Classifier avg precision: {:0.3f}".format(average_precision_score(y_train, log_reg_pred)))
    ax1.plot(knn_precision, knn_recall, 
             label='KNears Neighbors Classifier avg precision: {:.3f}'.format(average_precision_score(y_train, knn_pred)))
    ax1.plot(tree_precision, tree_recall, 
             label='Decision Tree Classifier avg precision: {:.3f}'.format(average_precision_score(y_train, tree_pred)))
    ax1.plot(rf_precision, rf_recall, 
             label='Random Forest Classifier avg precision: {:.3f}'.format(average_precision_score(y_train, rf_pred)))
    ax1.set_xlabel('Precision', fontsize = 16)
    ax1.set_ylabel('Recall', fontsize = 16)
    #ax1.axis([-0.01, 1, 0, 1])
    ax1.set_title('Precision-Recall Curve', fontsize = 18)
    ax1.legend(loc = 'best')
   
    plt.show()


#Plot learning curve
def plot_learning_curve(classifier1, classifier2, classifier3, classifier4, X, y,
                        ylim=None, cv=5, n_jobs=4, train_sizes=np.linspace(.1, 1.0, 10)):
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(20,14), sharey=True)
    if ylim is not None:
        plt.ylim(*ylim)
    # First classifier
    train_sizes, train_scores, test_scores = learning_curve(classifier1, X, y, cv=cv,
                                                            n_jobs=n_jobs, train_sizes=train_sizes,
                                                            random_state=3,shuffle=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    ax1.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="#ff9124")
    ax1.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")
    ax1.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",
             label="Training score")
    ax1.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",
             label="Cross-validation score")
    ax1.set_title("Logistic Regression Learning Curve", fontsize=14)
    ax1.set_xlabel('Training sample')
    ax1.set_ylabel('Score')
    ax1.grid(True)
    ax1.legend(loc="best")
    
    # Second Estimator 
    train_sizes, train_scores, test_scores = learning_curve(
        classifier2, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,
        random_state=3, shuffle=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    ax2.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="#ff9124")
    ax2.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")
    ax2.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",
             label="Training score")
    ax2.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",
             label="Cross-validation score")
    ax2.set_title("Knears Neighbors Learning Curve", fontsize=14)
    ax2.set_xlabel('Training sample')
    ax2.set_ylabel('Score')
    ax2.grid(True)
    ax2.legend(loc="best")
    
    # Third Estimator
    train_sizes, train_scores, test_scores = learning_curve(
        classifier3, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, random_state=3,shuffle=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    ax3.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="#ff9124")
    ax3.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")
    ax3.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",
             label="Training score")
    ax3.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",
             label="Cross-validation score")
    ax3.set_title("Decision Tree Classifier \n Learning Curve", fontsize=14)
    ax3.set_xlabel('Training sample')
    ax3.set_ylabel('Score')
    ax3.grid(True)
    ax3.legend(loc="best")
    
    # Fourth Estimator
    train_sizes, train_scores, test_scores = learning_curve(
        classifier4, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, 
        random_state=3,shuffle=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    ax4.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="#ff9124")
    ax4.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")
    ax4.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",
             label="Training score")
    ax4.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",
             label="Cross-validation score")
    ax4.set_title("Random Forest\n Learning Curve", fontsize=14)
    ax4.set_xlabel('Training sample')
    ax4.set_ylabel('Score')
    ax4.grid(True)
    ax4.legend(loc="best")
    return plt



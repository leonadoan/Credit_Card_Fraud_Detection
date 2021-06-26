# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 20:52:59 2021

@author: DOAN, LE MINH THAO - A0213039
"""

# Loading packages

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import numpy as np
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from sklearn.preprocessing import RobustScaler  # avoid the outliner effect
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE



# Define functions:

# Data preparation - normalise, split dataset,...
class data_preparation():
    
    def  __init__(self, data, resample):
        self.data = data
        self.resample = resample
        
    # Scale amount data - Used robustscaler to remove outliers' influence
    # This scaler removes median and scales data according to the quantile range 
    def scale_data(self):
        self.x = self.data['Amount'].to_numpy()
        self.data['normAmount'] = RobustScaler().fit_transform(self.x.reshape(-1,1))
        return self.data
    
    def pre_process(self):
        df = self.scale_data()
        df = df.drop(['Amount', 'Time'],axis=1)
        return df
    
    # To reduce time train, resample data
    def get_resample(self):
        df = self.pre_process()
        non_fraud = df[df['Class'] == 0].sample(self.resample, random_state=3)
        fraud = df[df['Class'] == 1]
        
        # Merge 2 subset
        new_df = non_fraud.append(fraud).sample(frac=1, random_state=3).reset_index(drop=True)
        y = new_df["Class"].values
        
        print("After resampling, Number of Fraudulent Transactions : {}".format(sum(y==1)))
        print("After resampling, Number of Normal Transactions : {}".format(sum(y==0)))
        return new_df
    
    # Split data to train and test set
    def split_resampledata(self, TEST_SIZE = 0.2, RANDOM_STATE = 38):
        new_df = self.get_resample()
        X = new_df.drop(['Class'], axis = 1)
        y = new_df["Class"].values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE,
                                                            random_state=RANDOM_STATE,
                                                            shuffle=True)
        
        print("----------------------------------------------------")
        print(f"For the resample data {self.resample} normal transacions")
        print("Number transactions X_train dataset: ", X_train.shape)
        print("Number transactions y_train dataset: ", y_train.shape)
        print("Number transactions X_test dataset: ", X_test.shape)
        print("Number transactions y_test dataset: ", y_test.shape)
        print("----------------------------------------------------")
        print("Number of Fraudulent Transactions in train set : {}".format(sum(y_train==1)))
        print("Number of Normal Transactions in train set : {}".format(sum(y_train==0)))
        print("Number of Fraudulent Transactions in test set : {}".format(sum(y_test==1)))
        print("Number of Normal Transactions in test set : {}".format(sum(y_test==0)))
        
        return [X_train, X_test, y_train, y_test]
    
        

    def split_fulldata(self, TEST_SIZE = 0.2, RANDOM_STATE = 38):
        new_df = self.pre_process()
        X = new_df.drop(['Class'], axis = 1)
        y = new_df["Class"].values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE,
                                                            random_state=RANDOM_STATE,
                                                            shuffle=True)
        
        print("\nNumber transactions X_train dataset: ", X_train.shape)
        print("Number transactions y_train dataset: ", y_train.shape)
        print("Number transactions X_test dataset: ", X_test.shape)
        print("Number transactions y_test dataset: ", y_test.shape)
        print("----------------------------------------------------")
        print("Number of Fraudulent Transactions in train set : {}".format(sum(y_train==1)))
        print("Number of Normal Transactions in train set : {}".format(sum(y_train==0)))
        print("Number of Fraudulent Transactions in test set : {}".format(sum(y_test==1)))
        print("Number of Normal Transactions in test set : {}".format(sum(y_test==0)))
    
        return [X_train, X_test, y_train, y_test]
    
    # Correlation analysis
    def correlation(self):
        data = self.get_resample()
        colormap = plt.cm.Reds
        plt.figure(figsize=(12,10))
        sns.heatmap(data.corr(),linewidths=0.1,vmax=0.8, 
                    square=True, cmap = colormap, linecolor='white')
        plt.title('Correlation matrix', fontsize=14)
        plt.show()
        
    def plot_density(self):
        data = self.get_resample()
        var = data.columns.values
        var = np.delete(var, -2)
        
        i = 0
        t0 = data.loc[data['Class'] == 0]
        t1 = data.loc[data['Class'] == 1]
        
        sns.set_style('whitegrid')
        plt.figure()
        fig, ax = plt.subplots(6,5,figsize=(30,25))
        
        for feature in var:
            i += 1
            plt.subplot(6,5,i)
            sns.distplot(t0[feature], label="Class = 0")
            sns.distplot(t1[feature], label="Class = 1")
            plt.xlabel(feature, fontsize=12)
            locs, labels = plt.xticks()
            plt.tick_params(axis='both', which='major', labelsize=12)
        plt.show()
    
# Visualisation by PCA, t-SNE
class Dimensionality_Reduction_visualisation():
    
    def  __init__(self, data):
        self.data = data
    
    # Reduce dimension with t-SNE
    def set_TSNE(self, random_state=38):
        X = self.data.drop('Class', axis=1).values
        t0 = time.time()
        X_tsne = TSNE(n_components=2, random_state=random_state).fit_transform(X)
        t1 = time.time()
        print("T-SNE took {} s".format(t1 - t0))
        
        return X_tsne
    
    # Reduce dimension with PCA
    def set_PCA(self, random_state=38):
        X = self.data.drop('Class', axis=1).values
        t0 = time.time()
        X_pca = PCA(n_components=2, random_state=random_state).fit_transform(X)
        t1 = time.time()
        print("PCA took {:.2} s".format(t1 - t0))
        
        return X_pca
    
    def plot_reduce_dimension(self):
        X_tsne = self.set_TSNE()
        X_pca = self.set_PCA()
        y = self.data['Class']
        
        # Plot 2 graph
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,8))
        f.suptitle('Dimensionality Reduction', fontsize=16)
        
        b = mpatches.Patch(color='#282fed', label='No Fraud')
        r = mpatches.Patch(color='#f52318', label='Fraud')
        
        # t-SNE scatter plot
        ax1.scatter(X_tsne[:,0], X_tsne[:,1], c=(y == 0), cmap='bwr', 
                    label='No Fraud', linewidths=3)
        ax1.scatter(X_tsne[:,0], X_tsne[:,1], c=(y == 1), cmap='bwr',
                    label='Fraud', linewidths=3)
        ax1.set_title('t-SNE', fontsize=14)
        ax1.legend(handles=[b, r])
        ax1.grid(True)
        
        # pca scatter plot
        ax2.scatter(X_pca[:,0], X_pca[:,1], c=(y == 0), cmap='bwr', 
                    label='No Fraud', linewidths=3)
        ax2.scatter(X_pca[:,0], X_pca[:,1], c=(y == 1), cmap='bwr',
                    label='Fraud', linewidths=3)
        ax2.set_title('PCA', fontsize=14)
        ax2.legend(handles=[b, r])
        ax2.grid(True)

            
        plt.show()
    

# Sampling to solve imbalance dataset
# Use NearMiss and SMOTE
class sampling():
    
    def  __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def undersampling(self):
        nm = NearMiss(version=3)   #random_state didn't work
        X_train_nm, y_train_nm = nm.fit_resample(self.X_train, self.y_train)
        
        print("\nAfter UnderSampling, counts of fraud: {}".format(sum(y_train_nm==1)))
        print("After UnderSampling, counts of normal: {}".format(sum(y_train_nm==0)))
        
        return [X_train_nm, y_train_nm]
    
    def oversampling(self, RANDOM_STATE=38):
        sm = SMOTE(random_state=RANDOM_STATE)
        X_train_sm, y_train_sm = sm.fit_resample(self.X_train, self.y_train)
        
        print("\nAfter OverSampling, counts of fraud: {}".format(sum(y_train_sm==1)))
        print("After OverSampling, counts of normal: {}".format(sum(y_train_sm==0)))
        print('\nAfter OverSampling, the shape of X train: {}'.format(X_train_sm.shape))
        print('After OverSampling, the shape of y train: {} \n'.format(y_train_sm.shape))

        return [X_train_sm, y_train_sm]
    
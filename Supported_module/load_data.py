# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 20:52:59 2021

@author: DOAN, LE MINH THAO - A0213039
"""

# Loading packages
import pandas  as pd
import matplotlib.pyplot as plt


import numpy as np
import seaborn as sns



# Define functions:
# Load data, preprocessing + EDA 
class dataset():
    
    def  __init__(self, filepath):
        self.filepath = filepath
        
        
    def load_data(self):
        data = pd.read_csv(self.filepath)
        return data
    
    def count_value(self):
        data = self.drop_duplicate()
        print("After removing duplicated value:", data.shape)
        print(f'Number of Non-Fraudulent Transactions = {data["Class"].value_counts()[0]}')
        print(f'Number of Fraudulent Transactions = {data["Class"].value_counts()[1]}')
        print(f'Percent of Non-Fraudulent Transactions = {round(data["Class"].value_counts()[0]/len(data) * 100,2)}%')
        print(f'Percent of Fraudulent Transactions = {round(data["Class"].value_counts()[1]/len(data) * 100,2)}%')
    
        sns.countplot('Class', data=data, palette="Set2")
        plt.title('Distribution of normal and fraud transactions', fontsize=14)
        plt.show()
     
    # Get information from data
    def info(self):
        print(self.load_data().info())
        
    # Check missing value:
    def check_missing(self):
        print(self.load_data().isna().sum())

    # Check duplicated value:
    def check_duplicate(self):
        print("Number of duplicated valued",self.load_data().duplicated().sum())
    
    # Drop duplicated value
    def drop_duplicate(self):
        df = self.load_data().drop_duplicates(keep='first')
        return df

    # Plot the amount distribution of normal and fraud transactions    
    def plot_amount(self):
        data = self.drop_duplicate()
        Fraud = data[data["Class"]==1]
        Normal = data[data["Class"]==0]
        plt.figure(figsize=(12,6))
        plt.subplot(121)
        Fraud.Amount.plot.hist(title="Fraud Transacations", density="True")
        plt.subplot(122)
        Normal.Amount.plot.hist(title="Normal Transactions", density="True")
        plt.suptitle('Fraud and Normal Transactions amount', fontsize=14)
        plt.show()
        
    def plot_outlier(self):
        data = self.drop_duplicate()
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))
        sns.boxplot(ax = ax1, x="Class", y="Amount", hue="Class",data=data, palette="Set2",showfliers=True)
        sns.boxplot(ax = ax2, x="Class", y="Amount", hue="Class",data=data, palette="Set2",showfliers=False)
        plt.suptitle('Distribution of amount', fontsize=14)
        plt.show()
    
    def descriptive(self):
        data = self.drop_duplicate()
        print(data.iloc[:,:30].describe())
        
    # Correlation analysis
    def correlation(self):
        data = self.drop_duplicate()
        colormap = plt.cm.Reds
        plt.figure(figsize=(12,10))
        sns.heatmap(data.corr(),linewidths=0.1,vmax=0.8, 
                    square=True, cmap = colormap, linecolor='white')
        plt.title('Correlation matrix', fontsize=14)
        plt.show()
        
    def plot_density(self):
        data = self.drop_duplicate()
        var = data.columns.values
        var = np.delete(var, -1)
        
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
        

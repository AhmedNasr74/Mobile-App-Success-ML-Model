import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm, datasets
from skimage import color,transform
from itertools import accumulate
from ttictoc import TicToc
import pickle
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import  classification_report, r2_score
from sklearn.externals import joblib
import datetime as dt
def Feature_Encoder(X , cols):
    for c in cols:
        encoder = LabelEncoder()
        X[c] = encoder.fit_transform(X[c].astype(str))
    return X

def get_preprocessed_file_path():
    savingPath='../dataset/preprocessed_DS.csv'
    return savingPath

##########################################################################################

def generate_preprocessed_file():
    savingPath= get_preprocessed_file_path()
    if Path(savingPath).is_file() == False:
        timer = TicToc('preprocessing time')
        timer.tic()
        print('Generating preprocessed dataset file started ..........')
        ds_path = "../dataset/Mobile_App_Success_Milestone_2.csv" # Training
        if ds_path.__contains__('xlsx'):
            dataset = pd.read_excel(ds_path , parse_dates=['Last Updated'])
        else:
            dataset = pd.read_csv(ds_path , parse_dates=['Last Updated'],low_memory= False)
        dataset.dropna(how='any', inplace=True)
        
        dataset['App_Rating'] = dataset['App_Rating'].str.replace('Low_Rating','1')
        dataset['App_Rating'] = dataset['App_Rating'].str.replace('Intermediate_Rating','2')
        dataset['App_Rating'] = dataset['App_Rating'].str.replace('High_Rating','3')
        dataset['App_Rating'] = pd.to_numeric(dataset['App_Rating'] , downcast='integer', errors='coerce' )
        dataset['Last Updated'] = pd.to_datetime(dataset['Last Updated'], errors='coerce' )
        most_freq_date = dataset['Last Updated'].value_counts().idxmax()
        dataset['Last Updated'] = dataset['Last Updated'].fillna(most_freq_date)
        dataset['Price']    = dataset['Price'].str.replace('$','')
        dataset['Installs'] = dataset['Installs'].str.replace('+','')
        dataset['Installs'] = dataset['Installs'].str.replace(',','')
        
        dataset['Reviews'] = pd.to_numeric(dataset['Reviews'] , downcast='integer', errors='coerce' )
        dataset['Price'] = pd.to_numeric(dataset['Price'] , downcast='float', errors='coerce' )
        dataset['Installs'] = pd.to_numeric(dataset['Installs'] , downcast='integer', errors='coerce' )
        dataset.fillna(0)
        dataset["Size"] = dataset["Size"].str.replace("Varies with device",str(len(dataset[dataset['Size'] == "Varies with device"])))
        dataset["Size"] = dataset["Size"].str.replace(",","")
        dataset["Size"] = dataset["Size"].str.replace("+","")
        dataset["Size"] = (dataset["Size"].replace(r'[kM]+$', '', regex=True).astype(float) *\
                dataset["Size"].str.extract(r'[\d\.]+([kM]+)', expand=False).fillna(1).replace(['k','M'], [10**3, 10**6]).astype(int))
        dataset.fillna(0)
        contentRatings = dataset['Content Rating'].unique()
        for Uval in contentRatings:
           meanVal = len(dataset[dataset['Content Rating'] == Uval])
           dataset['Content Rating'] = dataset['Content Rating'].str.replace(Uval , str(meanVal))
        
        Categories = dataset['Category'].unique()
        for Uval in Categories:
           repeatingTimes = len(dataset[dataset['Category'] == Uval])
           dataset['Category'] = dataset['Category'].str.replace(Uval , str(repeatingTimes))
           
#        for i in range(len(dataset)) :
#            dataset.iloc[ i , 7] = datetime.timestamp(dataset.iloc[ i , 7]) # Last Update          
        dataset['Content Rating'] = pd.to_numeric(dataset['Content Rating'] , downcast='float', errors='coerce' )
        dataset['Category']= pd.to_numeric(dataset['Category'], downcast='integer', errors='coerce' )
        cols = ('App Name' ,'Last Updated' , 'Minimum Version', 'Latest Version')
        dataset = Feature_Encoder(dataset, cols)
        cols = dataset.columns
        for c in cols:
            dataset[c] = dataset[c].fillna(dataset[c].value_counts().idxmax())
        dataset.to_csv(path_or_buf=savingPath ,  index = False)
        print('Preprocessed File Generated Successfully.......')
        timer.toc()
        print('Preprocessing Time : ' + str(round(timer.elapsed/60, 5 )) + ' Minutes')
        return savingPath
    else:
        return get_preprocessed_file_path()
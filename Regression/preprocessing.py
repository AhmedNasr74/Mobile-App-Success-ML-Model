import numpy as np
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder,Imputer
from sklearn.preprocessing import PolynomialFeatures
import re
def Feature_Encoder(X , cols):
    for c in cols:
        encoder = LabelEncoder()
        X[c] = encoder.fit_transform(X[c])
    return X
def preprocessingTesting(dataset):
    dataset['Price'] = dataset['Price'].astype(str)
    dataset['Installs'] = dataset['Installs'].astype(str)
    dataset['Size'] = dataset['Size'].astype(str)
    dataset['App Name'] = dataset['App Name'].astype(str)
    dataset['Category'] = dataset['Category'].astype(str)
    dataset['Content Rating'] = dataset['Content Rating'].astype(str)
    dataset['Minimum Version'] = dataset['Minimum Version'].astype(str)
    dataset['Latest Version'] = dataset['Latest Version'].astype(str)

    
    
    
    dataset['Price'] = dataset['Price'].str.replace('$','')
    dataset['Installs'] = dataset['Installs'].str.replace('+','')
    dataset['Installs'] = dataset['Installs'].str.replace(',','')
    dataset['Reviews'] = pd.to_numeric(dataset['Reviews'] , downcast='integer', errors='coerce' )
    dataset['Price'] = pd.to_numeric(dataset['Price'] , downcast='float', errors='coerce' )
    dataset['Rating'] = pd.to_numeric(dataset['Rating'] , downcast='float', errors='coerce' )
    dataset['Installs'] = pd.to_numeric(dataset['Installs'] , downcast='integer', errors='coerce' )
    dataset.fillna(0)
    dataset['Size'] = dataset['Size'].str.replace('M','000').replace('.','')
    dataset['Size'] = dataset['Size'].str.replace('k','')
    dataset['Size'] = dataset['Size'].str.replace(',','')
    dataset['Size'] = dataset['Size'].str.replace('+','')
    dataset['Size'] = dataset['Size'].str.replace('Varies with device','0')
    dataset['Size'] = pd.to_numeric(dataset['Size'] , downcast='float', errors='coerce' )
    dataset.fillna(0)
    # Encoding
    cols = ('App Name','Category', 'Content Rating', 'Last Updated', 'Minimum Version', 'Latest Version')
    dataset = Feature_Encoder(dataset, cols)
    return dataset
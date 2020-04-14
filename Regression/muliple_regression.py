from preprocessing import *

dataset = pd.read_csv("../dataset/Predicting_Mobile_App_Success.csv" , parse_dates=['Last Updated'])

dataset.dropna(how='any', inplace=True)

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
#--------------------------------------------------------------------------------------------------------
# Encoding
cols = ('App Name','Category', 'Content Rating', 'Last Updated', 'Minimum Version', 'Latest Version')
dataset = Feature_Encoder(dataset, cols)
dataset.dropna(how='any', inplace=True)
# corrolation:
corr = dataset.corr()
plt.show()
#Top 50% Correlation training features with the Value
#top_feature = corr.index[abs(corr['Rating']>0.5)]
#Correlation plot
plt.subplots(figsize=(10, 10))
#top_corr = dataset[top_feature].corr()
sns.heatmap(corr, annot=True)
plt.show()

Y = dataset.iloc[:, 2]
X = dataset.iloc[:, [ 1, 3, 4, 5, 6, 7, 8 ] ]

#--------------------------------------------------------------------------------------------------------
# Split the data to training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=True, random_state=0)
#--------------------------------------------------------------------------------------------------------
# Model
cls = linear_model.LinearRegression()
cls.fit(X_train,y_train)
prediction= cls.predict(X_test)
print('Mean Square Error After Training', metrics.mean_squared_error(y_test, prediction))
plt.show()


from preprocessing import *
dataset = pd.read_csv("../dataset/Predicting_Mobile_App_Success.csv" , parse_dates=['Last Updated'])
dataset.dropna(how='any', inplace=True)
testingDS = pd.read_csv("../dataset/test/regression test.csv" , parse_dates=['Last Updated'])
testingDS = preprocessingTesting(testingDS)
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
Y_New_Test = testingDS.iloc[:, 2]
X_New_Test = testingDS.iloc[:, [ 1, 3, 4, 5, 6, 7, 8 ] ]
#--------------------------------------------------------------------------------------------------------
# Split the data to training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, shuffle=True, random_state=0)
#--------------------------------------------------------------------------------------------------------
# Model
poly_features = PolynomialFeatures(degree=2)
# transforms the existing features to higher degree features.
X_train_poly = poly_features.fit_transform(X_train)
# fit the transformed features to Linear Regression
poly_model = linear_model.LinearRegression()
poly_model.fit(X_train_poly, y_train)
# predicting on training data-set
y_train_predicted = poly_model.predict(X_train_poly)
# predicting on test data-set
prediction = poly_model.predict(poly_features.fit_transform(X_test))
print('Mean Square Error After Training', metrics.mean_squared_error(y_test, prediction))
prediction2 = poly_model.predict(poly_features.fit_transform(X_New_Test))
print('Mean Square Error Of New Testing File', metrics.mean_squared_error(Y_New_Test, prediction2))

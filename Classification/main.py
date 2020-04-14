from Models import *


dataset = pd.read_csv(generate_preprocessed_file())
Y = dataset.iloc[:, 10]
X = dataset.iloc[:, [1,2, 3, 4, 5, 6, 7,8,9] ]


#--------------------------------------------------------------------------------------------------------
# Split the data to training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=True, random_state=0)
#--------------------------------------------------------------------------------------------------------

#===================Start Calling Models=============================
AdaBoost_Model(X_train, X_test, y_train, y_test)
DecisionTree_Model(X_train, X_test, y_train, y_test)
LogisticRegression_Model(X_train, X_test, y_train, y_test)
KNN_Model(X_train, X_test, y_train, y_test)
SVM_Model(X_train, X_test, y_train, y_test)

#===================start Call with PCA =================================
AdaBoost_Model_PCA(X_train, X_test, y_train, y_test)
DecisionTree_Model_PCA(X_train, X_test, y_train, y_test)
LogisticRegression_Model_PCA(X_train, X_test, y_train, y_test)
KNN_Model_PCA(X_train, X_test, y_train, y_test)
SVM_Model_PCA(X_train, X_test, y_train, y_test)

#===================Show plots==============================================
Plot_Accuracy()
Plot_Total_train_Time()
Plot_Total_test_Time()



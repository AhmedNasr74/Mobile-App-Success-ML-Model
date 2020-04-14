from ProcessData import *

AccuracyList = []
name = []
TrainingTime = []
train = TicToc('timer')
TestingTime = []
test = TicToc('timer')

def AdaBoost_Model(X_train, X_test, y_train, y_test, d = 1):
    timer = TicToc('timer')
    timer.tic()
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    train.tic()
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=d),n_estimators=100)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    bdt.fit(X_train,y_train)
    train.toc()
    TrainingTime.append(round(train.elapsed/60 , 5))
    test.tic()
    X_test = sc.transform(X_test)
    X_test = scaler.transform(X_test)
    y_prediction = bdt.predict(X_test)
    accuracy=np.mean(y_prediction == y_test)*100
    test.toc()
    TestingTime.append(round(test.elapsed / 60, 5))
    AccuracyList.append(accuracy)
    name.append("AdaBoost")
#    print('Mean Square Error For AdaBoost Classifier : ', round(metrics.mean_squared_error(y_test, y_prediction) , 5))
    print("The achieved accuracy using Adaboost is " + str(round(accuracy , 3)))
    timer.toc()
#    print('AdaBoost Classifier Time : ' + str(round(timer.elapsed/60 , 5))+ ' Minutes')
    print('----------------------------------------------------------')
    filename = 'Models/AdaBoost.sav'
    pickle.dump(bdt,open(filename,'wb'))
    return bdt
#----------------------------------------------------------------------------------------------------------------------------
def DecisionTree_Model(X_train, X_test, y_train, y_test,d=3):
    timer = TicToc('timer')
    timer.tic()
    train.tic()
    clf = tree.DecisionTreeClassifier(max_depth=d)
    clf.fit(X_train,y_train)
    train.toc()
    TrainingTime.append(round(train.elapsed / 60, 5))
    test.tic()
    y_prediction = clf.predict(X_test)
    accuracy=np.mean(y_prediction == y_test)*100
    AccuracyList.append(accuracy)
    test.toc()
    TestingTime.append(round(test.elapsed / 60, 5))
    name.append("Tree")
#    print('Mean Square Error For Decision Tree Classifier : ', round(metrics.mean_squared_error(y_test, y_prediction) , 5))
    print("The achieved accuracy using Decision Tree is " + str(round(accuracy , 3)))
    timer.toc()
#    print('Decision Tree Time : ' + str(round(timer.elapsed/60 , 5))+ ' Minutes')
    print('----------------------------------------------------------')
    filename = 'Models/Tree.sav'
    pickle.dump(clf, open(filename, 'wb'))
    return  clf
#-----------------------------------------------------------------------------------------------------------------------------
def LogisticRegression_Model(X_train, X_test, y_train, y_test):
    timer = TicToc('timer')
    timer.tic()
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    train.tic()
    cls = LogisticRegression()
    cls.fit(X_train,y_train)
    train.toc()
    TrainingTime.append(round(train.elapsed / 60, 5))
    test.tic()
    X_test = sc.transform(X_test)
    prediction= cls.predict(X_test)
    accuracy = np.mean(prediction == y_test)*100
    test.toc()
    TestingTime.append(round(test.elapsed / 60, 5))
    AccuracyList.append(accuracy)
    name.append("Logistic")
#    print('Mean Square Error For Logistic Regression : ', metrics.mean_squared_error(np.asarray(y_test), prediction))
    print('The achieved accuracy using Logistic Regression is  '+str(round(accuracy , 3)))
    timer.toc()
#    print('Logistic Regression Time : ' + str(round(timer.elapsed/60 , 5))+ ' Minutes')
    print('----------------------------------------------------------')
    filename = 'Models/Logistic.sav'
    pickle.dump(cls, open(filename, 'wb'))
    return  cls
#-----------------------------------------------------------------------------------------------------------------------------
def KNN_Model(X_train, X_test, y_train, y_test , K=50):
    timer = TicToc('timer')
    timer.tic()
    train.tic()
    knn = KNeighborsClassifier(n_neighbors=K)
    knn.fit(X_train, y_train)
    train.toc()
    TrainingTime.append(round(train.elapsed / 60, 5))
    test.tic()
    prediction = knn.predict(X_test)
    accuracy = np.mean(prediction==y_test) * 100
    test.toc()
    TestingTime.append(round(test.elapsed / 60, 5))
    AccuracyList.append(accuracy)
    name.append("KNN")
#    print('Mean Square Error For KNN Classifier : ', round(metrics.mean_squared_error(y_test, prediction) , 5))
    print("The achieved accuracy using KNN is " + str(round(accuracy , 3)))
    timer.toc()
#    print('KNN Classifier Time : ' + str(round(timer.elapsed/60 , 5))+ ' Minutes')
    print('----------------------------------------------------------')
    filename = 'Models/Knn.sav'
    pickle.dump(knn, open(filename, 'wb'))
    return  knn
#----------------------------------------------------------------------------------------------------------------------------
def KNN_Model_KTrials(X_train, X_test, y_train, y_test , K=40):
    timer = TicToc('timer')
    timer.tic()
    train.tic()
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    train.toc()
    TrainingTime.append(round(train.elapsed / 60, 5))
    test.tic()
    X_test = scaler.transform(X_test)
    test.toc()
    error = []
    accuracy=[]
    # Calculating error for K values between 1 and K
    for i in range(1, K):
        train.tic()
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        train.toc()
        test.tic
        pred_i = knn.predict(X_test)
        error.append(np.mean(pred_i != y_test))
        accuracy.append(np.mean(pred_i==y_test)*100)
        test.toc()
    TrainingTime.append(round(train.elapsed / 60, 5))
    test.toc()
    TestingTime.append(round(test.elapsed / 60, 5))
    timer.toc()
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, K), error, color='red', linestyle='dashed', marker='o',
             markerfacecolor='blue', markersize=10)
    plt.title('Error Rate K Value')
    plt.xlabel('K Value')
    plt.ylabel('Mean Error')
    plt.show()
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, K), accuracy, color='red', linestyle='dashed', marker='o',
             markerfacecolor='blue', markersize=10)
    plt.title('Accurcy Rate K Value')
    plt.xlabel('K Value')
    plt.ylabel('Acc')
    plt.show()
    print('MSE Scores')
    print(error)
    print('Accurcy scores')
    print(accuracy)
    print('KNN Classifier Time : ' + str(round(timer.elapsed/60 , 5))+ ' Minutes')
    print('----------------------------------------------------------')
#----------------------------------------------------------------------------------------------------------------------------
def SVM_Model(X_train, X_test, y_train, y_test, c = 0.5, d = 2):
    timer = TicToc('timer')
    timer.tic()
    train.tic()
    svc = svm.SVC(kernel='poly', C= c , degree=d).fit(X_train, y_train)
    train.toc()
    TrainingTime.append(round(train.elapsed / 60, 5))
    test.tic()
    predictions = svc.predict(X_test)
    accuracy = np.mean(predictions == y_test)*100
    AccuracyList.append(accuracy)

    test.toc()
    TestingTime.append(round(test.elapsed / 60, 5))
    name.append("SVM")
#    print('Mean Square Error For SVM Classification : ', round(metrics.mean_squared_error(y_test, predictions) , 5))
    print('The achieved accuracy using SVM is  '+ str(round(accuracy , 3)))
    print('----------------------------------------------------------')
    timer.toc()
#    print('SVM Classifier Time : ' + str(round(timer.elapsed/60 , 5))+ ' Minutes')
    filename = 'Models/SVM.sav'
    pickle.dump(svc, open(filename, 'wb'))
    return svc
#----------------------------------------------------------------------------------------------------------------------------
def Kmean_Model(X_train, X_test, y_train, y_test, k=5):
    timer = TicToc('timer')
    timer.tic()
    train.tic()
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X_train,y_train)
    train.toc()
    TrainingTime.append(round(train.elapsed / 60, 5))
    test.tic()
    predict = kmeans.predict(X_test)
    accuracy = np.mean(predict == y_test)*100
    AccuracyList.append(accuracy)
    test.toc()
    TestingTime.append(round(test.elapsed / 60, 5))
    name.append("Kmean")
#    print('Mean Square Error For K-mean Classification : ',round(metrics.mean_squared_error(y_test, predict) , 5))
    print('The achieved accuracy using K-mean is  ' + str(round(accuracy , 3)))
    timer.toc()
#    print('AdaBoost Classifier Time : ' + str(round(timer.elapsed/60 , 5))+ ' Minutes')
    filename = 'Models\Kmean.sav'
    pickle.dump(kmeans, open(filename, 'wb'))
    return kmeans
#----------------------------------------------------------------------------------------------------------------------------



#--------------------------------------------------------------------------------------------------------------------
#PCA
def PCA1(X_train,X_test):
    pca_model = PCA(n_components=4)
    X_train = pca_model.fit_transform(X_train)
    X_test = pca_model.fit_transform(X_test)
    return X_train,X_test

def AdaBoost_Model_PCA(X_train, X_test, y_train, y_test, n=6,d=1):
    timer = TicToc('timer')
    timer.tic()
    train.tic()
    pca_model = PCA(n_components=4)
    pca_model.fit(X_train)
    X_train_PCA = pca_model.transform(X_train)
    X_test1_PCA = pca_model.transform(X_test)
    X_train_PCA,X_test1_PCA = PCA1(X_train,X_test)

    # AdaBoostClassifier With PCA
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=100)
    scaler = StandardScaler()
    scaler.fit(X_train_PCA)
    X_trainAda = scaler.transform(X_train_PCA)
    X_testAda = scaler.transform(X_test1_PCA)
    bdt.fit(X_trainAda, y_train)
    train.toc()
    TrainingTime.append(round(train.elapsed / 60, 5))
    test.tic()
    y_prediction = bdt.predict(X_testAda)
    accuracy = np.mean(y_prediction == y_test) * 100
    test.toc()
    TestingTime.append(round(test.elapsed / 60, 5))
    AccuracyList.append(accuracy)
    name.append("AdaBoost")
#    print('Mean Square Error For AdaBoost Classifier with PCA : ', metrics.mean_squared_error(y_test, y_prediction))
    print("The achieved accuracy using Adaboost with PCA is " + str(round(accuracy , 3)))
    timer.toc()
#    print('Decision Tree Time : ' + str(round(timer.elapsed / 60, 5)) + ' Minutes')
    print('----------------------------------------------------------')
    filename = 'Models\AdaBoostPCA.sav'
    pickle.dump(bdt, open(filename, 'wb'))
    return bdt
# ----------------------------------------------------------------------------------------------------------------------------
def DecisionTree_Model_PCA(X_train, X_test, y_train, y_test,p=4,d=1):
    timer = TicToc('timer')
    timer.tic()
    train.tic()
    pca_model = PCA(n_components=p)
    pca_model.fit(X_train)
    X_train_PCA = pca_model.transform(X_train)
    clf = tree.DecisionTreeClassifier(max_depth=d)
    clf.fit(X_train_PCA, y_train)
    train.toc()
    TrainingTime.append(round(train.elapsed / 60, 5))
    test.tic()
    X_test1_PCA = pca_model.transform(X_test)
    y_prediction = clf.predict(X_test1_PCA)
    accuracy = np.mean(y_prediction == y_test) * 100
    test.toc()
    TestingTime.append(round(test.elapsed / 60, 5))
    AccuracyList.append(accuracy)
    name.append("TreePCA")
#    print('Mean Square Error For Decision with PCA Tree Classifier : ', round(metrics.mean_squared_error(y_test, y_prediction), 5))
    print("The achieved accuracy using Decision With PCA Tree is " + str(round(accuracy, 3)))
    timer.toc()
#    print('Decision Tree Time : ' + str(round(timer.elapsed / 60, 5)) + ' Minutes')
    print('----------------------------------------------------------')
    filename = 'Models\TreePCA.sav'
    pickle.dump(clf, open(filename, 'wb'))
    return clf
# -----------------------------------------------------------------------------------------------------------------------------
def LogisticRegression_Model_PCA(X_train, X_test, y_train, y_test):
    timer = TicToc('timer')
    timer.tic()
    train.tic()
    pca_model = PCA(n_components=4)
    pca_model.fit(X_train)
    X_train_PCA = pca_model.transform(X_train)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train_PCA)
    cls = LogisticRegression()
    cls.fit(X_train, y_train)
    train.toc()
    TrainingTime.append(round(train.elapsed / 60, 5))
    test.tic()
    X_test1_PCA = pca_model.transform(X_test)
    X_test = sc.transform(X_test1_PCA)
    prediction = cls.predict(X_test)
    accuracy = np.mean(prediction == y_test) * 100
    test.toc()
    TestingTime.append(round(test.elapsed / 60, 5))
    AccuracyList.append(accuracy)
    name.append("Logistic""PCA")
#    print('Mean Square Error For Logistic with PCA Regression : ', metrics.mean_squared_error(np.asarray(y_test), prediction))
    print('The achieved accuracy using Logistic with PCA Regression is  ' + str(round(accuracy, 3)))
    timer.toc()
#    print('Logistic Regression Time : ' + str(round(timer.elapsed / 60, 5)) + ' Minutes')
    print('----------------------------------------------------------')
    filename = 'Models/LogisticPCA.sav'
    pickle.dump(cls, open(filename, 'wb'))
    return cls
# -----------------------------------------------------------------------------------------------------------------------------
def KNN_Model_PCA(X_train, X_test, y_train, y_test, K=40):
    timer = TicToc('timer')
    timer.tic()
    train.tic()
    pca_model = PCA(n_components=4)
    pca_model.fit(X_train)
    X_train_PCA = pca_model.transform(X_train)
    knn = KNeighborsClassifier(n_neighbors=K)
    knn.fit(X_train_PCA, y_train)
    train.toc()
    TrainingTime.append(round(train.elapsed / 60, 5))
    test.tic()
    X_test1_PCA = pca_model.transform(X_test)
    prediction = knn.predict(X_test1_PCA)
    accuracy = np.mean(prediction == y_test) * 100
    test.toc()
    TestingTime.append(round(test.elapsed / 60, 5))
    AccuracyList.append(accuracy)
    name.append("KNNPCA")
#    print('Mean Square Error For KNN with PCA Classifier : ', round(metrics.mean_squared_error(y_test, prediction), 5))
    print("The achieved accuracy using KNN with PCA is " + str(round(accuracy, 3)))
    timer.toc()
#    print('KNN Classifier Time : ' + str(round(timer.elapsed / 60, 5)) + ' Minutes')
    print('----------------------------------------------------------')
    filename = 'Models/KnnPCA.sav'
    pickle.dump(knn, open(filename, 'wb'))
    return knn
# ----------------------------------------------------------------------------------------------------------------------------
def SVM_Model_PCA(X_train, X_test, y_train, y_test,d=2):
    timer = TicToc('timer')
    timer.tic()
    train.tic()
    pca_model = PCA(n_components=4)
    pca_model.fit(X_train)
    X_train_PCA = pca_model.transform(X_train)
    svc = svm.SVC(kernel='poly', C=0.5, degree=d).fit(X_train_PCA, y_train)
    train.toc()
    TrainingTime.append(round(train.elapsed / 60, 5))
    test.tic()
    X_test1_PCA = pca_model.transform(X_test)
    predictions = svc.predict(X_test1_PCA)
    accuracy = np.mean(predictions == y_test) * 100
    AccuracyList.append(accuracy)
    test.toc()
    TestingTime.append(round(test.elapsed / 60, 5))
    name.append("SVMPCA")
#    print('Mean Square Error For SVM with PCA Classification : ', round(metrics.mean_squared_error(y_test, predictions), 5))
    print('The achieved accuracy using SVM with PCA is  ' + str(round(accuracy, 3)))
    timer.toc()
#    print('----------------------------------------------------------')
#    print('SVM Classifier Time : ' + str(round(timer.elapsed / 60, 5)) + ' Minutes')
    filename = 'Models/SVMPCA.sav'
    pickle.dump(pca_model, open(filename, 'wb'))
    return pca_model
# ----------------------------------------------------------------------------------------------------------------------------
def Kmean_Model_PCA(X_train, X_test, y_train, y_test,k=5):
    timer = TicToc('timer')
    timer.tic()
    train.tic()
    pca_model = PCA(n_components=4)
    pca_model.fit(X_train)
    X_train_PCA = pca_model.transform(X_train)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X_train_PCA)
    train.toc()
    TrainingTime.append(round(train.elapsed / 60, 5))
    test.tic()
    X_test1_PCA = pca_model.transform(X_test)
    predict = kmeans.predict(X_test1_PCA)
    accuracy = np.mean(predict == y_test) * 100
    test.toc()
    TestingTime.append(round(test.elapsed / 60, 5))
    AccuracyList.append(accuracy)
    name.append("KmeanPCA")
#    print('Mean Square Error For K-mean with PCA Classification : ', round(metrics.mean_squared_error(y_test, predict), 5))
    print('The achieved accuracy using K-mean with PCA is  ' + str(round(accuracy, 3)))
    timer.toc()
#    print('Kmean Classifier Time : ' +  str(round(timer.elapsed / 60, 5)) + ' Minutes')
    filename = 'Models/KmeanPCA.sav'
    pickle.dump(kmeans, open(filename, 'wb'))
    return kmeans
#--------------------------------------------------------------------------------------------------------------------------
def KNN_Model_KTrials_PCA(X_train, X_test, y_train, y_test , K=40):
    timer = TicToc('timer')
    timer.tic()
    train.tic()
    pca_model = PCA(n_components=4)
    pca_model.fit(X_train)
    X_train_PCA = pca_model.transform(X_train)
    train.toc()
    scaler = StandardScaler()
    scaler.fit(X_train_PCA)
    X_train = scaler.transform(X_train_PCA)
    test.tic()
    X_test1_PCA = pca_model.transform(X_test)
    X_test = scaler.transform(X_test1_PCA)
    test.toc()
    error = []
    accuracy=[]
    # Calculating error for K values between 1 and K
    for i in range(1, K):
        train.tic()
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        train.toc()
        test.tic()
        pred_i = knn.predict(X_test)
        error.append(np.mean(pred_i != y_test))
        accuracy.append(np.mean(pred_i==y_test)*100)
        test.toc()
    timer.toc()
    TestingTime.append(round(test.elapsed / 60, 5))
    TrainingTime.append(round(train.elapsed / 60, 5))
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, K), error, color='red', linestyle='dashed', marker='o',
             markerfacecolor='blue', markersize=10)
    plt.title('Error Rate K Value')
    plt.xlabel('K Value')
    plt.ylabel('Mean Error')
    plt.show()
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, K), accuracy, color='red', linestyle='dashed', marker='o',
             markerfacecolor='blue', markersize=10)
    plt.title('Accurcy Rate K Value')
    plt.xlabel('K Value')
    plt.ylabel('Acc')
    plt.show()
    print('MSE Scores')
    print(error)
    print('Accurcy scores')
    print(accuracy)
#    print('KNN Classifier Time : ' + str(round(timer.elapsed/60 , 5))+ ' Minutes')
    print('----------------------------------------------------------')
#===================================================================================================================
# Show Plots
def Plot_Accuracy():
    #print(name)
    #print(AccuracyList)
    plt.bar(name ,AccuracyList)
    plt.title('Accurcy bar')
    plt.xlabel('name')
    plt.ylabel('Accuracy')
    plt.show()

def Plot_Total_train_Time():
    #print(name)
    #print(TrainingTime)
    plt.bar(name ,TrainingTime)
    plt.title('Total train Time')
    plt.xlabel('name')
    plt.ylabel('Train Time')
    plt.show()

def Plot_Total_test_Time():
    plt.bar(name ,TestingTime)
    plt.title('Total Test Time')
    plt.xlabel('name')
    plt.ylabel('Test Time')
    plt.show()

def LoadModel(filename,X_test =0,y_test=0):
    load = pickle.load(open(filename,'rb'))
    #load.transform(X_test)
    output = load.predict(X_test)
    AccuracyOutput = np.mean(output == y_test)
    ModelName = filename.split('/')[1].split('.')[0]
    #print('Mean Square Error  of '+ModelName+' : ', round(metrics.mean_squared_error(y_test, output), 5))
    #print('Mean Square Error  of '+ModelName+' : ', metrics.mean_squared_error(y_test, output))
    print(str(ModelName) +' Accuracy of :' +str(AccuracyOutput*100) + '%')
    return  output

def PCA1_Model(model , X_test,Y_test):
    X_test = np.array(X_test).reshape(1,-1)
    Y_test = np.array(Y_test).reshape(1,-1)
    
    X_test = model.fit_transform(X_test)
    Y_test = model.fit_transform(Y_test)
    return X_test,Y_test
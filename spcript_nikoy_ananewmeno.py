
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
import seaborn as sns
from sklearn.model_selection import KFold


#fortwnoume dedomena training kai testing
#training = pd.read_excel("Wine_Training.xlsx")
#testing = pd.read_excel("Wine_Testing.xlsx")


#loading train data points 
training = pd.read_excel ('/home/nnanos/Documents/ΜΑΘΗΜΑΤΑ/ΘΕΩΡΙΑ_ΑΠΟΦΑΣΕΩΝ/code/Wine_Training.xlsx')
#converting train dataframe to numpy array
train_dataset = training.to_numpy()

#loading test input vectors
testing =  pd.read_excel ('/home/nnanos/Documents/ΜΑΘΗΜΑΤΑ/ΘΕΩΡΙΑ_ΑΠΟΦΑΣΕΩΝ/code/Wine_Testing.xlsx')
unknown_inputs = testing.to_numpy()


#A contains the inputs to our model as row vectors (each row vector represents the features of a specific wine)
A = train_dataset[:,:train_dataset.shape[1]-1]
#b is a column vector that contains the labeled outputs for each input row vector (these are the wine ratings) 
b = train_dataset[:,train_dataset.shape[1]-1]




choice = input('choose means of prediction:\n1) Logistic Regression\n2) Decision Tree\n3) Random Forest\n')


if choice=='1':
    #--------Logistic Regression------------

    flag = input('\n\nChoose 1 if you want to standardise the input data before training the model else choose 0\n:')

    if flag :
        scal = StandardScaler()
        A = scal.fit_transform(A)

    #using k-fold cross validation to test the algorithm-------------------
    kf = KFold(n_splits=5)
    KFold(n_splits=5, random_state=None, shuffle=False)
    k = kf.get_n_splits()

    test_accuray_for_each_fold = []
    for train_index, valid_index in kf.split(A):

        x_train , x_test = A[train_index], A[valid_index]
        y_train, y_test = b[train_index], b[valid_index]

        model = LogisticRegression()
        model.fit(x_train, y_train)
        y_pred_logisticreg = model.predict(x_test)

        
        test_accuray_for_each_fold.append( model.score(x_test, y_test) )

    
    #obtaining the mean test accuracy loss over all the folds    
    mean_test_accuracy = np.mean(test_accuray_for_each_fold)
    print(mean_test_accuracy)
    #-------------------------------------------------------------------------

    
    #epeita 8ewroume to neo testing arxeio gia na kanei ektimish krasiou me agnwsto quality
    #load wine attributes except quality
    x_train_new = training.drop(columns=['quality'])

    #load wine quality
    y_train_new = training[['quality']]

    #load test set
    x_test_new = testing


    model1 = LogisticRegression()
    model1.fit(x_train_new, y_train_new)
    y_pred_logisticreg_new = model1.predict(x_test_new)

    print(y_pred_logisticreg_new)
    plt.figure()
    plt.hist(y_pred_logisticreg_new, bins=50)
    plt.axvline(0, linestyle='--', color='r', label='lower boundary of the accepted ratings')
    plt.axvline(10, linestyle='--', color='b', label='upper boundary of the accepted ratings')
    plt.title("Distribution of wine quality based on Logistic Regression Algorithm")
    plt.legend()
    plt.tight_layout()
    plt.show()


elif choice=='2':

    #using k-fold cross validation 
    kf = KFold(n_splits=5)
    KFold(n_splits=5, random_state=None, shuffle=False)
    k = kf.get_n_splits()

    

    test_accuray_for_each_fold = []
    for train_index, valid_index in kf.split(A):

        x_train , x_test = A[train_index], A[valid_index]
        y_train, y_test = b[train_index], b[valid_index]

        model = DecisionTreeClassifier()
        model = model.fit(x_train, y_train)
        y_pred_dectree = model.predict(x_test)
                
                
        test_accuray_for_each_fold.append( model.score(x_test, y_test) )

    
    #obtaining the mean test accuracy loss over all the folds    
    mean_test_accuracy = np.mean(test_accuray_for_each_fold)
    print(mean_test_accuracy)
    #-------------------------------------------------------------------------

    x_train_new = training.drop(columns=['quality'])

    # load wine quality
    y_train_new = training[['quality']]

    # load test set
    x_test_new = testing

    model1 = DecisionTreeClassifier()
    model1.fit(x_train_new, y_train_new)
    y_pred_dectree_new = model1.predict(x_test_new)

    print(y_pred_dectree_new)
    plt.figure()
    plt.hist(y_pred_dectree_new, bins=50)
    plt.axvline(0, linestyle='--', color='r', label='lower boundary of the accepted ratings')
    plt.axvline(10, linestyle='--', color='b', label='upper boundary of the accepted ratings')
    plt.title("Distribution of wine quality based on Decision Tree Algorithm")
    plt.legend()
    plt.tight_layout()
    plt.show()


elif choice=='3':

    flag = input('\n\nChoose 1 if you want to standardise the input data before training the model else choose 0\n:')
    if flag :
        scal = StandardScaler()
        A = scal.fit_transform(A)

    estim = int(input('Give number of estimators:\n'))

    #using k-fold cross validation -----------------------------------------
    kf = KFold(n_splits=5)
    KFold(n_splits=5, random_state=None, shuffle=False)
    k = kf.get_n_splits()


    test_accuray_for_each_fold = []
    for train_index, valid_index in kf.split(A):

        x_train , x_test = A[train_index], A[valid_index]
        y_train, y_test = b[train_index], b[valid_index]

        model = RandomForestClassifier(n_estimators=estim)
        model.fit(x_train, y_train)
        y_pred_ranfor = model.predict(x_test)

        test_accuray_for_each_fold.append( model.score(x_test, y_test) )


    #obtaining the mean test accuracy loss over all the folds    
    mean_test_accuracy = np.mean(test_accuray_for_each_fold)
    print(mean_test_accuracy)
    #-------------------------------------------------------------------------


    x_train_new = training.drop(columns=['quality'])

    # load wine quality
    y_train_new = training[['quality']]

    # load test set
    x_test_new = testing

    model1 = DecisionTreeClassifier()
    model1.fit(x_train_new, y_train_new)
    y_pred_dectree_new = model1.predict(x_test_new)

    print(y_pred_dectree_new)
    plt.figure()
    plt.hist(y_pred_dectree_new, bins=50)
    plt.axvline(0, linestyle='--', color='r', label='lower boundary of the accepted ratings')
    plt.axvline(10, linestyle='--', color='b', label='upper boundary of the accepted ratings')
    plt.title("Distribution of wine quality based on Decision Tree Algorithm")
    plt.legend()
    plt.tight_layout()
    plt.show()

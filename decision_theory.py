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


#fortwnoume dedomena training kai testing
training = pd.read_excel("Wine_Training.xlsx")
testing = pd.read_excel("Wine_Testing.xlsx")


#-----------diagramata-----------------
col = training.columns

#for i in range(len(col)-1):
#    print(col[i])
#    plt.bar(data['quality'], data[col[i]], color='maroon')
#    plt.title('Relation of ' + col[i] + ' with wine')
#    plt.xlabel('quality')
#    plt.ylabel(col[i])
#    plt.show()

#----------dedomena krasiwn--------------
#fig = sns.pairplot(training)
#fig.savefig('Wine_Correlations.png')



#----------proetoimasia dedomenwn----------------
#arxika 8a 8esoume apo ta hdh gnwsta dedomena mas ena pososto gia train kai ena pososto gia test gia na doume
#kata poso swsta ta katatasei to montelo mas. dinoume ena pososto 20% gia test kai 80% gia train

stats = training.drop(columns=['quality'])
qual = training[['quality']]


x_train, x_test, y_train, y_test = train_test_split(stats, qual, test_size = 0.25, random_state = 2021)



choice = input('choose means of prediction:\n1) Logistic Regression\n2) Decision Tree\n3) Random Forest\n')
if choice=='1':
    #--------Logistic Regression------------

    scal = StandardScaler()

    x_train = scal.fit_transform(x_train)
    x_test = scal.fit_transform(x_test)

    model = LogisticRegression()
    model.fit(x_train, y_train)
    y_pred_logisticreg = model.predict(x_test)

    print("MSE: ", mean_squared_error(y_test, y_pred_logisticreg))

    print("Testing validation :", model.score(x_test, y_test))

    print(classification_report(y_test, y_pred_logisticreg))

    print(confusion_matrix(y_test, y_pred_logisticreg))

    print(y_pred_logisticreg)



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

    model = DecisionTreeClassifier()
    model = model.fit(x_train, y_train)
    y_pred_dectree = model.predict(x_test)

    print("MSE: ", mean_squared_error(y_test, y_pred_dectree))

    print("Testing validation :", model.score(x_test, y_test))

    print(classification_report(y_test, y_pred_dectree))

    print(confusion_matrix(y_test, y_pred_dectree))

    print(y_pred_dectree)

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

    scal = StandardScaler()

    x_train = scal.fit_transform(x_train)
    x_test = scal.fit_transform(x_test)

    estim = int(input('Give number of estimators:\n'))

    model = RandomForestClassifier(n_estimators=estim)
    model.fit(x_train, y_train)
    y_pred_ranfor = model.predict(x_test)


    print("MSE: ", mean_squared_error(y_test, y_pred_ranfor))

    print("Testing validation :", model.score(x_test, y_test))

    print(classification_report(y_test, y_pred_ranfor))

    print(confusion_matrix(y_test, y_pred_ranfor))

    print(y_pred_ranfor)

    x_train_new = training.drop(columns=['quality'])

    # load wine quality
    y_train_new = training[['quality']]

    # load test set
    x_test_new = testing

    model1 = RandomForestClassifier(n_estimators=estim)
    model1.fit(x_train_new, y_train_new)
    y_pred_ranfor_new = model1.predict(x_test_new)

    print(y_pred_ranfor_new)
    plt.figure()
    plt.hist(y_pred_ranfor_new, bins=50)
    plt.axvline(0, linestyle='--', color='r', label='lower boundary of the accepted ratings')
    plt.axvline(10, linestyle='--', color='b', label='upper boundary of the accepted ratings')
    plt.title("Distribution of wine quality based on Decision Tree Algorithm")
    plt.legend()
    plt.tight_layout()
    plt.show()
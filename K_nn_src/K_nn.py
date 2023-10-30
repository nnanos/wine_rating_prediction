import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier



#loading data phase ---------------------------------------------------------------------------



#loading train data points 
df_train = pd.read_excel ('Wine_Training.xlsx')
#converting train dataframe to numpy array
train_dataset = df_train.to_numpy()

#loading test input vectors
df_test =  pd.read_excel ('Wine_Testing.xlsx')
unknown_inputs = df_test.to_numpy()


#A contains the inputs to our model as row vectors (each row vector represents the features of a specific wine)
A = train_dataset[:,:train_dataset.shape[1]-1]
#b is a column vector that contains the labeled outputs for each input row vector (these are the wine ratings) 
b = train_dataset[:,train_dataset.shape[1]-1]

#We are using k-fold (k=5) cross validation to test each algorithm
#at this process we split the training dataset k times in different ways and train the model k times 
# and each time evaluating with the test set that is created





#see the problem as a classification one 

#αλγόριθμος που υλοποιήθηκε (παρόμοιος με τον k-nn):

#A*unknown_input = similarity_vector  (πέρνουμε όλα τα εσωτερικά γινόμενα της άγνωστης εισόδου με όλα τα διανύσματα του training set)

#sortaroume το similarity_vector και πέρνουμε τα top k όμοια (μεγαλύτερες τιμές similarity) indices με αποτέλεσμα να μπορούμε να πάρουμε και τα αντίστοιχα ratings

#Πέρνω το πιο συχνά εμφανιζόμενο rating από τα προηγούμενα (αυτό αποτελεί και την κλάση που προβλέφθηκε)

#Κάνω όλα τα προηγούμενα βήματα για όλες τις άγνωστες εισόδους . 

#K-nearest neighbors algorithm examination (custom algorithm)-------------------------------------------------
def k_nn_custom(k,test_input,train_data,labeled_outputs):

    #test_input = test_input/np.linalg.norm(test_input)
    
    #get all the dot products 
    all_similarities_distanses = np.matmul(train_data,test_input.T)

    #get the top k indices that gave the highest dot product (if we normalized the vectors the dot becomes cosine similarity [-1,1])
    ascending_order_indices =  np.argsort( all_similarities_distanses ) 

    #get the ratings of the top k nearest neighbors
    top_k_inds = ascending_order_indices[range(len(ascending_order_indices)-1,(len(ascending_order_indices)-1)-k,-1)]
    top_k_classes = labeled_outputs[top_k_inds]

    #get the most frequent class
    classes , occurences = np.unique(top_k_classes,return_counts=True)
    #the maximum occurence , this is the class prediction
    ind = np.argmax(occurences)
    class_prediction = classes[ind]


    return  class_prediction
#----------------------------------------------------------------------------------------------------------------

#input from the user the k parameter
k = int(input("Give the desired k \n:"))

#PREPROCESSING----------------------------------------------------------------------------


#standardizing the input (shifting the empirical mean of each column to zero and then normalizing by its empirical variance  ) 
# (assuming the input data is following a gaussian distribution )
'''
for i in range(A.shape[1]):

    m = np.mean(A[:,i])
    std = np.std(A[:,i])

    A[:,i] = ( A[:,i] - m ) / std
#------------------------------------------------------------------------------------------------------------------------------


'''


#normalize by its norm every row (known inputs) of A----------------------------------
for i in range(A.shape[0]):
    A[i,:] = A[i,:]/np.sqrt(np.dot(A[i,:],A[i,:]))

#normalize by its norm every row of unknown inputs 
for i in range(unknown_inputs.shape[0]):
    unknown_inputs[i,:] = unknown_inputs[i,:]/np.sqrt(np.dot(unknown_inputs[i,:],unknown_inputs[i,:]))
#-------------------------------------------------------------------------------------


#validating the algorithm---------------------------------------------------------------------

#using k-fold cross validation to test the algorithm
kf = KFold(n_splits=5)
KFold(n_splits=5, random_state=None, shuffle=False)
#k = kf.get_n_splits()

metrics_for_each_fold = []
for train_index, test_index in kf.split(A):

    A_train , A_test = A[train_index], A[test_index]
    b_train, b_test = b[train_index], b[test_index]

    #evaluation of the custom algorithm-------------------------------------------------------------------------------------
    #make predictions for the A_test input and then evaluating the cost function (mse) .We can do this because we know the answer b_test
    predictions = []
    for test_in in A_test:
        predictions.append( k_nn_custom(k,test_in,A_train,b_train) )
    
    #calculate the mse and the accuracy of the current fold 
    error = b_test-predictions
    
    #computing accuracy of the classification in the validation set
    number_of_correct_preds = np.count_nonzero(error==0)
    tmp_accuracy_custom = number_of_correct_preds/len(b_test)  
    #tmp_accuracy_custom = accuracy_score( b_test ,  predictions )   

    #computing mse as (eTe)/N
    tmp_mse_custom = np.dot(error,error)/len(b_test)
    #-------------------------------------------------------------------------------------------------------------------

    #evaluation of the built-in algorithm-------------------------------------------------------------------------------------
    #comparing with the builtin function of sklearn
    neigh = KNeighborsClassifier(n_neighbors=k)

    neigh.fit(A_train, b_train)

    predictions_sklearn = neigh.predict(A_test)

    #calculate the mse of the current fold
    error = b_test-predictions_sklearn

    #computing accuracy of the classification in the validation set
    number_of_correct_preds = np.count_nonzero(error==0)
    tmp_accuracy_sklearn = number_of_correct_preds/len(b_test)
    #tmp_accuracy_sklearn = accuracy_score( b_test ,  predictions_sklearn )  

    #computing mse as (eTe)/N
    tmp_mse_sklearn = np.dot(error,error)/len(b_test)
    #-----------------------------------------------------------------------------------------------------

    metrics_for_each_fold.append( [tmp_mse_custom,tmp_accuracy_custom,tmp_mse_sklearn,tmp_accuracy_sklearn] )




#obtaining a mean mse (and accuracy) from all the folds
metrics_for_each_fold = np.array(metrics_for_each_fold)
mean_validation_metrics = np.mean(metrics_for_each_fold, axis=0)

print('The mse and accuracy obtained from the average of all the fold validations (for the two algorithms) is:\n\nMSE:\n ',mean_validation_metrics[0],'(custom k-nn)\n',mean_validation_metrics[2],'(sklearn k-nn)\n\nAccuracy:\n',mean_validation_metrics[1],'(custom k-nn)\n',mean_validation_metrics[3],'(sklearn k-nn)')



#---------------------------------------------------------------------------------------------



#Obtaining all the predictions for all the unknown inputs (TESTING phase)------------------------------
predictions = []
for test_in in unknown_inputs:

    predictions.append( k_nn_custom(k,test_in,A,b) )


#comparing with the builtin function of sklearn
neigh = KNeighborsClassifier(n_neighbors=k)

neigh.fit(A, b)

predictions_sklearn = neigh.predict(unknown_inputs)

#plotting the dynamic range of the output response to the unknown inputs (sklearn case)
plt.figure()
plt.hist(predictions_sklearn , bins = 50)
plt.axvline(0, linestyle='--', color='r',label='lower boundary of the accepted ratings')
plt.axvline(10, linestyle='--', color='b',label='upper boundary of the accepted ratings')
plt.title("dynamic range of the predicted ratings (sklearn knn)")
plt.legend()
plt.tight_layout()


#plotting the dynamic range of the output response to the unknown inputs (custrom case)
plt.figure()
plt.hist(predictions , bins = 50)
plt.axvline(0, linestyle='--', color='r',label='lower boundary of the accepted ratings')
plt.axvline(10, linestyle='--', color='b',label='upper boundary of the accepted ratings')
plt.title("dynamic range of the predicted ratings (custom knn)")
plt.legend()
plt.tight_layout()
plt.show() 
#---------------------------------------------------------------------------------------------------------- 

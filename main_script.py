
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import scipy
import torch
#import perceptron_model
import perceptron_model_test
import sklearn
from sklearn.preprocessing import MinMaxScaler
from pytorchtools import EarlyStopping
from sklearn.linear_model import LinearRegression
#import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#some comments for the problem : All the following methods that we are going to examine are probabilistic (we are 
# trying to estimate the parameteres θ of a predictor) . We could say that the predictor represents a transformation (or a function)
#of a multivariate random variable -> (features of a wine) to a scalar random variable -> (rating of that wine) . We do not assume 
#any type of pdf that our data follows . But we asume that the system is static because the output depends only on the current input and not on any 
#previous inputs (and it is probably a non linear system because if we put as an input a wine that it's features are a linear combination of 2 others 
# then clearly the output rating is not a linear combination of the two individual ratings (response at each wine)..   ) 


#loading data phase ---------------------------------------------------------------------------

#loading train data points 
df_train = pd.read_excel ('/home/nnanos/Documents/ΜΑΘΗΜΑΤΑ/ΘΕΩΡΙΑ_ΑΠΟΦΑΣΕΩΝ/code/Wine_Training.xlsx')
#converting train dataframe to numpy array
train_dataset = df_train.to_numpy()

#loading test input vectors
df_test =  pd.read_excel ('/home/nnanos/Documents/ΜΑΘΗΜΑΤΑ/ΘΕΩΡΙΑ_ΑΠΟΦΑΣΕΩΝ/code/Wine_Testing.xlsx')
unknown_inputs = df_test.to_numpy()


#A contains the inputs to our model as row vectors (each row vector represents the features of a specific wine)
A = train_dataset[:,:train_dataset.shape[1]-1]
#b is a column vector that contains the labeled outputs for each input row vector (these are the wine ratings) 
b = train_dataset[:,train_dataset.shape[1]-1]

#We are using k-fold (k=5) cross validation to test each algorithm
#at this process we split the training dataset k times in different ways and train the model k times 
# and each time evaluating with the test set that is created


#I have checked for multicolinearity which in our design matrix A doesnt exist
#multicolinearity is the phenomenon where two or more predictors (explanatory variables (columns)) are linearly dependent or highily corellated
#where in this case we cannot invert the moment matrix A^TA to get the least squares fit (so it is connected with the numerical stability or condition of A)
#If we had multicolinearity then we would have to use REGULARIZATION of the ill-posed problem (Ridge Regression L2 or Lasso L1 )

#----------------------------------------------------------------------------------------------


#VARIOUS METHODs EXAMINATION-----------------------------------

l = int(input("1->Linear Regression\n2->Feedforward Neural Net (one hidden layer)\n3->Custom K-nn\n:"))


#Linear Regression-------------------------------------------------------------------------------------------------------------------
if (l == 1) :
    #Here we examine the multiple linear regression method (multiple because we have 10 independent variable and 1 dependent)
    #we use least squares aproach to fit the model (closed form solution x = (A^TA)^-1A^Tb projection of b to the subspace spanned by the collumns of A) 
    # (minimizes the L2 norm of the residual or the MSE (e=b-Ax where x are the parameters of the model))--------

    '''
    #normalizing each input vector (rows) (COMMENT : eventually it does not have any effect)
    for i in range(A.shape[1]):
        A[:,i] = ( A[:,i] - np.min(A[:,i]) ) / ( np.max(A[:,i]) - np.min(A[:,i]) ) 
    '''

    '''
    #standardizing the input (shifting the empirical mean of each column to zero and then normalizing by its empirical variance  ) 
    # (assuming the input data is following a gaussian distribution )
    #We seen that it does the same thing with the built-in function from sklearn
    for i in range(A.shape[1]):

        m = np.mean(A[:,i])
        std = np.std(A[:,i])

        A[:,i] = ( A[:,i] - m ) / std

        #A[:,i] = ( A[:,i] - m )
        #A[:,i] = A[:,i]/std
       
    '''

    '''
    A_stand_sklearn = []
    #standardizing with the built in function of sklearn to compare with the custom
    for i in range(A.shape[1]):
    
        # fit on training data column
        scale = StandardScaler().fit(A[:,[i]])
        
        # transform the training data column
        A_stand_sklearn.append(scale.transform(A[:,[i]]))
    '''

    #concatenating a ones vector (adding another deegre of freedom) to capture the case that our unwknown linear function 
    # (the hyperplane) is shifted from the origin 
    A = np.concatenate((A,np.array([np.ones(A.shape[0])]).T) ,axis=1 )
    #concatenating a ones vector 
    unknown_inputs = np.concatenate((unknown_inputs,np.array([np.ones(unknown_inputs.shape[0])]).T) ,axis=1 )

    #using k-fold cross validation to test the algorithm
    kf = KFold(n_splits=5)
    KFold(n_splits=5, random_state=None, shuffle=False)
    k = kf.get_n_splits()

    valid_loss_for_each_fold = []
    for train_index, valid_index in kf.split(A):

        A_train , A_valid = A[train_index], A[valid_index]
        b_train, b_valid = b[train_index], b[valid_index]

        #train phase extracting the linear model parameters x-------
        res = np.linalg.lstsq(A_train, b_train )
        x = res[0]
        #---------------------------------------------------

        #validating a fold----------------------------------------------------------
        b_pred = np.matmul(A_valid,x) #obtaining the predictions with an MV
        tmp_mse = mean_squared_error(b_valid, b_pred) #obtaining the mse 
        #---------------------------------------------------------------------------

        valid_loss_for_each_fold.append(tmp_mse)

    #obtaining a mean mse from all the folds
    mean_test_mse = np.mean(valid_loss_for_each_fold)

    print('The mse obtained from the average of all the fold validations is: ',mean_test_mse)


    #training the algorithm with all the training set A
    res = np.linalg.lstsq(A, b, rcond=None)
    x = res[0] 

    #checking the outputs to the unknown inputs and plotting the histogram of the dynamic range of the output ratings
    b_pred = np.matmul(unknown_inputs,x) #obtaining the predictions with an MV
    plt.figure()
    plt.hist(b_pred , bins = 50)
    plt.axvline(0, linestyle='--', color='r',label='lower boundary of the accepted ratings')
    plt.axvline(10, linestyle='--', color='b',label='upper boundary of the accepted ratings')
    plt.title("dynamic range of the predicted ratings")
    plt.legend()
    plt.tight_layout()
    plt.show() 
    #------------------------------------------------------------ 

    
    #Performing multiple linear regresion with the built-in function of the library sklearn 
    #and comparing it with the custom one above
    '''
    model = LinearRegression().fit(A , b)
    r_sq = model.score(A, b)
    print('\n\n\ncoefficient of determination:', r_sq)
    '''

#---------------------------------------------------------------------------------------------------------------------------------


'''
#PCA (SVD)----------

#normalization
for i in range(A.shape[1]):

    m = np.mean(A[:,i])
    std = np.std(A[:,i])

    A[:,i] = ( A[:,i] - m ) / std

    #A[:,i] = ( A[:,i] - m )
    #A[:,i] = A[:,i]/std


#covariance matrix (empirical estimated by sample points)
cov = np.matmul(A.T,A)
cov = cov/A.shape[0]
#same as np.corecoef(A.T)

#eig decomposition
[L,Q] = np.linalg.eig(cov)
#sorting the eigenvalues and the corresponding eigenvectors
idx = L.argsort()[::-1]   
L = L[idx]
Q = Q[:,idx]

#Q contains the eigenvectors of the covariance matrix of our data
# and is a linear orthogonal transform made from the data itself (not fixed like fourier) 
#or we can say that these are the principal components 
#so we can do low rank aproximations of our data (keeping those eigenvectors that correspond to the highest eigenvalues)
#and this is a way of dimensionality reduction and feature extraction 


#projecting the data to the eigenspace (keeping some of the principal components or eigenvectors)
C = np.matmul(Q.T,A.T)

#Or equivalently we can do an SVD decomposition
U,S,V = np.linalg.svd(A,True)
tmp = np.zeros((A.shape[0]-11,11))
S_full = np.concatenate((np.diag(S),tmp),axis=1)
U @ S_full @ V

#------------------
'''

#Feedforward Neural Net (one hidden layer)-------------------------------------------------------------------------------------------
if (l == 2) :
    '''
    #EXPERIMENTING WITH THE EFFECT OF SCALING (preprocessing of the input data) ON THE PERFORMANCE OF THE ALGORITHM-----------------
    
    #normalizing each input vector (rows) (COMMENT : eventually it does not have any effect)
    for i in range(A.shape[1]):
        A[:,i] = A[:,i]/np.sqrt(np.dot(A[:,i],A[:,i]))
    '''

    
    #standardizing the input (shifting the empirical mean of each column to zero and then normalizing by its empirical variance  ) 
    # (assuming the input data is following a gaussian distribution )
    #IT DOESNT WORK FOR LINEAR REGRESSION
    for i in range(A.shape[1]):

        m = np.mean(A[:,i])
        std = np.std(A[:,i])

        A[:,i] = ( A[:,i] - m ) / std
    

    '''
    #scaling between zero and one (using min max scaler) 
    scaler = MinMaxScaler()
    scaler.fit(A)
    A = scaler.transform(A)
    '''
    
    #-----------------------------------------------------------------------------------------------------------------------------

    def create_datasets(batch_size,train_data,test_data):

        # percentage of training set to use as validation
        valid_size = 0.2

        # convert data to torch.FloatTensor
        #transform = transforms.ToTensor()

        # obtain training indices that will be used for validation
        num_train = len(train_data)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        split = int(np.floor(valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]
        
        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        
        # load training data in batches
        train_loader = torch.utils.data.DataLoader(train_data,
                                                batch_size=batch_size,
                                                sampler=train_sampler,
                                                num_workers=0)
        
        # load validation data 
        valid_loader = torch.utils.data.DataLoader(train_data,
                                                #we set the batch size equal to the number of examples that we will feedforward (because we are at evaluation phase)
                                                batch_size=len(valid_index),
                                                sampler = valid_sampler,
                                                num_workers=0)
        
        # load test data in batches
        test_loader = torch.utils.data.DataLoader(test_data,
                                                batch_size=batch_size,
                                                num_workers=0)
        
        
        return train_loader,  valid_loader , test_loader


    #NEURAL NET PARAMETERS--------------------------------------------
    #we examine two types of models (in one we see the problem as a regression task and in the other as a classification one)
    

    model_type = str(input("Give model type to examine (options: regression || classification)\n:"))

    input_size = A.shape[1]
    hidden_layer_size = int(input('Give the number of hiiden nodes\n:'))
    if model_type == 'regression':
        output_size = 1
    else:
        output_size = 10
    
    n_epochs = 100
    batch_size = int(input('Give the batch_size\n:'))   #if batch size = number of all the data examples (sample size) then batch learning (μαζική μάθηση) +εγγυημένη σύγκλιση σε κάποιο τοπικό ελάχιστο -πολύς χώρος
                              #if batch size = 1 then stochastic gradient decent ή LMS (on-line μάθηση ) +μείωση πιθανότητας να παγιδευτεί σε τοπικό ελάχιστο,λιγότερος χώρος

    learning_rate = float(input('Give the learning_rate\n:'))   #0.001 for regression 
    # early stopping patience; how many epochs to wait after last time validation loss improved.
    patience = 10
    #--------------------------------------------------------------------------    

    
    #using k-fold cross validation to evaluate the algorithm---------------------------------------------------------
    kf = KFold(n_splits=5)
    KFold(n_splits=5, random_state=None, shuffle=True)
    k = kf.get_n_splits()

    train_dataset = torch.FloatTensor(train_dataset)

    valid_loss_for_each_fold = []
    train_loss_for_each_fold = []
    accuracy_for_each_fold = []
    for train_index, valid_index in kf.split(train_dataset):

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_index)
        valid_sampler = SubsetRandomSampler(valid_index)
        valid_dataset = train_dataset[valid_index,:]
        
        # load training data in batches
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=batch_size,
                                                sampler=train_sampler,
                                                num_workers=0)
        
        # load validation data in batches
        valid_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=len(valid_index),
                                                sampler = valid_sampler,
                                                num_workers=0)


        model, train_loss, valid_loss , accuracy  = perceptron_model_test.train_and_validate_the_model(model_type,
                                                        input_size,
                                                        hidden_layer_size,
                                                        output_size,
                                                        n_epochs ,
                                                        learning_rate,
                                                        patience,
                                                        train_loader,   
                                                        valid_loader,
                                                        )


        #we only need the last epochs validation to obtain the mean validation loss over all folds (keep in mind that the validation procedyre is executed in a smal)
        valid_loss = valid_loss[len(valid_loss)-1]
        valid_loss_for_each_fold.append(valid_loss)

        #same for accuracy
        if model_type == "classification":
            accuracy_last_epoch = accuracy[len(accuracy)-1] 
            accuracy_for_each_fold.append(accuracy_last_epoch)

    #obtaining the mean validation_loss (whatever loss function used) over all the folds
    mean_valid_loss = np.mean(valid_loss_for_each_fold)
    mean_accuracy = np.mean(accuracy_for_each_fold)
    #---------------------------------------------------------------------------------------------------------------


    #NOT USING K-FOLD (to monitor the learning curve and to test the model to the unknown inputs)---------------------------------------------------------------------------------------------
    train_dataset = torch.FloatTensor(train_dataset)
    train_loader,  valid_loader ,test_loader = create_datasets(batch_size,train_dataset,unknown_inputs )


    model, train_loss, valid_loss , _ = perceptron_model_test.train_and_validate_the_model(model_type,
                                                    input_size,
                                                    hidden_layer_size,
                                                    output_size,
                                                    n_epochs ,
                                                    learning_rate,
                                                    patience,
                                                    train_loader,   
                                                    valid_loader,
                                                    )

    
    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10,8))
    plt.plot(np.arange(len(train_loss))-0.5,train_loss, label='Training Loss')
    plt.plot(range(0,len(valid_loss)),valid_loss,label='Validation Loss')

    # find position of lowest validation loss
    minposs = valid_loss.index(min(valid_loss))+1 
    plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    if model_type=='regression':
        plt.ylabel('MSE')
        print('The average mse obtained over all the fold validations (last epoch) is:\n ' , mean_valid_loss)
    else:
        plt.ylabel('Cross Entropy loss')
        print('The average cross entropy loss and accuracy obtained over all the fold validations (last epoch)  is:\n\n ' , 'loss=', mean_valid_loss,'\naccuracy=',mean_accuracy)
    #plt.ylim(0, 0.5) # consistent scale
    #plt.xlim(0, len(train_loss)+1) # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    ##checking the outputs to the unknown inputs-------------------
    x = torch.FloatTensor(unknown_inputs)
    model.eval()
    b_pred = model(x)
    #ploting histogram to see the range of the response values (ratings) of the system
    if model_type=="classification":
        _, b_pred = b_pred.max(1)
        b_pred = b_pred.detach().numpy()
    else:
        b_pred = b_pred.detach().numpy()

    plt.figure()
    plt.hist(b_pred , bins = 50)
    plt.axvline(0, linestyle='--', color='r',label='lower boundary of the accepted ratings')
    plt.axvline(10, linestyle='--', color='b',label='upper boundary of the accepted ratings')
    plt.title("dynamic range of the predicted ratings")
    plt.legend()
    plt.tight_layout()
    #------------------------------------------------------------ 

    plt.show() 

    #-----------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------

if l==3:
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

    '''
    #normalize by its norm every row (known inputs) of A----------------------------------
    for i in range(A.shape[0]):
        A[i,:] = A[i,:]/np.sqrt(np.dot(A[i,:],A[i,:]))

    #normalize by its norm every row of unknown inputs 
    for i in range(unknown_inputs.shape[0]):
        unknown_inputs[i,:] = unknown_inputs[i,:]/np.sqrt(np.dot(unknown_inputs[i,:],unknown_inputs[i,:]))
    #-------------------------------------------------------------------------------------
    '''
    
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



#ADDING COMMAND LINE ARGUMENTS

a = 0
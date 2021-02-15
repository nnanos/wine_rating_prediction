
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

#----------------------------------------------------------------------------------------------


#VARIOUS METHODs EXAMINATION-----------------------------------

l = int(input("1->Linear Regression\n2->Feedforward Neural Net (one hidden layer)\n:"))


#Linear Regression-------------------------------------------------------------------------------------------------------------------
if (l == 1) :
    #Here we examine the multiple linear regression method (multiple because we have 10 independent variable and 1 dependent)
    #we use least squares aproach to fit the model (closed form solution x = (A^TA)^-1A^Tb projection of b to the subspace spanned by the collumns of A) 
    # (minimizes the L2 norm of the residual or the MSE (e=b-Ax where x are the parameters of the model))--------

    '''
    #normalizing each input vector (rows) (COMMENT : eventually it does not have any effect)
    for i in range(A.shape[1]-1):
        A[:,i] = A[:,i]/np.sqrt(np.dot(A[:,i],A[:,i]))
    '''

    '''
    #standardizing the input (shifting the empirical mean of each column to zero and then normalizing by its empirical variance  ) 
    # (assuming the input data is following a gaussian distribution )
    for i in range(A.shape[1]-1):

        m = np.mean(A[:,i])
        std = np.std(A[:,i])

        A[:,i] = ( A[:,i] - m ) / std

        #A[:,i] = ( A[:,i] - m )
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

    evaluation_for_each_fold = []

    for train_index, test_index in kf.split(A):

        A_train , A_test = A[train_index], A[test_index]
        b_train, b_test = b[train_index], b[test_index]

        #train phase extracting the linear model parameters x-------
        res = np.linalg.lstsq(A_train, b_train )
        x = res[0]
        #---------------------------------------------------

        #validating a fold----------------------------------------------------------
        b_pred = np.matmul(A_test,x) #obtaining the predictions with an MV
        tmp_mse = mean_squared_error(b_test, b_pred) #obtaining the mse 
        #---------------------------------------------------------------------------

        evaluation_for_each_fold.append(tmp_mse)

    #obtaining a mean mse from all the folds
    mean_test_mse = np.mean(evaluation_for_each_fold)

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



#PCA (SVD)----------

#covariance matrix (empirical estimated by sample points)
cov = np.matmul(A.T,A)
cov = cov/A.shape[0]

#------------------

#Feedforward Neural Net (one hidden layer)-------------------------------------------------------------------------------------------
if (l == 2) :
    
    #EXPERIMENTING WITH THE EFFECT OF SCALING (preprocessing of the input data) ON THE PERFORMANCE OF THE ALGORITHM-----------------
    '''
    #normalizing each input vector (rows) (COMMENT : eventually it does not have any effect)
    for i in range(A.shape[1]-1):
        A[:,i] = A[:,i]/np.sqrt(np.dot(A[:,i],A[:,i]))
    '''

        
    #standardizing the input (shifting the empirical mean of each column to zero and then normalizing by its empirical variance  ) 
    # (assuming the input data is following a gaussian distribution )
    #IT DOESNT WORK FOR LINEAR REGRESSION
    for i in range(A.shape[1]-1):

        m = np.mean(A[:,i])
        std = np.std(A[:,i])

        A[:,i] = ( A[:,i] - m ) / std
    '''

    '''
    #scaling between zero and one (using min max scaler) 
    scaler = MinMaxScaler()
    scaler.fit(A)
    A = scaler.transform(A)
    
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
        
        # load validation data in batches
        valid_loader = torch.utils.data.DataLoader(train_data,
                                                batch_size=batch_size,
                                                sampler=valid_sampler,
                                                num_workers=0)
        
        # load test data in batches
        test_loader = torch.utils.data.DataLoader(test_data,
                                                batch_size=batch_size,
                                                num_workers=0)
        
        return train_loader, test_loader, valid_loader


    #NEURAL NET PARAMETERS--------------------------------------------
    input_size = A.shape[1]
    hidden_layer_size = 5
    output_size = 1
    n_epochs = 100
    batch_size = 100
    learning_rate = 0.02
    # early stopping patience; how long to wait after last time validation loss improved.
    patience = 3
    #--------------------------------------------------------------------------

    #NOT USING K-FOLD ---------------------------------------------------------------------------------------------

    train_dataset = torch.FloatTensor(train_dataset)
    train_loader, test_loader, valid_loader = create_datasets(batch_size,train_dataset,unknown_inputs)

    model, train_loss, valid_loss = perceptron_model_test.train_the_model(input_size,
                                                    hidden_layer_size,
                                                    output_size,
                                                    n_epochs ,
                                                    learning_rate,
                                                    patience,
                                                    train_loader,   
                                                    valid_loader)

    
    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10,8))
    plt.plot(np.arange(len(train_loss))-0.5,train_loss, label='Training Loss')
    plt.plot(range(0,len(valid_loss)),valid_loss,label='Validation Loss')

    # find position of lowest validation loss
    minposs = valid_loss.index(min(valid_loss))+1 
    plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('MSE')
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
    


    '''
    #using k-fold cross validation to test the algorithm---------------------------------------------------------
    kf = KFold(n_splits=5)
    KFold(n_splits=5, random_state=None, shuffle=False)
    k = kf.get_n_splits()

    evaluation_for_each_fold = []

    for train_index, test_index in kf.split(A):

        A_train , A_test = A[train_index], A[test_index]
        b_train, b_test = b[train_index], b[test_index]

        #spliting the training dataset to a train and validation set
        X_train, X_valid, y_train, y_valid = sklearn.model_selection.train_test_split(A_train , b_train, test_size=0.1, random_state=1)

        #initializing the tensors ( for training and validation )  
        data_train = torch.tensor(X_train, dtype=torch.float64 , requires_grad=True)
        target_train = torch.tensor(y_train, dtype=torch.float64 , requires_grad=True)

        data_valid = torch.tensor(X_valid, dtype=torch.float64 , requires_grad=True)
        target_valid = torch.tensor(y_valid, dtype=torch.float64 , requires_grad=True)

        A_test = torch.FloatTensor(A_test)
        b_test = torch.FloatTensor(b_test)

        model, train_loss, valid_loss = perceptron_model.train_the_model(input_size,hidden_layer_size,
                                                                        output_size,
                                                                        n_epochs ,
                                                                        learning_rate,
                                                                        patience,
                                                                        data_train,
                                                                        target_train,
                                                                        data_valid,
                                                                        target_valid)


        #validating the current fold----------------------------------------------------------
        model.eval()
        y_pred = model(A_test) #obtaining the predictions with an MV
        tmp_mse = mean_squared_error(b_test.detach().numpy(), y_pred.detach().numpy()) #obtaining the mse 
        #---------------------------------------------------------------------------

        evaluation_for_each_fold.append(tmp_mse)

    #obtaining a mean mse for the testing phase from all the folds
    mean_test_mse = np.mean(evaluation_for_each_fold)
    #---------------------------------------------------------------------------------------------------------------
    '''

    '''
    #ploting MSE per epoch (to see how the training phase goes)
    plt.plot(loss_values)
    plt.xlabel('epochs')
    plt.ylabel('MSE')
    plt.show()

    ##checking the outputs to the unknown inputs-------------------
    x = torch.FloatTensor(unknown_inputs)
    model.eval()
    b_pred = model(x)
    

    #if the maximum response of the system is greater than 10 (maximum possibility for a rating )  
    # then the output is possibly garbage (or the preprocessing didnt work) 
    print(torch.max(b_pred))
    '''

    #------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------

a = 0
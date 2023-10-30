import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error




#some comments for the problem : All the following methods that we are going to examine are probabilistic (we are 
# trying to estimate the parameteres θ of a predictor) . We could say that the predictor represents a transformation (or a function)
#of a multivariate random variable -> (features of a wine) to a scalar random variable -> (rating of that wine) . We do not assume 
#any type of pdf that our data follows . But we asume that the system is static because the output depends only on the current input and not on any 
#previous inputs (and it is probably a non linear system because if we put as an input a wine that it's features are a linear combination of 2 others 
# then clearly the output rating is not a linear combination of the two individual ratings (response at each wine)..   ) 


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
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorchtools import EarlyStopping
import numpy as np

#feedforward network model with one hidden layer if we see the problem as a regression one
class FeedforwardNeuralNetModel_regression(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModel_regression, self).__init__()
        # Linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        #initializing the weights of the linear mapping (weight matrix)
        self.fc1.weight.data.fill_(0.01)
 

        # Non-linearity
        self.relu = nn.ReLU()

        # Linear function (readout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        #initializing the weights of the linear mapping (weight matrix)
        self.fc2.weight.data.fill_(0.01)  

    def forward(self, x):
        # Linear function  # LINEAR
        out = self.fc1(x)

        # Non-linearity  # NON-LINEAR
        out = self.relu(out)

        # Linear function (readout)  # LINEAR
        out = self.fc2(out)

        #forcing the output to be positive with the relu
        #out = self.relu(out)

        return out


#feedforward network model with one hidden layer if we see the problem as a classification one
class FeedforwardNeuralNetModel_classification(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModel_classification, self).__init__()
        # Linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        #initializing the weights of the linear mapping (weight matrix)
        self.fc1.weight.data.fill_(0.01)
 

        # Non-linearities
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.log_softmax = nn.LogSoftmax()
        self.relu = nn.ReLU()

        # Linear function (readout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        #initializing the weights of the linear mapping (weight matrix)
        self.fc2.weight.data.fill_(0.01)  

    def forward(self, x):
        # Linear function  # LINEAR
        out = self.fc1(x)

        # Non-linearity  # NON-LINEAR
        out = self.relu(out)

        # Linear function (readout)  # LINEAR
        out = self.fc2(out)

        #we doesent need to apply a softmax to create the distribution (we just take as the predicted class the highest response of the network )
        #the cross_entropy_loss does that by default to compute the loss 
        #out = self.softmax(out)

        return out



def train_and_validate_the_model(model_type,
                    input_size,
                    hidden_layer_size,
                    output_size,
                    n_epochs ,
                    learning_rate,
                    patience,
                    train_loader,
                    valid_loader):

        # to track the training loss as the model trains
        train_losses = []
        # to track the validation loss as the model trains
        valid_losses = []
        # to track the average training loss per epoch as the model trains
        avg_train_losses = []
        # to track the average validation loss per epoch as the model trains
        avg_valid_losses = []

        #to track the accuracy (only for classification type models)
        accuracy = []

        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=patience, verbose=True) #https://github.com/Bjarten/early-stopping-pytorch

        #instantiate a dense feedforward model , cost function to minimize and an optimizer
        if model_type=='regression':
            model = FeedforwardNeuralNetModel_regression(input_size , hidden_layer_size , output_size )
            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
        else:
            model = FeedforwardNeuralNetModel_classification(input_size , hidden_layer_size , output_size )
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

        


        ######################    
        # train the model #
        ######################
        for epoch in range(1,n_epochs+1): 

            model.train() # prep model for training
            for batch , data in enumerate(train_loader, 1):

                input_data = data[:,:data.shape[1]-1]
                if model_type=='regression':
                    target = data[:,data.shape[1]-1].unsqueeze(0)
                else:
                    #the target needs to be a vector of size minibatch and each entry is and index to the true class in the range [0,C-1] 
                    # where is the number of classes
                    target = data[:,data.shape[1]-1] - 1
                    target = target.long()

                # clear the gradients of all optimized variables
                optimizer.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(input_data)   

                # calculate the loss for the current minibatch
                loss = criterion(output,target)
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                optimizer.step()
                # record training loss for the current minibatch
                train_losses.append(loss.item())

            ######################    
            # validate the model #
            ######################
            
            model.eval() # prep model for evaluation
            for data in valid_loader:

                input_data = data[:,:data.shape[1]-1]
                if model_type=='regression':
                    target = data[:,data.shape[1]-1].unsqueeze(0)
                else:
                    target = data[:,data.shape[1]-1] - 1
                    target = target.long()


                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(input_data)

                #compute accuracy on the validation set if model type is classification
                if model_type=='classification':
                    _, out = output.max(1)
                    correct = (out==target).sum().item()
                    total = len(target)
                    accuracy_tmp = correct/total
                    accuracy.append(accuracy_tmp)
                else:
                    accuracy = None

                # calculate the valid_loss for the current minibatch
                loss = criterion(output, target)
                # record validation valid_loss for the current minibatch
                valid_losses.append(loss.item())
            

            
            # print training/validation statistics 
            # calculate average loss over an epoch
            train_loss = np.average(train_losses) #contains the mean train_loss over all train_batches of the current epoch
            valid_loss = np.average(valid_losses) #contains the mean valid_loss over all valid_batches of the current epoch
            avg_train_losses.append(train_loss) #contains the learning curve for all the epochs
            avg_valid_losses.append(valid_loss) #contains the validation curve for all the epochs
            
            epoch_len = len(str(n_epochs))
            
            print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                        f'train_loss: {train_loss:.5f} ' +
                        f'valid_loss: {valid_loss:.5f}')

                    
            print(print_msg)
        
            # clear lists to track next epoch
            train_losses = []
            valid_losses = []
            
            # early_stopping needs the validation loss to check if it has decresed, 
            # and if it has, it will make a checkpoint of the current model
            early_stopping(valid_loss, model)
            
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # load the last checkpoint with the best model
            #model.load_state_dict(torch.load('checkpoint.pt'))

            
        return model, avg_train_losses, avg_valid_losses , accuracy
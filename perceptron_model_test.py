import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorchtools import EarlyStopping
import numpy as np

#feedforward network model with one hidden layer
class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()
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
        out = self.relu(out)

        return out


def train_the_model(input_size,
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

        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=patience, verbose=True) #https://github.com/Bjarten/early-stopping-pytorch

        #instantiate a dense feedforward model , cost function to minimize and an optimizer
        model = FeedforwardNeuralNetModel(input_size , hidden_layer_size , output_size )
        criterion = torch.nn.MSELoss()
        #optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
        #criterion = torch.nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)



        for epoch in range(1,n_epochs+1): 

            model.train() # prep model for training
            for batch , data in enumerate(train_loader, 1):

                input_data = data[:,:data.shape[1]-1]
                target = data[:,data.shape[1]-1].unsqueeze(0)

                # clear the gradients of all optimized variables
                optimizer.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(input_data)
                # calculate the loss
                loss = criterion(output,target)
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                optimizer.step()
                # record training loss
                train_losses.append(loss.item())

            ######################    
            # validate the model #
            ######################
            
            model.eval() # prep model for evaluation
            for data in valid_loader:

                input_data = data[:,:data.shape[1]-1]
                target = data[:,data.shape[1]-1].unsqueeze(0)

                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(input_data)
                # calculate the loss
                loss = criterion(output, target)
                # record validation loss
                valid_losses.append(loss.item())
            

            
            # print training/validation statistics 
            # calculate average loss over an epoch
            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)
            
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
            model.load_state_dict(torch.load('checkpoint.pt'))

            
        return model, avg_train_losses, avg_valid_losses
=======================================================================
Wine rating prediction with Machine Learning
=======================================================================

Description
============

In this repository I do some experiments on email spam detection. 
The dataset that is being used is the spam_or_not_spam.csv file which contains two columns.
The first column includes the text from various emails while the second column informs us whether they were spam or not: value 1 for spam, 0 otherwise.
My goal is to try to guess the information in the second column using a neural network.
To approach this problem I am using Deep Learning and more specifically RNNs and LSTMs.


Module Description 
============


* **Load_data_functions.py**

In this module there are functions required to load the dataset and
preprocessing of it. The only pre-processing performed is that of conversion
of the words of each email in word embeddings. For the latter, two capabilities have been implemented
, either training a word2vec model on the existing dataset or
using the pre-trained google model (GoogleNews-vectors-negative300). The approach followed for the experiments was the first because:
1) I saw satisfactorily NN results and
2) because loading the google model into memory is costly as it is trained on a large volume of words (it is 3Gb) .The implemented functions simulate the following steps (which are executed
serially):
(*Note that everything is in memory because the dataset and word embeddings are
small)

	#.  **load_data()**
		Loads the given csv file into a dataframe. It then separates the emails from the
		respective labels and finally returns 2 lists. Each element of the first is an np
		array that contains the words (strs) of an email and the second contains the corresponding labels.

	#.  **get_embeded_emails()**
		Converts each email (sequence of strs) to a sequence of vectors (word
		embeddings). Now a list is returned, each element of which is a matrix of
		which represents an email (each vector is a word) .

	#.  **get_training_batches()**
		For each mini-batch of embedded emails, the emails are sorted by
		largest to smallest sequence , then zero padding all emails to
		length of the longest email of the mini-batch (because when training an RNN
		each batch sequence needs to be the same length (for efficient implementation
		of as MM multiplication) ) . So what is returned is a list of mini-batches
		each of which contains sequences (embedded emails) of the same length
		(we used packing).


* **Model.py**

The Model class represents the architecture of the network. Two possibilities were implemented,
a simple elman RNN and an LSTM which can be selected from input arguments.
In each case the output of the state of the last step of the recursive network
enters a linear level which implements an inner product.
Finally the neural net is trained ( evaluation mode) the response of the linear level which is a scalar becomes an input to a sigmoid non-linearity in order to model the probability that the email that processed the neural net to be spam or not. Whereas when the network is in training mode then
again the probability is modeled in the same way as before but now we don't have to pass
the response of the linear level from the sigmoid because this is done in the calculation of the
loss function ( BCEWithLogitsLoss() ) of pytorch. 


.. Image:: /Images/RNN_architecture.png


The image above shows details of the architecture of the implemented NNs
and specifically of vanilla RNN although LSTM follows the same mentality simply )
the cell becomes more complex so as to deal with i.e vanishing gradient problems
for long time series. I also list the objective used for
minimization.


* **train.py**

This module was implemented in order to train a recurrent neural network. The
basic logic is described in the following steps:

	#. Loading the data using the functions of the Load_data_functions.py module.

	#. Initialization of objects : Model,critireon,optimizer,scheduler

	#. Model training: If in the arguments has been given a Path for the checkpoint.pth of a pretrained network with simillar architecture then the training continues otherwise it starts from the beginning. In
	   each iteration of the training loop (epoch) the following functions in that order are executed:

		#. Forward pass in wich we feed the network with all training examples in mini-batches of 125 examples.

		#. Backward pass in wich the Backpropagation algorithm is executed. More specifically it is performed a gradient (of the loss function) calculation in terms of the network parameters 			   
		   and then the weights are updated with that gradient.

		#. Validation. After completing the previous two steps for all mini-batches
		   we feed the model with a very small subset of the dataset which is not used for the training phase and we calculate the error of the cost function ).
		   We use this method to do early stopping.

		#. Finally, I save some dicts which contain information about the models state
		   , the optimizer and the error of the epoch for trainning and validation.



* **Evaluation.py**
This module was implemented for the purpose of error analysis of training and the evaluation
of the algorithm on unknown data with the metrics **Recall**, **Precision** and **F1 score**.
So it is intended for execution after train.py has been executed which has produced the network checkpoint which contains the state dicts and json which contains the network training log.




Experiments
=============

First we have to note that word embeddings were not pre-processed (all that was done was
shuffle of emails initially) and that for all experiments (vanilla RNN k LSTM)
the following hyperparameters were used (with adam optimizer):

	* max-epochs = 140

	* learning-rate = 0.001

	* patience = 10 (For how many epochs to continue the training if the validation loss does not decrease further)

	* batch_size = 125

	* Dimensionality of word embeddings = 300

	* Dimensionality of state space = 128


In any case, the network parameters (weights) were frozen in the epoch with the best training and
validation loss (best epoch). In the images below I present the results for each case (RNN or LSTM):



* **vanilla RNN**

	
	* Model Architecture
	.. Image:: /Images/RNN/Model_Arch.png


	* Learning curve
	.. Image:: /Images/RNN/Learning_curve.png



	* Evaluation Metrics
	.. Image:: /Images/RNN/Metrics.png



* **LSTM**

	
	* Model Architecture
	.. Image:: /Images/LSTM/Model_arch.png
	
	
	* Learning curve
	.. Image:: /Images/LSTM/Learning_curve.png
	
	
	* Evaluation Metrics
	.. Image:: /Images/LSTM/Metrics.png


Reproduce the experiments
============

::

	pip install requierments.txt

	python train.py 
	“--Model_type” <RNN or LSTM> 
	“--output” <the folder you want to save log for training and checkpoint>
	“--model” <the path where the output folder of the pretrained model is located>
	“--root” <the path where the training dataset is loacted>
	(the last two args are optional in case you 1) want the training to continue and 2)
	to set another path for the dataset with emails)

	python evaluation.py
	“--root” <the path where the testing dataset is loacted>
	“--Model_type” <RNN or LSTM>
	“--model” <the path where the output folder of the previous command is located>





* the shuffled dataset (I suggest you use this dataset will also be the
  default path) so as to avoid another factor of randomness and to
  reproduce the results more correctly (it is the same as the original with the only
  difference that the emails have been shuffled and finally I have deleted some gaps that each email had at the beginning with sed).

* The output folders for LSTMs and RNNs created so that you don't have to
  perform step 2 (unless you want training to continue) just to see them
  learning curves and performance metrics.





Free software: MIT license
============


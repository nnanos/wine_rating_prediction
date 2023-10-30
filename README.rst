=======================================================================
Wine rating prediction with Machine Learning
=======================================================================

Description
============

The problem we face here is a very common one which is finding a function which 
describes a phenomenon where in our case given some characteristics of a wine 
(input) we need to find the corresponding rating (output) for this. We can say
for the sake of simplicity that the system which we are trying to model is static
because the output depends only on the current input and not on any previous inputs
but we cannot avoid its non inear nature because if we put as an input a wine that
its features are a linear combination of 2 others then clearly the output rating
is not a linear combination of the two individual ratings (response at each wine).


In the current problem will examine various supervised
(data-driven) machine learning algorithms. So we will approach the
problem with probabilistic methods where the input-output are modeled as 
random variables and our purpose now is to estimate the parameters of an
estimator (function or transformation of random variables) optimal in 
some sense (eg mean square error). Therefore the problem is reduced to an 
optimization problem where as known our goal is to minimize our cost function 
through the algorithm that we will use. Finally, it should be emphasized that 
the specific problem (and in general) we can see it either as a regression 
or as a classification. In the problem at hand it seems more natural to see it
as a classification one because we have to predict a rating which is in the 
range 0,1,...,10. The machine learning algorithms that we are going to examine 
are described below.







* K-nearest neighbor (custom)

The algorithm was implemented by us and is not exactly the python 
implementation (however the we compared with the built-in of sklearn).
It is described below:

	#. Normalize query_wine and each line (example) of train_data (divide by
       	   L2 norm of respectively) .

	#. Get all inner products of the query with the array train_data 
	   (a Matrix_vector Multiplication). The output vector consists of 
	   all similarity coeficients (range [-1,1]) of query_wine with all examples.

	#. We sort the vector with the resulting coefs and pass the indices of 
	   the top k responses (greater similarities of the similarity vector). 
	   Then we index the target_data where we pass the ratings of those
	   top k similar to query_wine .



	#. We get the most frequently displayed rating from the previous ones
	   (with the help of the histogram).

	
The main difference with K-nn is that the search of the top k similar vectors
is done based on the metric of the Euclidean distance (L2 norm) while we use
inner products. Also the crucial step is the 1st (since we use inner products) 
because in this way we become invariant to the scale of the vectors we compare 
(range response [-1,1]). Finally we should mention that we expect to get similar
results with the L2 distance too although now the range will be [0,2] (if
the vectors are normilized) and the smaller the better.


 
* Feedforward Nueral Net (multilayer perceptron with one hidden layer) :

Knowing that the problem has a non-linear nature we wanted to consider
algorithms as well such as Neural Nets which we know can deal with these
kinds of problems from the Universal approximation theorem. Therefore we will
try to train a multilayer perceptrons with one hidden layer. In general,
Neural Nets have some learnable parameters (weights) and some hyperparameters
(e.g. number of hidden layer nodes, learning step, batch size). Also it 
should be emphasized that obviously the solution (optimal parameters) which
minimize the cost function in terms of its parameters is not given by 
a closed solution (like e.g. linear regression). That is because by
introducing non-linearities (activation functions) the cost function
becomes non-convex and as a result it does not have a total minimum.
We must mention that the minimization of the cost function is done in
epochs ie is done with an iterative method (like gradient descent) while
the gradients are calculated with backpropagation algorithm. As a termination
criterion we define either a number of epochs or an early stopping procedure.
Finally, a Neural Net can be trained either "massively" (batch learning)
either "on-line" one example at a time or (intermediate solution)
in mini-batches.

**IMAGES**





EXPERIMENTS
=============


**Using K-fold cross validation:**

The k-fold cv technique is widely used in the evaluation of machine learning
algorithms (and especially supervised). More specifically what it does is 
iteratively split the initial train dataset (the one that also contains the
labeled outputs) in train and validation sets. In each iteration chooses a
point at which to split the dataset, then trains and validate the model on
the corresponding sets created and finally keeps the statistics for the 
training and validation of the current fold (repetition). The previous
procedure is repeated with the only difference that a different split point
is chosen each time as a result we check the algorithm for various pairs
of train and validation sets and so we get an average of these statistics
over the folds. Obviously, the selection of k plays a role here. 
As you increase it, the train set will increase in number and the validation
set the opposite.

(#train + #validation=#of total dataset for train)

**extreme cases :**

 * k=2 then:

   #train=#validation=#total / 2 

 * k=#total then:

   #train=#validation=0 





* K-nearest neighbor (custom)

For the evaluation of the algorithm, the Cross validation technique for k = 5
was used here as well metric the MSE. It should be noted that we performed the
same procedures (5-fold CV) for built in sklearn's algorithm 
(with the same parameter k=5 is meant) in which the metric that
used to determine the top k neighbors is the L2 distance.


#. Normalizing the data (making the vectors unitary)

	The mse and accuracy obtained from the average of all 
	the fold validations (for the two algorithms) is:

	**MSE:**
		0.8806938633193863 (custom k-nn)
		0.8806938633193863 (sklearn k-nn)

	**Accuracy:**
		0.48578451882845186 (custom k-nn)
		0.48578451882845186 (sklearn k-nn)
	
	**IMAGES**


	**Comments**



#. Without normalization of the data (raw data)



	**MSE:**
		1.0972559274755926 (custom k-nn)
		0.799700139470014 (sklearn k-nn)

	**Accuracy:**
		0.42807880055788006 (custom k-nn)
		0.47911785216178526 (sklearn k-nn)

	
	**IMAGES**


**Comments**

	We notice that in the raw data (without preprocessing) the algorithm we
	implemented has worse metrics compared to above (with preprocessing)
	and this can also be seen from the responses to unknown data. The 
	algorithm fails to classify the input patterns in the various classes
	and classifies them all in class 5. After debugging we noticed that for
	each query_wine the top k neighbors of the training set were the same
	which lies in the fact that for these neighbors the dot product
	(for each query wine) gave the largest response because their magnitude
	(of the top k) was logically much larger than the magnitude of all the
	remainders.

	**IMAGE**

	The previous relation shows that if we use the dot product as a
	metric to determine the neighbors then (since the metric of b is constant)
	the output of the metric will depend on the magnitude of the data
	and therefore the max (of these outputs). This means that the solution
	will be biased towards those training data that have a high magnitude
	(assuming, of course, that these α are in same direction and to some
	extent the angle is close to 1). On the other hand the L2 distance is
	, so to speak, invariant in magnitude because to decide on one of the
	top k neighbors we choose the min of the metric response.
	The following graph shows the possible situation:

**IMAGE**



	Finally, we must emphasize that the k-nn algorithm has the advantage 
	of being simple, as a result of which we can more easily analyze its
	behavior (as above). However like we saw it is not robust as its 
	performance depends a lot on the quality of the dataset that we also
	have for small disturbances at the input it may produce very different
	results in the output. Also its cost (in the general case not in our problem)
	in space but and in time is large and depends on the size of the problem
	(dimensionality of the input).
	

* Perceptron (Pytorch)

	**A hidden layer with relu non-linearity**

		First we must say that:

		#. For the validation of the algorithms we again used k-fold CV (k=5)
		   for some hyperparameters which after experimentation were 'frozen'.

		#. To terminate the training we used the following 2 criteria:
		
		   * Maximum number of epochs = 100
		   * We also used early-stopping in which we keep the validation loss 
             	     in each epoch (which is the loss to the subset of the 20% of the data
		     that we don't have used for training ) and if this
		     stops decreasing for some predetermined number of epochs (patience)
		     then we stop the training.

		#. If some run fo the algorithm make slightly different predictions
		   in the testing phase from the one shown in the experiments below
		   that is reasonable due to the stochastic gradient descent algorithm
		   used for weight refinement ( mean convergance ).








Reproduce the experiments
============






Free software: MIT license
============


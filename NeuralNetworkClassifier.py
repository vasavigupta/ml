import sys
import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import pdb
#np.set_printoptions(threshold=np.inf)
cn = 0;

def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W
    
    
    
def sigmoid(z):
    
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    return 1 / (1 + np.exp(-z))   
    
    

def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - divide the original data set to training, validation and testing set
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - feature selection"""
    
    mat = loadmat('mnist_all.mat') #loads the MAT object as a Dictionary
    
    #Pick a reasonable size for validation data
    #Your code here
    train_data = np.array([])
    train_label = np.array([])
    validation_data = np.array([])
    validation_label = np.array([])
    test_data = np.array([])
    test_label = np.array([])

    #Normalize
    train_data1 = np.array(mat["train0"])
    train_label1 = np.zeros((1,10))
    train_label1[0][0]=1
    for b in range(1,train_data1.shape[0]):
        train_label1 = np.concatenate((train_label1,np.zeros((1,10))))
        train_label1[b][0]=1
    validation_data = np.array([])
    validation_label = np.array([])
    test_data = np.array(mat["test0"])
    test_label = np.zeros((1,10))
    test_label[0][0]=1
    for a in range(1,test_data.shape[0]):
        test_label = np.concatenate((test_label,np.zeros((1,10))))
        test_label[a][0]=1

    for i in range(1,10):
        train_data1=np.concatenate((train_data1,np.array(mat["train"+str(i)])))
        temp_train_label1=np.zeros((((np.array(mat["train"+str(i)])).shape)[0],10))
        for j in range(0,((np.array(mat["train"+str(i)])).shape)[0]):
                temp_train_label1[j][i]=1
        train_label1=np.concatenate((train_label1,temp_train_label1))
        test_data=np.concatenate((test_data,np.array(mat["test"+str(i)])))
        temp_test_label=np.zeros((((np.array(mat["test"+str(i)])).shape)[0],10))
        for k in range(0,((np.array(mat["test"+str(i)])).shape)[0]):
                temp_test_label[k][i]=1
        test_label=np.concatenate((test_label,temp_test_label))
        #for j in range(0,((np.array(mat["test"+str(i)])).shape)[0]):
        #        temp_test_label[j][i]=1
    # print ("Feature Selection")
    #i = 0
    #while(i<train_data1.shape[1]):
    #    if(len(np.unique(train_data1[:,i]))<2):
    #       np.delete(train_data1,i,1)
    #       print(i)
    #    i=i+1

    #i = 0
    #while(i<test_data.shape[1]):
    #    if(len(np.unique(test_data[:,i]))<2):
    #       np.delete(test_data,i,1)
    #       print(i)
    #    i=i+1

    #print ("train data: "+str(train_data1.shape))
    #print ("test_data: " +str(test_data.shape))
    
    main = range(train_data1.shape[0])
    aperm =np.random.permutation(main)
    train_data=train_data1[aperm[10000:],:]
    validation_data=train_data1[aperm[0:10000],:]
    train_label=train_label1[aperm[10000:],:]
    validation_label=train_label1[aperm[0:10000],:]
    #print("aperm" + str(aperm))
    ##print (validation_data.shape)

    return np.divide(train_data,255.0), np.argmax(train_label,axis=1), np.divide(validation_data,255.0),np.argmax(validation_label,axis=1), np.divide(test_data,255.0), np.argmax(test_label,axis=1)
     
    

 
    
    

def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""
    
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
    
    w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    #........test dimensions............#
    ##print "nnOBJ"
    ##print "Training data"+str(training_data.shape)
    ##print "Training label"+str(training_label.shape)
    ##print "no of input nodes"+str(n_input)
    ##print "no of hidden"+str(n_hidden)
    ##print "no of classes"+str(n_class)
    #print ("W1"+str(w1.shape))
    #print ("W2"+str(w2.shape))

    
    #Your code here
    #
    #
    #
    #
    #
    #.......................Add Bias....................#
    tmp_train=training_data#--50000*784
    n=tmp_train.shape[0]
    #bias_in = np.ones((tmp_train.shape[0],1), dtype=np.float64)  ## create an input bias with 1's as per length of training data
    bias_in=np.ones((np.array(tmp_train.shape[0]),1))
    ####print "i/p bias"+str(bias_in.shape)
    x = np.append(tmp_train, bias_in,1)  ## append bias to training data
    z1=np.dot(x, w1.T)
    z = sigmoid(z1) ## calculate o/p of hidden nodes
    #print ("Z"+str(z.shape))#50*50000
    bias_hid=np.ones((np.array(z.shape[0]),1))
    z=np.append(z,bias_hid,1)
    o = sigmoid(np.dot(z,w2.T))  ## output - O , training_labels -Y
    #print ("O"+str(o.shape)) #10*50000

    ##........Grad_1 and Grad_2 calculation  at Output node..........##

    y=np.zeros((tmp_train.shape[0],10))
    for i in range (0 ,tmp_train.shape[0]):
        y[i][train_label[i]]=1
    #y=y.T
    #y=training_label
    #print ("y"+str(y.shape))
    delta=y-o
    #print ("delta"+str(delta.shape))
    scalar=(y-o)*(1-o)*o
    #print ("scalar"+str(scalar.shape))
    grad_w2=np.dot(scalar.T,z)
    grad_w2=-1*grad_w2
    #print ("grad_2"+str(grad_w2.shape))
    p1=(1-z)*z
    p1=(-1)*p1
    p2=np.dot(scalar,w2)*p1
    grad_w1=np.dot(x.T,p2)
    #grad_w1=-1*grad_w1
    grad_w1=grad_w1.T
    grad_w1=grad_w1[:-1,:]# remove last row
    #print ("grad_1"+str(grad_w1.shape))

    #...........................Error..........................#
    error=(y-o)
    error=np.square(error)
    error=np.sum(error)
    error=error/(2*n)



    #Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    #you would use code similar to the one below to create a flat array
    #obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)

    #..........Regularization....................#
    regular1=np.square(w1)
    regular1=np.sum(regular1)
    ##print "regular1"+str(regular1)
    regular2=np.square(w2)
    regular2=np.sum(regular2)
    ##print "regular2"+str(regular2)
    regular3=regular1+regular2
    regular4=lambdaval*regular3
    regular =regular4/(2*n)
    regular=error+regular
    obj_val=regular



    error_w1=lambdaval*w1
    error_w2=lambdaval*w2

    grad_w11=grad_w1/n+error_w1/n

    grad_w22=grad_w2/n+error_w2/n
    obj_grad = np.concatenate((grad_w11.flatten(), grad_w22.flatten()),0)
    ##print "obj_grad"+str(obj_grad)
    ##print "obj_val"+str(obj_val)
    global cn
    cn=cn+1
    #print (cn)
    #print ("obj_grad"+str(obj_grad.shape))
    #print ("obj_value"+str(obj_val))
    return (obj_val,obj_grad)


def nnPredict(w1,w2,data):
    
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels""" 
    labels = np.array([])
    #Your code here
    b1 = np.ones((np.array(data.shape[0]),1))
    x1=np.append(data,b1,axis=1)
    z1=np.dot(x1,w1.T)
    a1=sigmoid(z1)
    b2 = np.ones((np.array(a1.shape[0]),1))
    x2=np.append(a1,b2,axis=1)
    z2=np.dot(x2,w2.T)
    a2=sigmoid(z2)
    print(a2.shape)
    labels=np.argmax(a2,axis=1)
    #print (labels)
    return labels
 



"""**************Neural Network Script Starts here********************************"""
#print ("Starting")
train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();
#print("preprocess done")
#pdb.set_trace()

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]; 

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = int(sys.argv[2]);
				   
# set the number of nodes in output unit
n_class = 10;				   

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)

# set the regularization hyper-parameter
lambdaval = float(sys.argv[1]);


args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter' : 50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)

#In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
#and nnObjGradient. Check documentation for this function before you proceed.
#nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))


#Test the computed parameters

predicted_label = nnPredict(w1,w2,train_data)
#print ("train_data" + str(train_data))
#print ("train_label" + str(train_label))
#print ("validation_label" + str(validation_label))
#find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1,w2,validation_data)
print ("predicted label"+str( predicted_label))

#find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')


predicted_label = nnPredict(w1,w2,test_data)
#find the accuracy on Validation Dataset

print('\n Test set Accuracy:'  + str(100*np.mean((predicted_label == test_label).astype(float))) + '%')

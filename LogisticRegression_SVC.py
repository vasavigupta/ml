import pickle
import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn.svm import SVC

def preprocess():
    """
     Input:
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
    """

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, labeli = args

    n_data = train_data.shape[0]
    n_features = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_features + 1, 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    initialWeights = initialWeights.reshape(n_features+1,1)
    bias_in = np.ones((np.array(n_data),1))
    train_data = np.append(bias_in,train_data,1)
    #print(train_data.shape)
    #print(initialWeights.shape)
    z = np.dot(train_data,initialWeights)
    #print(z.shape)
    #print(labeli.shape)
    theta = sigmoid(z)
    theta = theta.reshape(n_data,1)
    #print(theta.shape)
    #temp1 = labeli*np.log(theta)
    #print(temp1.shape)
    #temp2 = (1-labeli)*(np.log(1-theta))
    #print(temp2.shape)
    #temp3 = temp1+temp2
    #print(temp3.shape)
    #temp4 = np.sum(temp3)
    #print(temp4)
    error = (-1.0/n_data) * np.sum((labeli * np.log(theta)) + ((1-labeli) * np.log(1-theta)))
    #print(error)
    temp5 =theta-labeli
    #print(temp5.shape)
    temp6 = temp5*train_data
    #print(temp6.shape)
    temp7 = np.sum(temp6,axis=0)
    #print (temp7.shape)
    error_grad = temp7/n_data
    error_grad = error_grad.reshape(error_grad.shape[0],1)
    error_grad= error_grad.flatten()
    #print(error_grad.shape)
    return error, error_grad


def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    n_data = data.shape[0]
    bias_in = np.ones((np.array(n_data),1))
    data = np.append(bias_in,data,1)
    z = np.dot(data,W)
    labels = sigmoid(z)
    #print(W)
    #print(labels)
    label = np.argmax(labels,axis=1)
    label =(label.reshape(n_data,1))
    return label


def mlrObjFunction(params, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector of size (D + 1) x 1
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
    train_data, labeli = args

    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]
    w = params.reshape(n_feature + 1, n_class)
    #print ("first", params.shape)
    #print ("first w", w.shape)
    #print (" w", w)
    train_data = np.hstack((np.ones((n_data, 1)), train_data))
    #print ("shape of traindata", train_data.shape)
    #print ("shape of true label", labeli.shape)

    tmp_mat = np.dot(train_data, w)
    train_data_tmp = np.zeros((train_data.shape[0], n_class))
    for i in range(n_data):
        tmp1 = np.sum(np.exp(tmp_mat[i]))
        tmp2 = np.exp(tmp_mat[i])
        tmp3 = tmp2 / tmp1
        train_data_tmp[i] = tmp3
    #print ("train_data_tmp shape" ,train_data_tmp.shape)
    theta = train_data_tmp

    #print ("theta shape", theta.shape)
    error = labeli * np.log(theta)
    error = np.sum(-1.0 * error)
    error = (error) / train_data.shape[0]

    print ("mlr error value", error)

    error_grad = np.zeros((n_feature + 1, n_class))
    error_grad = np.zeros((n_feature + 1, n_class))
    tmp3 = 1.0 * (theta - labeli)
    tmp4 = np.dot(train_data.T, tmp3)
    #print ("tmp4 shape", tmp4.shape)
    error_grad = tmp4
    error_grad = error_grad / train_data.shape[0]

    #print ("error_grad shape", error_grad.shape)
    #print ("error_grad flatten shape", error_grad.flatten().shape)

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    return error, error_grad.flatten()


def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))
    #print ("@@data predict shape", data.shape)
    #print ("@@w predict shape", W.shape)
    #data = np.vstack((np.ones((data.shape[0], 1)), data)).T
    data = np.hstack((np.ones((data.shape[0], 1)), data))
    #print ("data predict shape", data.shape)
    tmp_label= np.exp(np.dot(data, W))
    #print ("tmp_label predict shape", tmp_label.shape)
    label = np.argmax(tmp_label, axis=1)
    #print ("label predict shape", label.shape)
    #print ("label........", label)
    label=label.reshape(label.shape[0],1)
    #print ("label predict shape", label.shape)

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    return label


"""
Script for Logistic Regression
"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()

# Logistic Regression with Gradient Descent
W = np.zeros((n_feature + 1, n_class))
initialWeights = np.zeros((n_feature + 1, 1))
opts = {'maxiter': 100}
for i in range(n_class):
    labeli = Y[:, i].reshape(n_train, 1)
    args = (train_data, labeli)
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    W[:, i] = nn_params.x.reshape((n_feature + 1,))

# Find the accuracy on Training Dataset
predicted_label = blrPredict(W, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

with open('params.pickle', 'wb') as f1:
    pickle.dump(W, f1)

"""
Script for Support Vector Machine
"""

print('\n\n--------------SVM-------------------\n\n')
##################
# YOUR CODE HERE #
##################

print('\n-Linear Kernel (Default)-\n')
clf = SVC(kernel='linear');
clf.fit(train_data,train_label.flatten());
print('\n Results \n')
#predicted_label = clf.predict(train_data);
accuracy = clf.score(train_data,train_label.flatten());
print('\n Training set Accuracy:' + str(accuracy*100)+'%')
accuracy = clf.score(validation_data,validation_label.flatten());
print('\n Validation set Accuracy:' + str(accuracy*100)+'%')
accuracy = clf.score(test_data,test_label.flatten());
print('\n Testing set Accuracy:' + str(accuracy*100)+'%')

print('\n-------------------\n')


print('\n-Radial Basis Function (Gamma = 1)-\n')
clf = SVC(kernel='rbf',gamma=1.0);
clf.fit(train_data,train_label.flatten());
print('\n Results \n')
#predicted_label = clf.predict(train_data);
accuracy = clf.score(train_data,train_label.flatten());
print('\n Training set Accuracy:' + str(accuracy*100)+'%')
accuracy = clf.score(validation_data,validation_label.flatten());
print('\n Validation set Accuracy:' + str(accuracy*100)+'%')
accuracy = clf.score(test_data,test_label.flatten());
print('\n Testing set Accuracy:' + str(accuracy*100)+'%')

print('\n-------------------\n')

print('\n-Radial Basis Function (Gamma = default)-\n')
clf = SVC(kernel='rbf');
clf.fit(train_data,train_label.flatten());
print('\n Results \n')
#predicted_label = clf.predict(train_data);
accuracy = clf.score(train_data,train_label.flatten());
print('\n Training set Accuracy:' + str(accuracy*100)+'%')
accuracy = clf.score(validation_data,validation_label.flatten());
print('\n Validation set Accuracy:' + str(accuracy*100)+'%')
accuracy = clf.score(test_data,test_label.flatten());
print('\n Testing set Accuracy:' + str(accuracy*100)+'%')

print('\n-------------------\n')

print('\n-Radial Basis Function (Gamma = Default, Varying C)-\n')
print('\nFor C = 1\n')
clf = SVC(C=1,kernel='rbf');
clf.fit(train_data,train_label.flatten());
print('\n Results \n')
#predicted_label = clf.predict(train_data);
accuracy = clf.score(train_data,train_label.flatten());
print('\n Training set Accuracy:' + str(accuracy*100)+'%')
accuracy = clf.score(validation_data,validation_label.flatten());
print('\n Validation set Accuracy:' + str(accuracy*100)+'%')
accuracy = clf.score(test_data,test_label.flatten());
print('\n Testing set Accuracy:' + str(accuracy*100)+'%')

for i in range(10,110,10):
	print('\nFor C = ')
	print(i)
	print('\n')
	clf = SVC(C=i,kernel='rbf');
	clf.fit(train_data,train_label.flatten());
	print('\n Results \n')
	#predicted_label = clf.predict(train_data);
	accuracy = clf.score(train_data,train_label.flatten());
	print('\n Training set Accuracy:' + str(accuracy*100)+'%')
	accuracy = clf.score(validation_data,validation_label.flatten());
	print('\n Validation set Accuracy:' + str(accuracy*100)+'%')
	accuracy = clf.score(test_data,test_label.flatten());
	print('\n Testing set Accuracy:' + str(accuracy*100)+'%')

print('\n-------------------\n')





"""
Script for Extra Credit Part
"""
print ("MLR")
# FOR EXTRA CREDIT ONLY
W_b = np.zeros((n_feature + 1, n_class))
initialWeights_b = np.zeros((n_feature + 1, n_class))
opts_b = {'maxiter': 100}

args_b = (train_data, Y)
nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))

# Find the accuracy on Training Dataset
predicted_label_b = mlrPredict(W_b, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label_b = mlrPredict(W_b, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label_b = mlrPredict(W_b, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')

with open('params_bonus.pickle', 'wb') as f2:
    pickle.dump(W_b, f2)

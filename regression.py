import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

def ldaLearn(X,y):
    y = y.flatten()
    #print ('Y resize shape' + str(y.shape))
    diff_class = np.unique(y);
    #print ('classes' + str(diff_class))
    mean1 = np.mean(X[y == 1], 0)
    # print ('mean1'+str(mean1.shape))
    mean2 = np.mean(X[y == 2], 0)
    mean3 = np.mean(X[y == 3], 0)
    mean4 = np.mean(X[y == 4], 0)
    mean5 = np.mean(X[y == 5], 0)
    tmp = np.zeros((X.shape[1], 5))
    for i in range(diff_class.size):
        tmp[:, i] = np.mean(X[y == diff_class[i]], 0)
    #print ('mean  shape' + str(tmp.shape))
    means=tmp
    X=X.T
    covmat = np.cov(X)
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD
    
    return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD
    y = y.flatten()
    # print ('Y resize shape' + str(y.shape))
    diff_class = np.unique(y);
    # print ('classes' + str(diff_class))
    mean1 = np.mean(X[y == 1], 0)
    # print ('mean1'+str(mean1.shape))
    mean2 = np.mean(X[y == 2], 0)
    mean3 = np.mean(X[y == 3], 0)
    mean4 = np.mean(X[y == 4], 0)
    mean5 = np.mean(X[y == 5], 0)
    tmp = np.zeros((X.shape[1], 5))
    tmp2 = [np.zeros((X.shape[1], X.shape[1]))] * 5
    for i in range(diff_class.size):
        tmp[:, i] = np.mean(X[y == diff_class[i]], 0)
        tmp2[i] = np.cov(X[y == diff_class[i]].T)

    # print ('mean  shape' + str(tmp.shape))
    means = tmp
    X = X.T
    covmat = tmp2
    
    return means,covmat

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    # p=np.empty(5)
    diff_classes = np.unique(ytest)
    # print 'test classes'+str(diff_classes)
    prior_1 = ((ytest.flatten() == 1).sum()) / 100
    # print ('*******p1'+str(prior_1))
    prior_2 = ((ytest.flatten() == 2).sum()) / 100
    # print ('*******p1' + str(prior_2))
    prior_3 = ((ytest.flatten() == 3).sum()) / 100
    # print ('*******p1' + str(prior_3))
    prior_4 = ((ytest.flatten() == 4).sum()) / 100
    # print ('*******p1' + str(prior_4))
    prior_5 = ((ytest.flatten() == 5).sum()) / 100
    # print ('*******p1' + str(prior_5))

    # print ('means shape'+str(means.shape))
    # print ('test shape'+str(Xtest.shape))
    means = means.T
    # p = np.zeros((Xtest.shape[0],means.shape[0]))
    c_inverse = np.linalg.inv(covmat)
    det = np.linalg.det(covmat)
    # print 'det'+str(det)
    list = []
    p = []
    print ("D value",Xtest.shape[1]);
    denominator = np.power(2 * 3.14, Xtest.shape[1]/ 2) * np.sqrt(det)
    # print ('denominator scalar'+str(denominator))
    for i in range(Xtest.shape[0]):
        for j in range(means.shape[0]):
            # tmp1=np.transpose(np.subtract(Xtest[i,:],means[:,j]))
            tmp1 = (Xtest[i, :] - means[j, :]).T
            # tmp2=np.subtract(Xtest[i,:],means[:,j])
            # print ('********** xtest************'+str(Xtest[i,:]))
            # print ('********** mean************' + str(means[j,:]))
            tmp2 = Xtest[i, :] - means[j, :]
            # print ('tmp1 shape'+str(tmp1.shape))
            # print ('c_inverse shape'+str(c_inverse.shape))



            #           print ('Xtest[i,:]-means[j,:]'+str(np.subtract(Xtest[i,:],means[j,:])))

            #            print ('Xtest[i,:]-means[j,:]*inverse' + str(np.dot(Xtest[i, :] - means[j, :],c_inverse)))
            #           print ('Xtest[i,:]-means[j,:] .T' + str((Xtest[i, :] - means[j, :]).T))

            num = np.dot(np.dot(tmp2, c_inverse), (tmp1))
            num = np.exp(-0.5 * num)

            #            print ('num shape' + str(num))
            #           print ('denominator'+str(denominator))


            val = num / denominator
            #  if(ytest[i]==1):
            #     val=prior_1*val
            # elif ytest[i]==2:
            #   val=prior_2*val
            # elif ytest[i]==3:
            #   val=prior_3*val
            # elif ytest[i]==3:
            #   val=prior_4*val
            # else:
            #   val=prior_5*val



            list.append(val)
        # p.append((np.argmax(list, 0)))
        x = np.argmax(list, 0)

        #      print list
        #     print ('....................index.......................'+str(np.argmax(list, 0)))

        if (i == 0):
            ypred = np.array((x + 1))
        else:
            ypred = np.vstack((ypred, (x + 1)))
        list = []
        # p=[]

        # list.clear()



        #    print ('ypred'+str(ypred))
        #   print ('ytest' + str(ytest))

    acc = 100 * np.mean((ypred == ytest).astype(float))
    return acc,ypred

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    # IMPLEMENT THIS METHOD
    diff_classes = np.unique(ytest)
    # print 'test classes' + str(diff_classes)
    prior_1 = ((ytest.flatten() == 1).sum()) / 100
    # print ('*******p1' + str(prior_1))
    prior_2 = ((ytest.flatten() == 2).sum()) / 100
    # print ('*******p2' + str(prior_2))
    prior_3 = ((ytest.flatten() == 3).sum()) / 100
    # print ('*******p3' + str(prior_3))
    prior_4 = ((ytest.flatten() == 4).sum()) / 100
    # print ('*******p4' + str(prior_4))
    prior_5 = ((ytest.flatten() == 5).sum()) / 100
    # print ('*******p5' + str(prior_5))

    # print ('means shape' + str(means.shape))
    # print ('test shape' + str(Xtest.shape))
    means = means.T
    # p = np.zeros((Xtest.shape[0],means.shape[0]))

    list = []
    p = []

    # print ('denominator scalar' + str(denominator))
    for i in range(Xtest.shape[0]):
        for j in range(means.shape[0]):
            c_inverse = np.linalg.inv(covmats[j])
            det = np.linalg.det(covmats[j])
            denominator = np.power(2 * 3.14, 2 / 2) * np.sqrt(det)
            # print 'det' + str(det)
            # tmp1=np.transpose(np.subtract(Xtest[i,:],means[:,j]))
            tmp1 = (Xtest[i, :] - means[j, :]).T
            # tmp2=np.subtract(Xtest[i,:],means[:,j])
            # print ('********** xtest************' + str(Xtest[i, :]))
            # print ('********** mean************' + str(means[j, :]))
            tmp2 = Xtest[i, :] - means[j, :]
            # print ('tmp1 shape' + str(tmp1.shape))
            # print ('c_inverse shape' + str(c_inverse.shape))

            # print ('Xtest[i,:]-means[j,:]' + str(np.subtract(Xtest[i, :], means[j, :])))

            # print ('Xtest[i,:]-means[j,:]*inverse' + str(np.dot(Xtest[i, :] - means[j, :], c_inverse)))
            # print ('Xtest[i,:]-means[j,:] .T' + str((Xtest[i, :] - means[j, :]).T))

            num = np.dot(np.dot(tmp2, c_inverse), (tmp1))
            num = np.exp(-0.5 * num)

            # print ('num shape' + str(num))
            # print ('denominator' + str(denominator))

            val = num / denominator
            #  if(ytest[i]==1):
            #     val=prior_1*val
            # elif ytest[i]==2:
            #   val=prior_2*val
            # elif ytest[i]==3:
            #   val=prior_3*val
            # elif ytest[i]==3:
            #   val=prior_4*val
            # else:
            #   val=prior_5*val



            list.append(val)
        # p.append((np.argmax(list, 0)))
        x = np.argmax(list, 0)

        # print list
        # print ('....................index.......................' + str(np.argmax(list, 0)))

        if (i == 0):
            ypred = np.array((x + 1))
        else:
            ypred = np.vstack((ypred, (x + 1)))
        list = []
        # p=[]

        # list.clear()

    # print ('ypred' + str(ypred))
    # print ('ytest' + str(ytest))

    acc = 100 * np.mean((ypred == ytest).astype(float))

    return acc,ypred

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1                                                                
    # IMPLEMENT THIS METHOD
    print ('OLE Regression')
    tmp1 = np.linalg.inv(np.dot(np.transpose(X), X))
    tmp2 = np.dot(np.transpose(X), y)
    w = np.dot(tmp1, tmp2)
    print ('weight shape'+str(w.shape))
    # ('******Weight**********' + str(w))

    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD
    print ("Learn RidgeRegression")
    identitymat=np.identity(X.shape[1])
    lambdamat=np.multiply(lambd,identitymat)
    #print(lambdamat.shape)
    t=np.dot(np.transpose(X), X)
    #print(t.shape)
    sum=np.add(lambdamat,t)
    t1=np.linalg.inv(sum)
    tmp2 = np.dot(np.transpose(X), y)
    w = np.dot(t1, tmp2)
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # rmse
    #tmp=np.subtract(ytest , np.dot(Xtest,w))
    #print ('tmp*******'+str(tmp))
    #tmp1 = np.sum(tmp*tmp)
    #print ('tmp1*******' + str(tmp1))
    #tmp1 = tmp1 / Xtest.shape[0]
    #print ('tmp2*******' + str(tmp1))
    #tmp1 = np.sqrt(tmp1)
    #print ('tmp3*******' + str(tmp1))

    #rmse = tmp1
    #print ('w shape '+str(w.shape))
    #print ('Xtest shape'+str(Xtest.shape))
    #print ('Ytest shape' + str(ytest.shape))

    rmse= np.sqrt(np.sum(np.square(ytest - np.dot(Xtest, w)))/Xtest.shape[0])
    
    # IMPLEMENT THIS METHOD
    return rmse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD     
    w1=np.array([w]).T
    error1=(np.dot(np.transpose(np.subtract(y,np.dot(X,w1))),np.subtract(y,np.dot(X,w1))))/(2.0*X.shape[0])
    error2=(np.dot(lambd,np.dot(np.transpose(w1),w1)))/2
    error=error1+error2  
    error = error.flatten()   
    error_grad=((-np.dot(np.transpose(y),X)+np.dot(np.transpose(w),np.dot(np.transpose(X),X)))/X.shape[0]) + np.dot(lambd,np.transpose(w))                                        
    error_grad=error_grad.flatten()
    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xd - (N x (d+1))                                                         
    # IMPLEMENT THIS METHOD
    Xd = np.ones((x.shape[0], 1), dtype=np.int)
    #print("Initial Xd--------------------------------------------------------------------------------------------------")
    #print(Xd.shape)
    for j in range(1,p+1):
        tmp=np.power(x,j)
        t= np.reshape(tmp, (tmp.shape[0], 1))
        #print(t.shape)
        Xd = np.hstack((Xd, t))
    #print (Xd.shape)
    return Xd

# Main script

# Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])))
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)

plt.show()

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])))
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.close()
# Problem 2

if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle_train=testOLERegression(w,X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_itrain=testOLERegression(w_i,X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)


print ('Training Data')
print('RMSE without intercept '+str(mle_train))
print('RMSE with intercept '+str(mle_itrain))
print ('TEST data')
print('RMSE without intercept '+str(mle))
print('RMSE with intercept '+str(mle_i))
plt.close()
# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
rmses3 = np.zeros((k,1))
remse3_train = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    rmses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
plt.plot(lambdas,rmses3)

plt.show()
plt.close()
# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
rmses4 = np.zeros((k,1))
opts = {'maxiter' : 100}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    rmses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
plt.plot(lambdas,rmses4)
plt.show()
plt.close()

# Problem 5
pmax = 7
lambda_opt = lambdas[np.argmin(rmses3)]
rmses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    rmses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    rmses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)
plt.plot(range(pmax),rmses5)
plt.legend(('No Regularization','Regularization'))
plt.show()
plt.close()
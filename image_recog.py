# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 13:09:56 2017

@author: shrey
"""
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import os


#train the logistic regression model
def logistic_reg_train(data_x,data_y,epochs,m,lr=0.01):
    #initialize parameters
    weight=np.zeros((len(data_x),1))
    bias=0
    costt=[] #to store cost value for every epoch 
    for i in range(epochs):
        z=np.dot(weight.T,data_x)+bias  #calculate z
        a=1 / (1 + np.exp(-z))          #calculate sigmoid of z
        j=-(np.multiply(data_y,np.log(a))+np.multiply((1-data_y),np.log(1-a))) #this is cost
        cost=np.sum(j)/m #average cost of training set
        dz=a-data_y 
        dw=np.dot(data_x,dz.T)/m #this is the adjustment for weight
        db=np.sum(dz)/m  #adjustment for bias
        
        weight=weight-lr*dw #update weights
    
        bias=bias-lr*db #update bias
        
        print('loss for epoch '+str(i)+' is:'+ str(cost))
        costt.append(cost)
    plt.plot(costt)     #plot cost
    result={'weights':weight,'bias':bias,'cost':costt}
    return result        
 
 #takes test set and predict its output using the trained weights and bias       
def logistic_reg_predict(data,res):
    weight=res['weights']
    bias=res['bias']
    z=np.dot(weight.T,data)+bias
    predict=1 / (1 + np.exp(-z))
    return predict

#################################################################################
#this block makes the training and testing set
#################################################################################
num_train=600
num_test=250
pos=0
neg=0
index=0
test_index=0
train_array_x=np.zeros((num_train,4000),dtype=object)  #dimensions of images are 40 X 100
train_array_y=np.zeros(num_train)
test_array_x=np.zeros((num_test,4000),dtype=object)
test_array_y=np.zeros(num_test)
path = '/home/shrey/Desktop/project/CarData/TrainImages/'
for filename in os.listdir(path):
    arr= misc.imread(path+filename)
    if 'pos' in filename:
        if pos>=(num_train/2):
            test_array_x[test_index]=arr.flatten()
            test_index+=1
            continue
        pos+=1
        train_array_x[index]=arr.flatten()
        train_array_y[index]=1
    else:
        if neg>=num_train/2:
            continue
        neg+=1
        train_array_x[index]=arr.flatten()
        train_array_y[index]=0
        
    index+=1
    if index>num_train and test_index>num_test:
        break
#################################################################################
        
#################################################################################        

    
train_array_x=train_array_x.astype('float')  # make the type as float
test_array_x=test_array_x.astype('float')
train_array_y=train_array_y.reshape((1,600))
# we want to stack the training examples to be stacked vertically not horizontally and be normalized
train_array_x=train_array_x.T/255
test_array_x=test_array_x.T/255


res=logistic_reg_train(train_array_x,train_array_y,100,num_train)
pre=logistic_reg_predict(test_array_x,res)

#find the number of correctly classified images
pre=pre>0.5
pre=pre.astype('float')
c=(pre==1).sum()

#calculate accuracy
accuracy=(float(c)/float(test_array_x.shape[1]))*100

#plot cost
plt.plot(res['cost'])




    
    
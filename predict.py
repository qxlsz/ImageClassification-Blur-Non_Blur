import glob
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import scipy.misc as ms
import scipy.ndimage as nd
import argparse
import cv2
def resize(X, orig_reso, w=100, h=100):
    r,c = w,h
    X_new = np.zeros((X.shape[0],r*c))
    for i in range(X.shape[0]):
        X_new[i,:] = ms.imresize(X[i,:].reshape(orig_reso[0],orig_reso[1]),(r,c),interp='cubic').flatten()
    return X_new
def sigmoid(X):
    return 1.0 / (1.0 + np.exp(-X))
## Different Activaion Function
def h(theta,X,func='sig'):
    a = theta.dot(X.T)
    if(func== 'tanh'):
        return np.tanh(a)
    if func == 'none':
        return a
    if func == 'softplus':
        return np.log(1 + np.exp(a))
    if func == 'relu':
        return np.maximum(0.01*a, a)
    if func == 'softmax':
        a1 = np.exp(a)
        a1 = a1 / np.sum(a1, axis = 0, keepdims = True)
        return a1    
    return sigmoid(a)
def validate(theta1, theta2, X, act = 'sig'):
    aa1 = h(theta1,X,act)
    aa1 = np.insert(aa1, 0, 1, axis=0)
    aa2 = h(theta2,aa1.T,'softmax')
    accu_matrix = np.argmax(aa2,axis=0) 
    return accu_matrix
## Extracting the Info from Training Set Results from tmp dir
inp_image = ms.imread('/Users/rjosyula/Pictures/ImageClassification3/Images/Bad/raw_Image_279.bmp', mode ="L")
print(inp_image.shape)
## Rescalling the Inputs
#inp_image = inp_image.flatten()
inp_image = nd.median_filter(inp_image,3)
print(inp_image.shape)
orig_reso = (964,1280)
#inp_image = resize(inp_image, orig_reso)
inp_image = ms.imresize(inp_image.reshape(orig_reso[0],orig_reso[1]),(100,100),interp='cubic').flatten()
inp_image = inp_image/255.0
print(inp_image.shape)

inp_image = np.insert(inp_image, 0, 1, axis=0)
print(inp_image.shape)
## Orignal Resolution Of Image

## construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-model_dir", "--model_dir", required=False, help="path to model directory")
args = vars(ap.parse_args())
## Retrieving the Temp Folder for Result 
if args["model_dir"]:
  filename = args["model_dir"]+str("/")
else:
  filename = "/tmp/blur_clear/"
if not os.path.exists(os.path.dirname(filename)):
    print "No dir exists" + filename
    exit()
## Extracting the Info from Training Set Results from tmp dir
params = np.load(filename + str("/result.npy"))
## Predicting the Labels , Accuracy Score
pred_y = validate(params[()]['Theta1'], params[()]['Theta2'], inp_image)
print (pred_y)
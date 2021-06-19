from tqdm import tqdm
import numpy as np
import os.path
import os, re, glob
import sys
import random
import math
import cv2
import pandas
from PIL import Image
import pickle
from glob import glob
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

dataset_directory = 'D:/projects/simulation/data'

def centeredCrop(img, output_side_length):
	height, width, depth = img.shape
	new_height = output_side_length
	new_width = output_side_length
	if height > width:
		new_height = output_side_length * height / width
	else:
		new_width = output_side_length * width / height
	height_offset = int((new_height - output_side_length) / 2)
	width_offset = int((new_width - output_side_length) / 2)
	cropped_img = img[height_offset:height_offset + output_side_length,
	                          width_offset:width_offset + output_side_length]
	return cropped_img

def image2RGB(image):
	b,g,r = cv2.split(image)
	X = cv2.merge([r,g,b])
	return X

def imageprocess(image):
	X=image2RGB(image)

	#normal resolution
	X = cv2.resize(X, (224, 224),interpolation=cv2.INTER_LINEAR)

	#high resolution
	#X = cv2.resize(X, (480, 270),interpolation=cv2.INTER_LINEAR)
	#megahigh resolution  [360, 640]
	#X = cv2.resize(X, (640, 360),interpolation=cv2.INTER_LINEAR)

	#plt.imshow(X)
	#plt.show() 
	#X = centeredCrop(X, 224)
	return X

def dataset_process(dataset):
	images_out = [] #final result
	images_raw = []
	images_pose =[]
	Raw_Address =[]
	X1=[]
	for i in tqdm(range(len(dataset))):
		X1_r = cv2.imread(dataset[i])
		X1_r = imageprocess(X1_r)

		X1.append(X1_r)

	X1=np.array(X1)
	mean1=0
	#images = X1.astype('f4')
	#mean1=X1.mean(axis=0)
	#X1 = X1 - mean1
	
	#fig, ax = plt.subplots()
	#plt.imshow(X1[1])
	#plt.show()
	#X1=np.float32(X1)
	#plt.imshow(X1[1])
	#plt.show()
	#plt.imshow(X1[54])
	#plt.show()
	#N=0	
	#mean1 = np.zeros((np.shape(X1)[1], np.shape(X1)[2],3))
	#for i in tqdm(range(len(dataset))):
	#		mean1[:,:,0] += X1[i,:,:,0]
	#		mean1[:,:,1] += X1[i,:,:,1]
	#		mean1[:,:,2] += X1[i,:,:,2]
	#		N += 1
	#mean1 /= N
	

	##plt.imshow(X1[4,:,:])
	##plt.show() 

	#X1=X1-mean1


	#print(sys.getsizeof(X1))
	#X1=X1.astype("float16")
	#X2=X2.astype("float16")
	#X3=X3.astype("float16")	
	#print(sys.getsizeof(X1))
	#print("add them")


	#plt.imshow(X1[4,:,:])
	#plt.show() 
	
	#X1 = np.expand_dims(X1, axis=0)
	return [X1,mean1]

from tensorflow.keras.applications.resnet50 import preprocess_input
def resnet50_dataset_process(dataset):
	images_out = [] #final result
	images_raw = []
	images_pose =[]
	Raw_Address =[]
	X1=[]
	for i in tqdm(range(len(dataset))):
		X1_r = cv2.imread(dataset[i])
		X1_r = cv2.resize(X1_r, (224, 224),interpolation=cv2.INTER_LINEAR)
		X1.append(X1_r)

	X1=np.array(X1)
	X1 = preprocess_input(X1)
	mean1=0
	return [X1,mean1]

import tensorflow.keras.applications.inception_resnet_v2 as inceptionresnetv2

def inceptionresnetv2_dataset_process(dataset):
	images_out = [] #final result
	images_raw = []
	images_pose =[]
	Raw_Address =[]
	X1=[]
	for i in tqdm(range(len(dataset))):
		X1_r = cv2.imread(dataset[i])
		X1_r = cv2.resize(X1_r, (224, 224),interpolation=cv2.INTER_LINEAR)
		X1.append(X1_r)

	X1=np.array(X1)
	X1 = inceptionresnetv2.preprocess_input(X1)
	mean1=0
	return [X1,mean1]


def data_prepare():
	############################################################################
	######################### preparing the dataset ############################
	############################################################################
	 images=glob(dataset_directory+'/*/')
	 random.shuffle(images)
	 dataset=[]
	 Y=[]
	 #for i in range(np.shape(images)[0]):
	 #for i in range(10):
	 for i in range(4000):
	 	path1=images[i]+'/AboveCamera.jpg'
	 	path2=images[i]+'/MountCamera.jpg'
	 	path3=images[i]+'/MountCamera2.jpg'
	 	x=float(os.path.split(os.path.split(images[i])[0])[1].split()[0].split('_')[0])
	 	y=float(os.path.split(os.path.split(images[i])[0])[1].split()[0].split('_')[1])
	 	r=float(os.path.split(os.path.split(images[i])[0])[1].split()[0].split('_')[2])
	 	tmp=[path1,path2,path3]
	 	y=[x,y,r]
	 	y=np.asarray(y)
	 	Y.append(y)
	 	tmp=np.asarray(tmp)
	 	dataset.append(tmp)
	
	 dataset=np.asarray(dataset)

	 Y=np.asarray(Y)
	 pickle.dump([Y], open('Y.pickle', 'wb'),protocol=4)
	 #Y=0



	 #above,mean1 = dataset_process(dataset[:,0])
	 #pickle.dump([above,Y,mean1,dataset[:,0]], open('Above.pickle', 'wb'),protocol=4)
	 #above=0
	 #mean1=0

	 mount1,mean2 = resnet50_dataset_process(dataset[:,1])
	 pickle.dump([mount1,Y,mean2,dataset[:,1]], open('Mount.pickle', 'wb'),protocol=4)
	 mount1=0
	 mean2=0

	 #mount1,mean2 = dataset_process(dataset[:,1])
	 #pickle.dump([mount1,Y,mean2,dataset[:,1]], open('Mount.pickle', 'wb'),protocol=4)
	 #mount1=0
	 #mean2=0

	 #mount2,mean3 = dataset_process(dataset[:,2])
	 #pickle.dump([mount2,Y,mean3,dataset[:,2]], open('Mount2.pickle', 'wb'),protocol=4)
	 #mount2=0
	 #mean3=0
	 return
def data_prepare_stereo():
	 images=glob(dataset_directory+'/*/')
	 random.shuffle(images)
	 dataset=[]
	 Y=[]
	 for i in range(np.shape(images)[0]):
	 #for i in range(10):
	 #for i in range(4000):
	 	path1=images[i]+'/AboveCamera.jpg'
	 	path2=images[i]+'/MountCamera.jpg'
	 	path3=images[i]+'/MountCamera2.jpg'
	 	x=float(os.path.split(os.path.split(images[i])[0])[1].split()[0].split('_')[0])
	 	y=float(os.path.split(os.path.split(images[i])[0])[1].split()[0].split('_')[1])
	 	r=float(os.path.split(os.path.split(images[i])[0])[1].split()[0].split('_')[2])
	 	tmp=[path1,path2,path3]
	 	y=[x,y,r]
	 	y=np.asarray(y)
	 	Y.append(y)
	 	tmp=np.asarray(tmp)
	 	dataset.append(tmp)
	
	 dataset=np.asarray(dataset)

	 Y=np.asarray(Y)
	 pickle.dump([Y], open('Y.pickle', 'wb'),protocol=4)
	 #Y=0

	 

	 above,mean1 = dataset_process(dataset[:,0])
	 mount1,mean2 = dataset_process(dataset[:,1])
	 mount2,mean3 = dataset_process(dataset[:,2])

	 #above,mean1 = inceptionresnetv2_dataset_process(dataset[:,0])
	 #mount1,mean2 = inceptionresnetv2_dataset_process(dataset[:,1])
	 #mount2,mean3 = inceptionresnetv2_dataset_process(dataset[:,2])



	 pickle.dump([above,mount1,mount2,dataset,Y], open('Stereo.pickle', 'wb'),protocol=4)
	 above=0
	 mean1=0
	 mount1=0
	 mean2=0
	 return
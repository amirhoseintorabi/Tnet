import tensorflow.keras
from tensorflow.keras.layers import Input, Dense, Conv2D
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import ZeroPadding2D, Dropout, Flatten
from tensorflow.keras.layers import Reshape, Activation, BatchNormalization
from tensorflow.keras.layers import concatenate
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import h5py
import math
import pydot
import tensorflow.keras
from tensorflow.keras.regularizers import l2
from tensorflow.keras import activations
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,MaxPool2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import h5py
import math
import pydot
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,MaxPool2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.nasnet import NASNetLarge
from tensorflow.keras.applications.resnet50 import ResNet50
import cv2, numpy as np
import pydot
from tensorflow.keras.models import Model
import tensorflow.keras as keras







def euc_loss1x(y_true, y_pred):
    lx = K.sqrt(K.sum(K.square(y_true[:,:] - y_pred[:,:]), axis=1, keepdims=True))
    return (0.1 * lx)
def euc_loss2x(y_true, y_pred):
    lx = K.sqrt(K.sum(K.square(y_true[:,:] - y_pred[:,:]), axis=1, keepdims=True))
    return (0.1 * lx)
def euc_loss3x(y_true, y_pred):
    lx = K.sqrt(K.sum(K.square(y_true[:,:] - y_pred[:,:]), axis=1, keepdims=True))
    return (1 * lx)

def euc_loss1r(y_true, y_pred):
    lx = K.sqrt(K.sum(K.square(y_true[:,:] - y_pred[:,:]), axis=1, keepdims=True))
    return (0.1 * lx)
def euc_loss2r(y_true, y_pred):
    lx = K.sqrt(K.sum(K.square(y_true[:,:] - y_pred[:,:]), axis=1, keepdims=True))
    return (0.1 * lx)
def euc_loss3r(y_true, y_pred):
    lx = K.sqrt(K.sum(K.square(y_true[:,:] - y_pred[:,:]), axis=1, keepdims=True))
    return (1 * lx)















def create_posenet_org():
    # creates Posenet from GoogLeNet a.k.a. Inception v1 (Szegedy, 2015)
    with tf.device('/gpu:0'):
        input = Input(shape=(256, 455, 3))
        
        conv1 = Conv2D(64,7,2,padding='same',activation='relu',name='conv1')(input)        
        pool1 = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same',name='pool1')(conv1)
        norm1 = BatchNormalization(axis=3, name='norm1')(pool1)        
        reduction2 = Conv2D(64,1,padding='same',activation='relu',name='reduction2')(norm1)        
        conv2 = Conv2D(192,3,padding='same',activation='relu',name='conv2')(reduction2)
        norm2 = BatchNormalization(axis=3, name='norm2')(conv2)
        pool2 = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='valid',name='pool2')(norm2)
        icp1_reduction1 = Conv2D(96,1,1,padding='same',activation='relu',name='icp1_reduction1')(pool2)
        icp1_out1 = Conv2D(128,3,padding='same',activation='relu',name='icp1_out1')(icp1_reduction1)
        icp1_reduction2 = Conv2D(16,1,padding='same',activation='relu',name='icp1_reduction2')(pool2)
        icp1_out2 = Conv2D(32,5,padding='same',activation='relu',name='icp1_out2')(icp1_reduction2)   
        icp1_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name='icp1_pool')(pool2)
        icp1_out3 = Conv2D(32,1,padding='same',activation='relu',name='icp1_out3')(icp1_pool)       
        icp1_out0 = Conv2D(64,1,padding='same',activation='relu',name='icp1_out0')(pool2)

        
        icp2_in = concatenate([icp1_out0, icp1_out1, icp1_out2, icp1_out3], axis=3, name='icp2_in') 
        icp2_reduction1 = Conv2D(128,1,padding='same',activation='relu',name='icp2_reduction1')(icp2_in)
        icp2_out1 = Conv2D(192,3,padding='same',activation='relu',name='icp2_out1')(icp2_reduction1)     
        icp2_reduction2 = Conv2D(32,1,padding='same',activation='relu',name='icp2_reduction2')(icp2_in)
        icp2_out2 = Conv2D(96,5,padding='same',activation='relu',name='icp2_out2')(icp2_reduction2)
        icp2_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name='icp2_pool')(icp2_in)
        icp2_out3 = Conv2D(64,1,padding='same',activation='relu',name='icp2_out3')(icp2_pool)
        icp2_out0 = Conv2D(128,1,padding='same',activation='relu',name='icp2_out0')(icp2_in)        
        icp2_out = concatenate([icp2_out0, icp2_out1, icp2_out2, icp2_out3],axis=3,name='icp2_out')






        icp3_in = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same',name='icp3_in')(icp2_out)
        icp3_reduction1 = Conv2D(96,1,padding='same',activation='relu',name='icp3_reduction1')(icp3_in)
        icp3_out1 = Conv2D(208,3,padding='same',activation='relu',name='icp3_out1')(icp3_reduction1)
        icp3_reduction2 = Conv2D(16,1,padding='same',activation='relu',name='icp3_reduction2')(icp3_in)
        icp3_out2 = Conv2D(48,5,padding='same',activation='relu',name='icp3_out2')(icp3_reduction2)      
        icp3_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name='icp3_pool')(icp3_in)
        icp3_out3 = Conv2D(64,1,padding='same',activation='relu',name='icp3_out3')(icp3_pool)
        icp3_out0 = Conv2D(192,1,padding='same',activation='relu',name='icp3_out0')(icp3_in)       
        icp3_out = concatenate([icp3_out0, icp3_out1, icp3_out2, icp3_out3],axis=3,name='icp3_out')
        



        ##############
        # first output auxiliary        
        cls1_pool = AveragePooling2D(pool_size=(5,5),strides=(3,3),padding='valid',name='cls1_pool')(icp3_out)        
        cls1_reduction_pose = Conv2D(128,1,padding='same',activation='relu',name='cls1_reduction_pose')(cls1_pool)
        cls1_fc1_flat = Flatten()(cls1_reduction_pose)        
        cls1_fc1_pose = Dense(1024,activation='relu',name='cls1_fc1_pose')(cls1_fc1_flat)
        cls1_fc_pose_xyz = Dense(3,name='cls1_fc_pose_xyz')(cls1_fc1_pose)
        cls1_fc_fl = Dense(1,name='cls1_fc_fl')(cls1_fc1_pose)
        cls1_fc_pose_wpqr = Dense(4,name='cls1_fc_pose_wpqr')(cls1_fc1_pose)


               
        icp4_reduction1 = Conv2D(112,1,padding='same',activation='relu',name='icp4_reduction1')(icp3_out)
        icp4_out1 = Conv2D(224,3,padding='same',activation='relu',name='icp4_out1')(icp4_reduction1)        
        icp4_reduction2 = Conv2D(24,1,padding='same',activation='relu',name='icp4_reduction2')(icp3_out)
        icp4_out2 = Conv2D(64,5,padding='same',activation='relu',name='icp4_out2')(icp4_reduction2)
        icp4_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name='icp4_pool')(icp3_out)
        icp4_out3 = Conv2D(64,1,padding='same',activation='relu',name='icp4_out3')(icp4_pool)
        icp4_out0 = Conv2D(160,1,padding='same',activation='relu',name='icp4_out0')(icp3_out)
        icp4_out = concatenate([icp4_out0, icp4_out1, icp4_out2, icp4_out3],axis=3,name='icp4_out')


        icp5_reduction1 = Conv2D(128,1,padding='same',activation='relu',name='icp5_reduction1')(icp4_out)
        icp5_out1 = Conv2D(256,3,padding='same',activation='relu',name='icp5_out1')(icp5_reduction1)
        icp5_reduction2 = Conv2D(24,1,padding='same',activation='relu',name='icp5_reduction2')(icp4_out)
        icp5_out2 = Conv2D(64,5,padding='same',activation='relu',name='icp5_out2')(icp5_reduction2)
        icp5_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name='icp5_pool')(icp4_out)
        icp5_out3 = Conv2D(64,1,padding='same',activation='relu',name='icp5_out3')(icp5_pool)
        icp5_out0 = Conv2D(128,1,padding='same',activation='relu',name='icp5_out0')(icp4_out)
        icp5_out = concatenate([icp5_out0, icp5_out1, icp5_out2, icp5_out3],axis=3,name='icp5_out')


        icp6_reduction1 = Conv2D(144,1,padding='same',activation='relu',name='icp6_reduction1')(icp5_out)
        icp6_out1 = Conv2D(288,3,padding='same',activation='relu',name='icp6_out1')(icp6_reduction1)
        icp6_reduction2 = Conv2D(32,1,padding='same',activation='relu',name='icp6_reduction2')(icp5_out)
        icp6_out2 = Conv2D(64,5,padding='same',activation='relu',name='icp6_out2')(icp6_reduction2)     
        icp6_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name='icp6_pool')(icp5_out)
        icp6_out3 = Conv2D(64,1,padding='same',activation='relu',name='icp6_out3')(icp6_pool)
        icp6_out0 = Conv2D(112,1,padding='same',activation='relu',name='icp6_out0')(icp5_out)
        icp6_out = concatenate([icp6_out0, icp6_out1, icp6_out2, icp6_out3],axis=3,name='icp6_out')
       

        ############################
        #second  output auxiliary 
        cls2_pool = AveragePooling2D(pool_size=(5,5),strides=(3,3),padding='valid',name='cls2_pool')(icp6_out)
        cls2_reduction_pose = Conv2D(128,1,padding='same',activation='relu',name='cls2_reduction_pose')(cls2_pool)
        cls2_fc1_flat = Flatten()(cls2_reduction_pose)
        cls2_fc1 = Dense(1024,activation='relu',name='cls2_fc1')(cls2_fc1_flat)
        cls2_fc_pose_xyz = Dense(3,name='cls2_fc_pose_xyz')(cls2_fc1)
        cls2_fc_fl = Dense(1,name='cls2_fc_fl')(cls2_fc1)
        cls2_fc_pose_wpqr = Dense(4,name='cls2_fc_pose_wpqr')(cls2_fc1)    




        icp7_reduction1 = Conv2D(160,1,padding='same',activation='relu',name='icp7_reduction1')(icp6_out)
        icp7_out1 = Conv2D(320,3,padding='same',activation='relu',name='icp7_out1')(icp7_reduction1)
        icp7_reduction2 = Conv2D(32,1,padding='same',activation='relu',name='icp7_reduction2')(icp6_out)
        icp7_out2 = Conv2D(128,5,padding='same',activation='relu',name='icp7_out2')(icp7_reduction2)
        icp7_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name='icp7_pool')(icp6_out)
        icp7_out3 = Conv2D(128,1,padding='same',activation='relu',name='icp7_out3')(icp7_pool)     
        icp7_out0 = Conv2D(256,1,padding='same',activation='relu',name='icp7_out0')(icp6_out)
        icp7_out = concatenate([icp7_out0, icp7_out1, icp7_out2, icp7_out3],axis=3,name='icp7_out')
  

       
        icp8_in = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same',name='icp8_in')(icp7_out)
        icp8_reduction1 = Conv2D(160,1,padding='same',activation='relu',name='icp8_reduction1')(icp8_in)
        icp8_out1 = Conv2D(320,3,padding='same',activation='relu',name='icp8_out1')(icp8_reduction1)
        icp8_reduction2 = Conv2D(32,1,padding='same',activation='relu',name='icp8_reduction2')(icp8_in)
        icp8_out2 = Conv2D(128,5,padding='same',activation='relu',name='icp8_out2')(icp8_reduction2)
        icp8_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name='icp8_pool')(icp8_in)
        icp8_out3 = Conv2D(128,1,padding='same',activation='relu',name='icp8_out3')(icp8_pool)   
        icp8_out0 = Conv2D(256,1,padding='same',activation='relu',name='icp8_out0')(icp8_in)
        icp8_out = concatenate([icp8_out0, icp8_out1, icp8_out2, icp8_out3],axis=3,name='icp8_out')
        



        icp9_reduction1 = Conv2D(192,1,padding='same',activation='relu',name='icp9_reduction1')(icp8_out)
        icp9_out1 = Conv2D(384,3,padding='same',activation='relu',name='icp9_out1')(icp9_reduction1)
        icp9_reduction2 = Conv2D(48,1,padding='same',activation='relu',name='icp9_reduction2')(icp8_out)
        icp9_out2 = Conv2D(128,5,padding='same',activation='relu',name='icp9_out2')(icp9_reduction2)
        icp9_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name='icp9_pool')(icp8_out)
        icp9_out3 = Conv2D(128,1,padding='same',activation='relu',name='icp9_out3')(icp9_pool)
        icp9_out0 = Conv2D(384,1,padding='same',activation='relu',name='icp9_out0')(icp8_out)
        icp9_out = concatenate([icp9_out0, icp9_out1, icp9_out2, icp9_out3],axis=3,name='icp9_out')
        
        

        ########################
        # thirsd  output auxiliary 
        cls3_pool = AveragePooling2D(pool_size=(7,7),strides=(1,1),padding='valid',name='cls3_pool')(icp9_out)
        cls3_fc1_flat = Flatten()(cls3_pool)
        cls3_fc1_pose = Dense(2048,activation='relu',name='cls3_fc1_pose')(cls3_fc1_flat)     
        cls3_fc_pose_xyz = Dense(3,name='cls3_fc_pose_xyz')(cls3_fc1_pose)
        cls3_fc_fl = Dense(1,name='cls3_fc_fl')(cls3_fc1_pose)        
        cls3_fc_pose_wpqr = Dense(4,name='cls3_fc_pose_wpqr')(cls3_fc1_pose)
        

      


        
        model = Model(input, [cls1_fc_pose_xyz, cls1_fc_pose_wpqr,cls1_fc_fl, cls2_fc_pose_xyz, cls2_fc_pose_wpqr,cls2_fc_fl, cls3_fc_pose_xyz, cls3_fc_pose_wpqr,cls3_fc_fl])
        print(model.summary())
        plot_model(model, to_file='model_plot_org.png', show_shapes=True, show_layer_names=True,rankdir='LR')
    return model
def create_googlenet1pose_full_backup(weights_path=None, tune=False):
    # creates Posenet from GoogLeNet a.k.a. Inception v1 (Szegedy, 2015)
    with tf.device('/cpu:0'):
        input1 = Input(shape=(224, 224, 3))
        input2 = Input(shape=(224, 224, 3))
        
        model1=create_featureExtractor(input1,'pre1')
        model2=create_featureExtractor(input2,'pre2')
        f1=model1.output
        f2=model2.output


        icp6_out = concatenate([f1,f2])

        icp7_reduction1 = Conv2D(160,1,padding='same',activation='relu',name='icp7_reduction1')(icp6_out)
        icp7_out1 = Conv2D(320,3,padding='same',activation='relu',name='icp7_out1')(icp7_reduction1)
        icp7_reduction2 = Conv2D(32,1,padding='same',activation='relu',name='icp7_reduction2')(icp6_out)
        icp7_out2 = Conv2D(128,5,padding='same',activation='relu',name='icp7_out2')(icp7_reduction2)
        icp7_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name='icp7_pool')(icp6_out)
        icp7_out3 = Conv2D(128,1,padding='same',activation='relu',name='icp7_out3')(icp7_pool)     
        icp7_out0 = Conv2D(256,1,padding='same',activation='relu',name='icp7_out0')(icp6_out)
        icp7_out = concatenate([icp7_out0, icp7_out1, icp7_out2, icp7_out3],axis=3,name='icp7_out')
  

       
        icp8_in = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same',name='icp8_in')(icp7_out)
        icp8_reduction1 = Conv2D(160,1,padding='same',activation='relu',name='icp8_reduction1')(icp8_in)
        icp8_out1 = Conv2D(320,3,padding='same',activation='relu',name='icp8_out1')(icp8_reduction1)
        icp8_reduction2 = Conv2D(32,1,padding='same',activation='relu',name='icp8_reduction2')(icp8_in)
        icp8_out2 = Conv2D(128,5,padding='same',activation='relu',name='icp8_out2')(icp8_reduction2)
        icp8_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name='icp8_pool')(icp8_in)
        icp8_out3 = Conv2D(128,1,padding='same',activation='relu',name='icp8_out3')(icp8_pool)   
        icp8_out0 = Conv2D(256,1,padding='same',activation='relu',name='icp8_out0')(icp8_in)
        icp8_out = concatenate([icp8_out0, icp8_out1, icp8_out2, icp8_out3],axis=3,name='icp8_out')
        



        icp9_reduction1 = Conv2D(192,1,padding='same',activation='relu',name='icp9_reduction1')(icp8_out)
        icp9_out1 = Conv2D(384,3,padding='same',activation='relu',name='icp9_out1')(icp9_reduction1)
        icp9_reduction2 = Conv2D(48,1,padding='same',activation='relu',name='icp9_reduction2')(icp8_out)
        icp9_out2 = Conv2D(128,5,padding='same',activation='relu',name='icp9_out2')(icp9_reduction2)
        icp9_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name='icp9_pool')(icp8_out)
        icp9_out3 = Conv2D(128,1,padding='same',activation='relu',name='icp9_out3')(icp9_pool)
        icp9_out0 = Conv2D(384,1,padding='same',activation='relu',name='icp9_out0')(icp8_out)
        icp9_out = concatenate([icp9_out0, icp9_out1, icp9_out2, icp9_out3],axis=3,name='icp9_out')
        
        

        #########################
        ## thirsd  output auxiliary 
        cls3_pool = AveragePooling2D(pool_size=(7,7),strides=(1,1),padding='valid',name='cls3_pool')(icp9_out)
        cls3_fc1_flat = Flatten()(cls3_pool)
        cls3_fc1_pose = Dense(1024,activation='relu',name='cls3_fc1_pose')(cls3_fc1_flat)     
        f3_pose_xyz = Dense(3,name='f3_pose_xyz')(cls3_fc1_pose)
        cls3_fc_fl = Dense(1,name='cls3_fc_fl')(cls3_fc1_pose)        
        f3_pose_wpqr = Dense(4,name='f3_pose_wpqr')(cls3_fc1_pose)

        

        cls2_pool = AveragePooling2D(pool_size=(5,5),strides=(3,3),padding='valid',name='cls2_pool')(f2)
        cls2_reduction_pose = Conv2D(128,1,padding='same',activation='relu',name='cls2_reduction_pose')(cls2_pool)
        cls2_fc1_flat = Flatten()(cls2_reduction_pose)
        cls2_fc1 = Dense(1024,activation='relu',name='cls2_fc1')(cls2_fc1_flat)
        f2_pose_xyz = Dense(3,name='f2_pose_xyz')(cls2_fc1)
        cls2_fc_fl = Dense(1,name='cls2_fc_fl')(cls2_fc1)
        f2_pose_wpqr = Dense(4,name='f2_pose_wpqr')(cls2_fc1)    



        cls1_pool = AveragePooling2D(pool_size=(5,5),strides=(3,3),padding='valid',name='cls1_pool')(f1)
        cls1_reduction_pose = Conv2D(128,1,padding='same',activation='relu',name='cls1_reduction_pose')(cls1_pool)
        cls1_fc1_flat = Flatten()(cls1_reduction_pose)
        cls1_fc1 = Dense(1024,activation='relu',name='cls1_fc1')(cls1_fc1_flat)
        f1_pose_xyz = Dense(3,name='f1_pose_xyz')(cls1_fc1)
        cls1_fc_fl = Dense(1,name='cls1_fc_fl')(cls1_fc1)
        f1_pose_wpqr = Dense(4,name='f1_pose_wpqr')(cls1_fc1)    


        #f3_fc1_pose = Dense(2048,activation='relu',name='f3_fc1_pose')(cls3_fc1_flat)
        #f3_pose_xyz = Dense(3,name='f3_pose_xyz')(f3_fc1_pose)
        #f3_pose_wpqr = Dense(4,name='f3_pose_wpqr')(f3_fc1_pose)

        
        posenet = Model([input1,input2], [f1_pose_xyz,f1_pose_wpqr,f2_pose_xyz,f2_pose_wpqr,f3_pose_xyz, f3_pose_wpqr])
        print(posenet.summary())
        plot_model(posenet, to_file='model_plot.png', show_shapes=True, show_layer_names=True,rankdir='LR')

    return posenet
def create_googlenet1pose_full(model_name="model1",width=455,length=256):
    with tf.device('/gpu:0'):
        input = Input(shape=(length, width, 3))
        
        conv1 = Conv2D(64,7,2,padding='same',activation='relu',name=model_name+'_'+'conv1')(input)        
        pool1 = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same',name=model_name+'_'+'pool1')(conv1)
        norm1 = BatchNormalization(axis=3, name=model_name+'_'+'norm1')(pool1)        
        reduction2 = Conv2D(64,1,padding='same',activation='relu',name=model_name+'_'+'reduction2')(norm1)        
        conv2 = Conv2D(192,3,padding='same',activation='relu',name=model_name+'_'+'conv2')(reduction2)
        norm2 = BatchNormalization(axis=3, name=model_name+'_'+'norm2')(conv2)
        pool2 = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='valid',name=model_name+'_'+'pool2')(norm2)
        icp1_reduction1 = Conv2D(96,1,1,padding='same',activation='relu',name=model_name+'_'+'icp1_reduction1')(pool2)
        icp1_out1 = Conv2D(128,3,padding='same',activation='relu',name=model_name+'_'+'icp1_out1')(icp1_reduction1)
        icp1_reduction2 = Conv2D(16,1,padding='same',activation='relu',name=model_name+'_'+'icp1_reduction2')(pool2)
        icp1_out2 = Conv2D(32,5,padding='same',activation='relu',name=model_name+'_'+'icp1_out2')(icp1_reduction2)   
        icp1_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name=model_name+'_'+'icp1_pool')(pool2)
        icp1_out3 = Conv2D(32,1,padding='same',activation='relu',name=model_name+'_'+'icp1_out3')(icp1_pool)       
        icp1_out0 = Conv2D(64,1,padding='same',activation='relu',name=model_name+'_'+'icp1_out0')(pool2)

        
        icp2_in = concatenate([icp1_out0, icp1_out1, icp1_out2, icp1_out3], axis=3, name=model_name+'_'+'icp2_in') 
        icp2_reduction1 = Conv2D(128,1,padding='same',activation='relu',name=model_name+'_'+'icp2_reduction1')(icp2_in)
        icp2_out1 = Conv2D(192,3,padding='same',activation='relu',name=model_name+'_'+'icp2_out1')(icp2_reduction1)     
        icp2_reduction2 = Conv2D(32,1,padding='same',activation='relu',name=model_name+'_'+'icp2_reduction2')(icp2_in)
        icp2_out2 = Conv2D(96,5,padding='same',activation='relu',name=model_name+'_'+'icp2_out2')(icp2_reduction2)
        icp2_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name=model_name+'_'+'icp2_pool')(icp2_in)
        icp2_out3 = Conv2D(64,1,padding='same',activation='relu',name=model_name+'_'+'icp2_out3')(icp2_pool)
        icp2_out0 = Conv2D(128,1,padding='same',activation='relu',name=model_name+'_'+'icp2_out0')(icp2_in)        
        icp2_out = concatenate([icp2_out0, icp2_out1, icp2_out2, icp2_out3],axis=3,name=model_name+'_'+'icp2_out')






        icp3_in = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same',name=model_name+'_'+'icp3_in')(icp2_out)
        icp3_reduction1 = Conv2D(96,1,padding='same',activation='relu',name=model_name+'_'+'icp3_reduction1')(icp3_in)
        icp3_out1 = Conv2D(208,3,padding='same',activation='relu',name=model_name+'_'+'icp3_out1')(icp3_reduction1)
        icp3_reduction2 = Conv2D(16,1,padding='same',activation='relu',name=model_name+'_'+'icp3_reduction2')(icp3_in)
        icp3_out2 = Conv2D(48,5,padding='same',activation='relu',name=model_name+'_'+'icp3_out2')(icp3_reduction2)      
        icp3_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name=model_name+'_'+'icp3_pool')(icp3_in)
        icp3_out3 = Conv2D(64,1,padding='same',activation='relu',name=model_name+'_'+'icp3_out3')(icp3_pool)
        icp3_out0 = Conv2D(192,1,padding='same',activation='relu',name=model_name+'_'+'icp3_out0')(icp3_in)       
        icp3_out = concatenate([icp3_out0, icp3_out1, icp3_out2, icp3_out3],axis=3,name=model_name+'_'+'icp3_out')
        
               
        icp4_reduction1 = Conv2D(112,1,padding='same',activation='relu',name=model_name+'_'+'icp4_reduction1')(icp3_out)
        icp4_out1 = Conv2D(224,3,padding='same',activation='relu',name=model_name+'_'+'icp4_out1')(icp4_reduction1)        
        icp4_reduction2 = Conv2D(24,1,padding='same',activation='relu',name=model_name+'_'+'icp4_reduction2')(icp3_out)
        icp4_out2 = Conv2D(64,5,padding='same',activation='relu',name=model_name+'_'+'icp4_out2')(icp4_reduction2)
        icp4_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name=model_name+'_'+'icp4_pool')(icp3_out)
        icp4_out3 = Conv2D(64,1,padding='same',activation='relu',name=model_name+'_'+'icp4_out3')(icp4_pool)
        icp4_out0 = Conv2D(160,1,padding='same',activation='relu',name=model_name+'_'+'icp4_out0')(icp3_out)
        icp4_out = concatenate([icp4_out0, icp4_out1, icp4_out2, icp4_out3],axis=3,name=model_name+'_'+'icp4_out')


        icp5_reduction1 = Conv2D(128,1,padding='same',activation='relu',name=model_name+'_'+'icp5_reduction1')(icp4_out)
        icp5_out1 = Conv2D(256,3,padding='same',activation='relu',name=model_name+'_'+'icp5_out1')(icp5_reduction1)
        icp5_reduction2 = Conv2D(24,1,padding='same',activation='relu',name=model_name+'_'+'icp5_reduction2')(icp4_out)
        icp5_out2 = Conv2D(64,5,padding='same',activation='relu',name=model_name+'_'+'icp5_out2')(icp5_reduction2)
        icp5_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name=model_name+'_'+'icp5_pool')(icp4_out)
        icp5_out3 = Conv2D(64,1,padding='same',activation='relu',name=model_name+'_'+'icp5_out3')(icp5_pool)
        icp5_out0 = Conv2D(128,1,padding='same',activation='relu',name=model_name+'_'+'icp5_out0')(icp4_out)
        icp5_out = concatenate([icp5_out0, icp5_out1, icp5_out2, icp5_out3],axis=3,name=model_name+'_'+'icp5_out')


        icp6_reduction1 = Conv2D(144,1,padding='same',activation='relu',name=model_name+'_'+'icp6_reduction1')(icp5_out)
        icp6_out1 = Conv2D(288,3,padding='same',activation='relu',name=model_name+'_'+'icp6_out1')(icp6_reduction1)
        icp6_reduction2 = Conv2D(32,1,padding='same',activation='relu',name=model_name+'_'+'icp6_reduction2')(icp5_out)
        icp6_out2 = Conv2D(64,5,padding='same',activation='relu',name=model_name+'_'+'icp6_out2')(icp6_reduction2)     
        icp6_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name=model_name+'_'+'icp6_pool')(icp5_out)
        icp6_out3 = Conv2D(64,1,padding='same',activation='relu',name=model_name+'_'+'icp6_out3')(icp6_pool)
        icp6_out0 = Conv2D(112,1,padding='same',activation='relu',name=model_name+'_'+'icp6_out0')(icp5_out)
        icp6_out = concatenate([icp6_out0, icp6_out1, icp6_out2, icp6_out3],axis=3,name=model_name+'_'+'icp6_out')
       


        icp7_reduction1 = Conv2D(160,1,padding='same',activation='relu',name=model_name+'_'+'icp7_reduction1')(icp6_out)
        icp7_out1 = Conv2D(320,3,padding='same',activation='relu',name=model_name+'_'+'icp7_out1')(icp7_reduction1)
        icp7_reduction2 = Conv2D(32,1,padding='same',activation='relu',name=model_name+'_'+'icp7_reduction2')(icp6_out)
        icp7_out2 = Conv2D(128,5,padding='same',activation='relu',name=model_name+'_'+'icp7_out2')(icp7_reduction2)
        icp7_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name=model_name+'_'+'icp7_pool')(icp6_out)
        icp7_out3 = Conv2D(128,1,padding='same',activation='relu',name=model_name+'_'+'icp7_out3')(icp7_pool)     
        icp7_out0 = Conv2D(256,1,padding='same',activation='relu',name=model_name+'_'+'icp7_out0')(icp6_out)
        icp7_out = concatenate([icp7_out0, icp7_out1, icp7_out2, icp7_out3],axis=3,name=model_name+'_'+'icp7_out')
  

       
        icp8_in = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same',name=model_name+'_'+'icp8_in')(icp7_out)
        icp8_reduction1 = Conv2D(160,1,padding='same',activation='relu',name=model_name+'_'+'icp8_reduction1')(icp8_in)
        icp8_out1 = Conv2D(320,3,padding='same',activation='relu',name=model_name+'_'+'icp8_out1')(icp8_reduction1)
        icp8_reduction2 = Conv2D(32,1,padding='same',activation='relu',name=model_name+'_'+'icp8_reduction2')(icp8_in)
        icp8_out2 = Conv2D(128,5,padding='same',activation='relu',name=model_name+'_'+'icp8_out2')(icp8_reduction2)
        icp8_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name=model_name+'_'+'icp8_pool')(icp8_in)
        icp8_out3 = Conv2D(128,1,padding='same',activation='relu',name=model_name+'_'+'icp8_out3')(icp8_pool)   
        icp8_out0 = Conv2D(256,1,padding='same',activation='relu',name=model_name+'_'+'icp8_out0')(icp8_in)
        icp8_out = concatenate([icp8_out0, icp8_out1, icp8_out2, icp8_out3],axis=3,name=model_name+'_'+'icp8_out')
        



        icp9_reduction1 = Conv2D(192,1,padding='same',activation='relu',name=model_name+'_'+'icp9_reduction1')(icp8_out)
        icp9_out1 = Conv2D(384,3,padding='same',activation='relu',name=model_name+'_'+'icp9_out1')(icp9_reduction1)
        icp9_reduction2 = Conv2D(48,1,padding='same',activation='relu',name=model_name+'_'+'icp9_reduction2')(icp8_out)
        icp9_out2 = Conv2D(128,5,padding='same',activation='relu',name=model_name+'_'+'icp9_out2')(icp9_reduction2)
        icp9_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name=model_name+'_'+'icp9_pool')(icp8_out)
        icp9_out3 = Conv2D(128,1,padding='same',activation='relu',name=model_name+'_'+'icp9_out3')(icp9_pool)
        icp9_out0 = Conv2D(384,1,padding='same',activation='relu',name=model_name+'_'+'icp9_out0')(icp8_out)
        icp9_out = concatenate([icp9_out0, icp9_out1, icp9_out2, icp9_out3],axis=3,name=model_name+'_'+'icp9_out')
        
        
        ############################
        # first output auxiliary        
        cls1_pool = AveragePooling2D(pool_size=(5,5),strides=(3,3),padding='valid',name=model_name+'_'+'cls1_pool')(icp3_out)        
        cls1_reduction_pose = Conv2D(128,1,padding='same',activation='relu',name=model_name+'_'+'cls1_reduction_pose')(cls1_pool)
        cls1_fc1_flat = Flatten()(cls1_reduction_pose)        
        cls1_fc1_pose = Dense(1024,activation='relu',name=model_name+'_'+'cls1_fc1_pose')(cls1_fc1_flat)
        cls1_fc_pose_xyz = Dense(3,name=model_name+'_'+'cls1_fc_pose_xyz')(cls1_fc1_pose)

        ############################
        #second  output auxiliary 
        cls2_pool = AveragePooling2D(pool_size=(5,5),strides=(3,3),padding='valid',name=model_name+'_'+'cls2_pool')(icp6_out)
        cls2_reduction_pose = Conv2D(128,1,padding='same',activation='relu',name=model_name+'_'+'cls2_reduction_pose')(cls2_pool)
        cls2_fc1_flat = Flatten()(cls2_reduction_pose)
        cls2_fc1 = Dense(1024,activation='relu',name=model_name+'_'+'cls2_fc1')(cls2_fc1_flat)
        cls2_fc_pose_xyz = Dense(3,name=model_name+'_'+'cls2_fc_pose_xyz')(cls2_fc1)

        ########################
        # third  output auxiliary 
        cls3_pool = AveragePooling2D(pool_size=(7,7),strides=(1,1),padding='valid',name=model_name+'_'+'cls3_pool')(icp9_out)
        cls3_fc1_flat = Flatten()(cls3_pool)
        cls3_fc1_pose = Dense(2048,activation='relu',name=model_name+'_'+'cls3_fc1_pose')(cls3_fc1_flat)     
        cls3_fc_pose_xyz = Dense(3,name=model_name+'_'+'cls3_fc_pose_xyz')(cls3_fc1_pose)
        

      


        
        model = Model(input, [cls1_fc_pose_xyz, cls2_fc_pose_xyz, cls3_fc_pose_xyz])
        print(model.summary())
        plot_model(model, to_file=model_name + 'model_plot_org.png', show_shapes=True, show_layer_names=True,rankdir='LR')
        return model
def create_googlenet1pose_full_separatedPose(input,model_name="model_1camera"):
    with tf.device('/gpu:0'):        
        conv1 = Conv2D(64,7,2,padding='same',activation='relu',name=model_name+'_'+'conv1')(input)        
        pool1 = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same',name=model_name+'_'+'pool1')(conv1)
        norm1 = BatchNormalization(axis=3, name=model_name+'_'+'norm1')(pool1)        
        reduction2 = Conv2D(64,1,padding='same',activation='relu',name=model_name+'_'+'reduction2')(norm1)        
        conv2 = Conv2D(192,3,padding='same',activation='relu',name=model_name+'_'+'conv2')(reduction2)
        norm2 = BatchNormalization(axis=3, name=model_name+'_'+'norm2')(conv2)
        pool2 = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='valid',name=model_name+'_'+'pool2')(norm2)
        icp1_reduction1 = Conv2D(96,1,1,padding='same',activation='relu',name=model_name+'_'+'icp1_reduction1')(pool2)
        icp1_out1 = Conv2D(128,3,padding='same',activation='relu',name=model_name+'_'+'icp1_out1')(icp1_reduction1)
        icp1_reduction2 = Conv2D(16,1,padding='same',activation='relu',name=model_name+'_'+'icp1_reduction2')(pool2)
        icp1_out2 = Conv2D(32,5,padding='same',activation='relu',name=model_name+'_'+'icp1_out2')(icp1_reduction2)   
        icp1_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name=model_name+'_'+'icp1_pool')(pool2)
        icp1_out3 = Conv2D(32,1,padding='same',activation='relu',name=model_name+'_'+'icp1_out3')(icp1_pool)       
        icp1_out0 = Conv2D(64,1,padding='same',activation='relu',name=model_name+'_'+'icp1_out0')(pool2)

        
        icp2_in = concatenate([icp1_out0, icp1_out1, icp1_out2, icp1_out3], axis=3, name=model_name+'_'+'icp2_in') 
        icp2_reduction1 = Conv2D(128,1,padding='same',activation='relu',name=model_name+'_'+'icp2_reduction1')(icp2_in)
        icp2_out1 = Conv2D(192,3,padding='same',activation='relu',name=model_name+'_'+'icp2_out1')(icp2_reduction1)     
        icp2_reduction2 = Conv2D(32,1,padding='same',activation='relu',name=model_name+'_'+'icp2_reduction2')(icp2_in)
        icp2_out2 = Conv2D(96,5,padding='same',activation='relu',name=model_name+'_'+'icp2_out2')(icp2_reduction2)
        icp2_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name=model_name+'_'+'icp2_pool')(icp2_in)
        icp2_out3 = Conv2D(64,1,padding='same',activation='relu',name=model_name+'_'+'icp2_out3')(icp2_pool)
        icp2_out0 = Conv2D(128,1,padding='same',activation='relu',name=model_name+'_'+'icp2_out0')(icp2_in)        
        icp2_out = concatenate([icp2_out0, icp2_out1, icp2_out2, icp2_out3],axis=3,name=model_name+'_'+'icp2_out')






        icp3_in = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same',name=model_name+'_'+'icp3_in')(icp2_out)
        icp3_reduction1 = Conv2D(96,1,padding='same',activation='relu',name=model_name+'_'+'icp3_reduction1')(icp3_in)
        icp3_out1 = Conv2D(208,3,padding='same',activation='relu',name=model_name+'_'+'icp3_out1')(icp3_reduction1)
        icp3_reduction2 = Conv2D(16,1,padding='same',activation='relu',name=model_name+'_'+'icp3_reduction2')(icp3_in)
        icp3_out2 = Conv2D(48,5,padding='same',activation='relu',name=model_name+'_'+'icp3_out2')(icp3_reduction2)      
        icp3_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name=model_name+'_'+'icp3_pool')(icp3_in)
        icp3_out3 = Conv2D(64,1,padding='same',activation='relu',name=model_name+'_'+'icp3_out3')(icp3_pool)
        icp3_out0 = Conv2D(192,1,padding='same',activation='relu',name=model_name+'_'+'icp3_out0')(icp3_in)       
        icp3_out = concatenate([icp3_out0, icp3_out1, icp3_out2, icp3_out3],axis=3,name=model_name+'_'+'icp3_out')
        
               
        icp4_reduction1 = Conv2D(112,1,padding='same',activation='relu',name=model_name+'_'+'icp4_reduction1')(icp3_out)
        icp4_out1 = Conv2D(224,3,padding='same',activation='relu',name=model_name+'_'+'icp4_out1')(icp4_reduction1)        
        icp4_reduction2 = Conv2D(24,1,padding='same',activation='relu',name=model_name+'_'+'icp4_reduction2')(icp3_out)
        icp4_out2 = Conv2D(64,5,padding='same',activation='relu',name=model_name+'_'+'icp4_out2')(icp4_reduction2)
        icp4_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name=model_name+'_'+'icp4_pool')(icp3_out)
        icp4_out3 = Conv2D(64,1,padding='same',activation='relu',name=model_name+'_'+'icp4_out3')(icp4_pool)
        icp4_out0 = Conv2D(160,1,padding='same',activation='relu',name=model_name+'_'+'icp4_out0')(icp3_out)
        icp4_out = concatenate([icp4_out0, icp4_out1, icp4_out2, icp4_out3],axis=3,name=model_name+'_'+'icp4_out')


        icp5_reduction1 = Conv2D(128,1,padding='same',activation='relu',name=model_name+'_'+'icp5_reduction1')(icp4_out)
        icp5_out1 = Conv2D(256,3,padding='same',activation='relu',name=model_name+'_'+'icp5_out1')(icp5_reduction1)
        icp5_reduction2 = Conv2D(24,1,padding='same',activation='relu',name=model_name+'_'+'icp5_reduction2')(icp4_out)
        icp5_out2 = Conv2D(64,5,padding='same',activation='relu',name=model_name+'_'+'icp5_out2')(icp5_reduction2)
        icp5_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name=model_name+'_'+'icp5_pool')(icp4_out)
        icp5_out3 = Conv2D(64,1,padding='same',activation='relu',name=model_name+'_'+'icp5_out3')(icp5_pool)
        icp5_out0 = Conv2D(128,1,padding='same',activation='relu',name=model_name+'_'+'icp5_out0')(icp4_out)
        icp5_out = concatenate([icp5_out0, icp5_out1, icp5_out2, icp5_out3],axis=3,name=model_name+'_'+'icp5_out')


        icp6_reduction1 = Conv2D(144,1,padding='same',activation='relu',name=model_name+'_'+'icp6_reduction1')(icp5_out)
        icp6_out1 = Conv2D(288,3,padding='same',activation='relu',name=model_name+'_'+'icp6_out1')(icp6_reduction1)
        icp6_reduction2 = Conv2D(32,1,padding='same',activation='relu',name=model_name+'_'+'icp6_reduction2')(icp5_out)
        icp6_out2 = Conv2D(64,5,padding='same',activation='relu',name=model_name+'_'+'icp6_out2')(icp6_reduction2)     
        icp6_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name=model_name+'_'+'icp6_pool')(icp5_out)
        icp6_out3 = Conv2D(64,1,padding='same',activation='relu',name=model_name+'_'+'icp6_out3')(icp6_pool)
        icp6_out0 = Conv2D(112,1,padding='same',activation='relu',name=model_name+'_'+'icp6_out0')(icp5_out)
        icp6_out = concatenate([icp6_out0, icp6_out1, icp6_out2, icp6_out3],axis=3,name=model_name+'_'+'icp6_out')
       


        icp7_reduction1 = Conv2D(160,1,padding='same',activation='relu',name=model_name+'_'+'icp7_reduction1')(icp6_out)
        icp7_out1 = Conv2D(320,3,padding='same',activation='relu',name=model_name+'_'+'icp7_out1')(icp7_reduction1)
        icp7_reduction2 = Conv2D(32,1,padding='same',activation='relu',name=model_name+'_'+'icp7_reduction2')(icp6_out)
        icp7_out2 = Conv2D(128,5,padding='same',activation='relu',name=model_name+'_'+'icp7_out2')(icp7_reduction2)
        icp7_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name=model_name+'_'+'icp7_pool')(icp6_out)
        icp7_out3 = Conv2D(128,1,padding='same',activation='relu',name=model_name+'_'+'icp7_out3')(icp7_pool)     
        icp7_out0 = Conv2D(256,1,padding='same',activation='relu',name=model_name+'_'+'icp7_out0')(icp6_out)
        icp7_out = concatenate([icp7_out0, icp7_out1, icp7_out2, icp7_out3],axis=3,name=model_name+'_'+'icp7_out')
  

       
        icp8_in = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same',name=model_name+'_'+'icp8_in')(icp7_out)
        icp8_reduction1 = Conv2D(160,1,padding='same',activation='relu',name=model_name+'_'+'icp8_reduction1')(icp8_in)
        icp8_out1 = Conv2D(320,3,padding='same',activation='relu',name=model_name+'_'+'icp8_out1')(icp8_reduction1)
        icp8_reduction2 = Conv2D(32,1,padding='same',activation='relu',name=model_name+'_'+'icp8_reduction2')(icp8_in)
        icp8_out2 = Conv2D(128,5,padding='same',activation='relu',name=model_name+'_'+'icp8_out2')(icp8_reduction2)
        icp8_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name=model_name+'_'+'icp8_pool')(icp8_in)
        icp8_out3 = Conv2D(128,1,padding='same',activation='relu',name=model_name+'_'+'icp8_out3')(icp8_pool)   
        icp8_out0 = Conv2D(256,1,padding='same',activation='relu',name=model_name+'_'+'icp8_out0')(icp8_in)
        icp8_out = concatenate([icp8_out0, icp8_out1, icp8_out2, icp8_out3],axis=3,name=model_name+'_'+'icp8_out')
        



        icp9_reduction1 = Conv2D(192,1,padding='same',activation='relu',name=model_name+'_'+'icp9_reduction1')(icp8_out)
        icp9_out1 = Conv2D(384,3,padding='same',activation='relu',name=model_name+'_'+'icp9_out1')(icp9_reduction1)
        icp9_reduction2 = Conv2D(48,1,padding='same',activation='relu',name=model_name+'_'+'icp9_reduction2')(icp8_out)
        icp9_out2 = Conv2D(128,5,padding='same',activation='relu',name=model_name+'_'+'icp9_out2')(icp9_reduction2)
        icp9_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name=model_name+'_'+'icp9_pool')(icp8_out)
        icp9_out3 = Conv2D(128,1,padding='same',activation='relu',name=model_name+'_'+'icp9_out3')(icp9_pool)
        icp9_out0 = Conv2D(384,1,padding='same',activation='relu',name=model_name+'_'+'icp9_out0')(icp8_out)
        icp9_out = concatenate([icp9_out0, icp9_out1, icp9_out2, icp9_out3],axis=3,name=model_name+'_'+'icp9_out')
        
        
        ############################
        # first output auxiliary        
        cls1_pool = AveragePooling2D(pool_size=(5,5),strides=(3,3),padding='valid',name=model_name+'_'+'cls1_pool')(icp3_out)        
        cls1_reduction_pose = Conv2D(128,1,padding='same',activation='relu',name=model_name+'_'+'cls1_reduction_pose')(cls1_pool)
        cls1_fc1_flat = Flatten()(cls1_reduction_pose)        
        cls1_fc1_pose = Dense(2048,activation='relu',name=model_name+'_'+'cls1_fc1_pose')(cls1_fc1_flat)    
        cls1_fc_pose_xy = Dense(2,name=model_name+'_'+'cls1_fc_pose_xy')(cls1_fc1_pose)   
        cls1_fc_pose_r  = Dense(1,name=model_name+'_'+'cls1_fc_pose_r')(cls1_fc1_pose)

        ############################
        #second  output auxiliary 
        cls2_pool = AveragePooling2D(pool_size=(5,5),strides=(3,3),padding='valid',name=model_name+'_'+'cls2_pool')(icp6_out)
        cls2_reduction_pose = Conv2D(128,1,padding='same',activation='relu',name=model_name+'_'+'cls2_reduction_pose')(cls2_pool)
        cls2_fc1_flat = Flatten()(cls2_reduction_pose)
        cls2_fc1_pose = Dense(2048,activation='relu',name=model_name+'_'+'cls2_fc1')(cls2_fc1_flat)    
        cls2_fc_pose_xy = Dense(2,name=model_name+'_'+'cls2_fc_pose_xy')(cls2_fc1_pose)   
        cls2_fc_pose_r  = Dense(1,name=model_name+'_'+'cls2_fc_pose_r')(cls2_fc1_pose)

        ########################
        # third  output auxiliary 
        cls3_pool = AveragePooling2D(pool_size=(7,7),strides=(1,1),padding='valid',name=model_name+'_'+'cls3_pool')(icp9_out)
        cls3_fc1_flat = Flatten()(cls3_pool)
        cls3_fc1_pose = Dense(2048,activation='relu',name=model_name+'_'+'cls3_fc1_pose')(cls3_fc1_flat)     
        cls3_fc_pose_xy = Dense(2,name=model_name+'_'+'cls3_fc_pose_xy')(cls3_fc1_pose)   
        cls3_fc_pose_r = Dense(1,name=model_name+'_'+'cls3_fc_pose_r')(cls3_fc1_pose)
        

      


        
        model = Model(input, [cls1_fc_pose_xy,cls1_fc_pose_r, cls2_fc_pose_xy,cls2_fc_pose_r, cls3_fc_pose_xy,cls3_fc_pose_r])
        print(model.summary())
        plot_model(model, to_file=model_name + '_net.png', show_shapes=True, show_layer_names=True,rankdir='LR')
        return model


def res_identity(x, filters): 
  #renet block where dimension doesnot change.
  #The skip connection is just simple identity conncection
  #we will have 3 blocks and then input will be added

  x_skip = x # this will be used for addition with the residual block 
  f1, f2 = filters

  #first block 
  x = Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
  x = BatchNormalization()(x)
  x = Activation(activations.relu)(x)

  #second block # bottleneck (but size kept same with padding)
  x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001))(x)
  x = BatchNormalization()(x)
  x = Activation(activations.relu)(x)

  # third block activation used after adding the input
  x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
  x = BatchNormalization()(x)
  # x = Activation(activations.relu)(x)

  # add the input 
  x = Add()([x, x_skip])
  x = Activation(activations.relu)(x)

  return x
def res_conv(x, s, filters):
  '''
  here the input size changes''' 
  x_skip = x
  f1, f2 = filters

  
  # first block
  x = Conv2D(f1, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=l2(0.001))(x)
  # when s = 2 then it is like downsizing the feature map
  x = BatchNormalization()(x)
  x = Activation(activations.relu)(x)

  # second block
  x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001))(x)
  x = BatchNormalization()(x)
  x = Activation(activations.relu)(x)

  #third block
  x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
  x = BatchNormalization()(x)

  # shortcut 
  x_skip = Conv2D(f2, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=l2(0.001))(x_skip)
  x_skip = BatchNormalization()(x_skip)

  # add 
  x = Add()([x, x_skip])
  x = Activation(activations.relu)(x)

  return x
def resnet50(input_im,name):
  x = ZeroPadding2D(padding=(3, 3))(input_im)

  # 1st stage
  # here we perform maxpooling, see the figure above

  x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2))(x)
  x = BatchNormalization()(x)
  
  x = Activation(activations.relu)(x)
  x = MaxPooling2D((3, 3), strides=(2, 2))(x)

  #2nd stage 
  # frm here on only conv block and identity block, no pooling

  x = res_conv(x, s=1, filters=(64, 256))
  x = res_identity(x, filters=(64, 256))
  x = res_identity(x, filters=(64, 256))

  # 3rd stage

  x = res_conv(x, s=2, filters=(128, 512))
  x = res_identity(x, filters=(128, 512))
  x = res_identity(x, filters=(128, 512))
  x = res_identity(x, filters=(128, 512))

  # 4th stage

  x = res_conv(x, s=2, filters=(256, 1024))
  x = res_identity(x, filters=(256, 1024))
  x = res_identity(x, filters=(256, 1024))
  x = res_identity(x, filters=(256, 1024))
  x = res_identity(x, filters=(256, 1024))
  x = res_identity(x, filters=(256, 1024))

  # 5th stage

  x = res_conv(x, s=2, filters=(512, 2048))
  x = res_identity(x, filters=(512, 2048))
  x = res_identity(x, filters=(512, 2048))

  ## ends with average pooling and dense connection

  #x = AveragePooling2D((2, 2), padding='same')(x)

  #x = Flatten()(x)
  #x = Dense(len(class_types), activation='softmax', kernel_initializer='he_normal')(x) #multi-class

  # define the model 

  model = Model(inputs=input_im, outputs=x, name=name+'Resnet50')

  return model



def create_poseres_3camera(weights_path=None, tune=False):
    # creates Posenet from GoogLeNet a.k.a. Inception v1 (Szegedy, 2015)
    with tf.device('/GPU:0'):
        input1 = Input(shape=(216,384, 3),name='Input1')
        input2 = Input(shape=(216,384, 3),name='Input2')
        input3 = Input(shape=(216,384, 3),name='Input3')
        
        model1=resnet50(input1,'pre1')
        model2=resnet50(input2,'pre2')
        model3=resnet50(input3,'pre3')

        f1=model1.output
        f2=model2.output
        f3=model3.output


        icp6_out = concatenate([f1,f2,f3])
        
        icp6_inc1 = Conv2D(2048,1,padding='same',activation='relu',name='icp6_inc1')(icp6_out)
        icp6_reduction1 = Conv2D(1024,1,padding='same',activation='relu',name='icp6_reduction1')(icp6_inc1)
        icp7_reduction1 = Conv2D(540,1,padding='same',activation='relu',name='icp7_reduction1')(icp6_reduction1)
        icp7_out1 = Conv2D(320,3,padding='same',activation='relu',name='icp7_out1')(icp7_reduction1)
        icp7_reduction2 = Conv2D(32,1,padding='same',activation='relu',name='icp7_reduction2')(icp6_out)
        icp7_out2 = Conv2D(128,5,padding='same',activation='relu',name='icp7_out2')(icp7_reduction2)
        icp7_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name='icp7_pool')(icp6_out)
        icp7_out3 = Conv2D(128,1,padding='same',activation='relu',name='icp7_out3')(icp7_pool)     
        icp7_out0 = Conv2D(256,1,padding='same',activation='relu',name='icp7_out0')(icp6_out)
        icp7_out = concatenate([icp7_out0, icp7_out1, icp7_out2, icp7_out3],axis=3,name='icp7_out')
  

       
        icp8_in = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same',name='icp8_in')(icp7_out)
        icp8_reduction1 = Conv2D(160,1,padding='same',activation='relu',name='icp8_reduction1')(icp8_in)
        icp8_out1 = Conv2D(320,3,padding='same',activation='relu',name='icp8_out1')(icp8_reduction1)
        icp8_reduction2 = Conv2D(32,1,padding='same',activation='relu',name='icp8_reduction2')(icp8_in)
        icp8_out2 = Conv2D(128,5,padding='same',activation='relu',name='icp8_out2')(icp8_reduction2)
        icp8_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name='icp8_pool')(icp8_in)
        icp8_out3 = Conv2D(128,1,padding='same',activation='relu',name='icp8_out3')(icp8_pool)   
        icp8_out0 = Conv2D(256,1,padding='same',activation='relu',name='icp8_out0')(icp8_in)
        icp8_out = concatenate([icp8_out0, icp8_out1, icp8_out2, icp8_out3],axis=3,name='icp8_out')
        



        icp9_reduction1 = Conv2D(192,1,padding='same',activation='relu',name='icp9_reduction1')(icp8_out)
        icp9_out1 = Conv2D(384,3,padding='same',activation='relu',name='icp9_out1')(icp9_reduction1)
        icp9_reduction2 = Conv2D(48,1,padding='same',activation='relu',name='icp9_reduction2')(icp8_out)
        icp9_out2 = Conv2D(128,5,padding='same',activation='relu',name='icp9_out2')(icp9_reduction2)
        icp9_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name='icp9_pool')(icp8_out)
        icp9_out3 = Conv2D(128,1,padding='same',activation='relu',name='icp9_out3')(icp9_pool)
        icp9_out0 = Conv2D(384,1,padding='same',activation='relu',name='icp9_out0')(icp8_out)
        icp9_out = concatenate([icp9_out0, icp9_out1, icp9_out2, icp9_out3],axis=3,name='icp9_out')
        
        

        #########################
        ## thirsd  output auxiliary 
        #cls4_pool = AveragePooling2D(pool_size=(7,7),strides=(1,1),padding='valid',name='cls4_pool')(icp9_out)
        cls4_fc1_flat = Flatten()(icp9_out)
        cls4_fc1_pose = Dense(1024,activation='relu',name='cls4_fc1_pose')(cls4_fc1_flat)     
        f4_pose_xyz = Dense(3,name='f4_pose_xyz')(cls4_fc1_pose)

        

        cls3_pool = AveragePooling2D(pool_size=(5,5),strides=(3,3),padding='valid',name='cls3_pool')(f3)
        cls3_reduction_pose = Conv2D(128,1,padding='same',activation='relu',name='cls3_reduction_pose')(cls3_pool)
        cls3_fc1_flat = Flatten()(cls3_reduction_pose)
        cls3_fc1 = Dense(1024,activation='relu',name='cls3_fc1')(cls3_fc1_flat)
        f3_pose_xyz = Dense(3,name='f3_pose_xyz')(cls3_fc1)  

        cls2_pool = AveragePooling2D(pool_size=(5,5),strides=(3,3),padding='valid',name='cls2_pool')(f2)
        cls2_reduction_pose = Conv2D(128,1,padding='same',activation='relu',name='cls2_reduction_pose')(cls2_pool)
        cls2_fc1_flat = Flatten()(cls2_reduction_pose)
        cls2_fc1 = Dense(1024,activation='relu',name='cls2_fc1')(cls2_fc1_flat)
        f2_pose_xyz = Dense(3,name='f2_pose_xyz')(cls2_fc1) 



        cls1_pool = AveragePooling2D(pool_size=(5,5),strides=(3,3),padding='valid',name='cls1_pool')(f1)
        cls1_reduction_pose = Conv2D(128,1,padding='same',activation='relu',name='cls1_reduction_pose')(cls1_pool)
        cls1_fc1_flat = Flatten()(cls1_reduction_pose)
        cls1_fc1 = Dense(1024,activation='relu',name='cls1_fc1')(cls1_fc1_flat)
        f1_pose_xyz = Dense(3,name='f1_pose_xyz')(cls1_fc1)



        
        posenet = Model([input1,input2,input3], [f1_pose_xyz,f2_pose_xyz,f3_pose_xyz,f4_pose_xyz])
        print(posenet.summary())
        plot_model(posenet, to_file='model_plot.png', show_shapes=True, show_layer_names=True,rankdir='LR')

    return posenet
def create_poseres_2camera(weights_path=None, tune=False):
    # creates Posenet from GoogLeNet a.k.a. Inception v1 (Szegedy, 2015)
    with tf.device('/GPU:0'):
        input1 = Input(shape=(224,224, 3),name='Input1')
        input2 = Input(shape=(224,224, 3),name='Input2')
        
        model1=resnet50(input1,'pre1')
        model2=resnet50(input2,'pre2')

        f1=model1.output
        f2=model2.output


        icp6_out = concatenate([f1,f2])
        
        icp6_inc1 = Conv2D(540,1,padding='same',activation='relu',name='icp6_inc1')(icp6_out)
        # icp6_reduction1 = Conv2D(1024,1,padding='same',activation='relu',name='icp6_reduction1')(icp6_inc1)
        # icp7_reduction1 = Conv2D(540,1,padding='same',activation='relu',name='icp7_reduction1')(icp6_reduction1)

        cls4_fc1_flat = Flatten()(icp6_inc1)
        cls4_fc1_pose = Dense(1024,activation='relu',name='cls4_fc1_pose')(cls4_fc1_flat)     
        f4_pose_xyz = Dense(3,name='f4_pose_xyz')(cls4_fc1_pose)

        

        cls3_pool = AveragePooling2D(pool_size=(5,5),strides=(3,3),padding='valid',name='cls3_pool')(f1)
        cls3_reduction_pose = Conv2D(128,1,padding='same',activation='relu',name='cls3_reduction_pose')(cls3_pool)
        cls3_fc1_flat = Flatten()(cls3_reduction_pose)
        cls3_fc1 = Dense(1024,activation='relu',name='cls3_fc1')(cls3_fc1_flat)
        f3_pose_xyz = Dense(3,name='f3_pose_xyz')(cls3_fc1)  



        cls2_pool = AveragePooling2D(pool_size=(5,5),strides=(3,3),padding='valid',name='cls2_pool')(f2)
        cls2_reduction_pose = Conv2D(128,1,padding='same',activation='relu',name='cls2_reduction_pose')(cls2_pool)
        cls2_fc1_flat = Flatten()(cls2_reduction_pose)
        cls2_fc1 = Dense(1024,activation='relu',name='cls2_fc1')(cls2_fc1_flat)
        f2_pose_xyz = Dense(3,name='f2_pose_xyz')(cls2_fc1)  


        
        posenet = Model([input1,input2], [f2_pose_xyz,f3_pose_xyz,f4_pose_xyz])
        #print(posenet.summary())
        #plot_model(posenet, to_file='model_plot.png', show_shapes=True, show_layer_names=True,rankdir='LR')

    return posenet
def create_poseres_1camera(input, name="model1_1camera"):
    # creates Posenet from GoogLeNet a.k.a. Inception v1 (Szegedy, 2015)
    if True:
        input1 = input        
        model1=resnet50(input1,name)
        f1=model1.output
        icp6_inc1 = Conv2D(540,1,padding='same',activation='relu',name=name+'_'+'icp6_inc1')(f1)
        cls2_fc1_flat = Flatten()(icp6_inc1) 
        cls2_fc_pose_xy = Dense(2,name=name+'_'+'cls2_fc_pose_xy')(cls2_fc1_flat)   
        cls2_fc_pose_r = Dense(1,name=name+'_'+'cls2_fc_pose_r')(cls2_fc1_flat)

        

        cls3_pool = AveragePooling2D(pool_size=(5,5),strides=(3,3),padding='valid',name=name+'_'+'cls3_pool')(f1)
        cls3_reduction_pose = Conv2D(128,1,padding='same',activation='relu',name=name+'_'+'cls3_reduction_pose')(cls3_pool)
        cls3_fc1_flat = Flatten()(cls3_reduction_pose)
        cls3_fc_pose_xy = Dense(2,name=name+'_'+'cls3_fc_pose_xy')(cls3_fc1_flat)   
        cls3_fc_pose_r = Dense(1,name=name+'_'+'cls3_fc_pose_r')(cls3_fc1_flat)




        
        model = Model(input1, [cls2_fc_pose_xy,cls2_fc_pose_r,cls3_fc_pose_xy,cls3_fc_pose_r])
        print(model.summary())
        plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True,rankdir='LR')

    return model
def create_resnet1pose_full_separatedPose(input,name="model1_1camera"):
 with tf.device('/gpu:0'):
  x = ZeroPadding2D(padding=(3, 3))(input)

  # 1st stage
  # here we perform maxpooling, see the figure above

  x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2))(x)
  x = BatchNormalization()(x)
  
  x = Activation(activations.relu)(x)
  x = MaxPooling2D((3, 3), strides=(2, 2))(x)

  #2nd stage 
  # frm here on only conv block and identity block, no pooling

  x = res_conv(x, s=1, filters=(64, 256))
  x = res_identity(x, filters=(64, 256))
  x = res_identity(x, filters=(64, 256))

  # 3rd stage

  x = res_conv(x, s=2, filters=(128, 512))
  x = res_identity(x, filters=(128, 512))
  x = res_identity(x, filters=(128, 512))
  x = res_identity(x, filters=(128, 512))

  # 4th stage

  x = res_conv(x, s=2, filters=(256, 1024))
  x = res_identity(x, filters=(256, 1024))
  x = res_identity(x, filters=(256, 1024))
  x = res_identity(x, filters=(256, 1024))
  x = res_identity(x, filters=(256, 1024))
  x = res_identity(x, filters=(256, 1024))

  # 5th stage

  x = res_conv(x, s=2, filters=(512, 2048))
  x = res_identity(x, filters=(512, 2048))
  x = res_identity(x, filters=(512, 2048))

  ## ends with average pooling and dense connection
  cls3_pool = AveragePooling2D(pool_size=(2,2),strides=(1,1),padding='valid')(x)
  cls3_fc1_flat = Flatten()(cls3_pool)

  cls3_fc1_pose = Dense(1024,activation='relu',name=name+'_'+'cls3_fc1_pose')(cls3_fc1_flat)     
  cls3_fc_pose_xy = Dense(2,name=name+'_'+'cls3_fc_pose_xy')(cls3_fc1_pose)   
  cls3_fc_pose_r = Dense(1,name=name+'_'+'cls3_fc_pose_r')(cls3_fc1_pose)

  #cls3_fc1_pose = Dense(1024,activation='relu')(cls3_fc1_flat)    
  #cls3_fc_pose_xy = Dense(2, activation='softmax', kernel_initializer='he_normal',name=name+'_'+'cls3_fc_pose_xy')(cls3_fc1_pose)   
  #cls3_fc_pose_r = Dense(1, activation='softmax', kernel_initializer='he_normal',name=name+'_'+'cls3_fc_pose_r')(cls3_fc1_pose)

  #define the model 
  model = Model(input, [cls3_fc_pose_xy,cls3_fc_pose_r])
  print(model.summary())
  plot_model(model, to_file=name + '_net.png', show_shapes=True, show_layer_names=True,rankdir='LR')
  return model






def auxiliary(model_name,input,aux_name):
    trans_L1_a2 = Conv2D(128,2,padding='same',activation='relu',name=model_name+"_"+'trans_L1'+aux_name)(input)
    trans_MP2_a2 = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name=model_name+"_"+'trans_MP2'+aux_name)(trans_L1_a2)    
    trans_flat_a2 = Flatten()(trans_MP2_a2)
    trans_fc_xy_a2 = Dense(512,activation='relu',name=model_name+"_"+'trans_fc_xy'+aux_name)(trans_flat_a2)     
    pose_xy = Dense(2,name=model_name+"_"+'pose_xy'+aux_name)(trans_fc_xy_a2)


    rot_L1_a2 = Conv2D(128,1,padding='same',activation='relu',name=model_name+"_"+'rot_L1'+aux_name)(input)
    rot_MP2_a2 = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name=model_name+"_"+'rot_MP2'+aux_name)(rot_L1_a2)    
    rot_flat_a2 = Flatten()(rot_MP2_a2)
    rot_fc_xy_a2 = Dense(512,activation='relu',name=model_name+"_"+'rot_fc_xy'+aux_name)(rot_flat_a2)     
    pose_r = Dense(1,name=model_name+"_"+'pose_r'+aux_name)(rot_fc_xy_a2)
    return [pose_xy,pose_r]
def regressor(model_name,input):
    trans_L1 = Conv2D(64,2,padding='same',activation='relu',name=model_name+"_"+'trans_L1')(input)
    trans_L2 = Conv2D(64,1,padding='same',activation='relu',name=model_name+"_"+'trans_L2')(trans_L1)
    trans_MP1 = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name=model_name+"_"+'trans_MP1')(trans_L2)
    trans_L3 = Conv2D(64,1,padding='same',activation='relu',name=model_name+"_"+'trans_L3')(trans_MP1)
    trans_L4 = Conv2D(128,1,padding='same',activation='relu',name=model_name+"_"+'trans_L4')(trans_L3)
    trans_MP2 = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name=model_name+"_"+'trans_MP2')(trans_L4)    
    trans_flat = Flatten()(trans_MP2)
    trans_fc_xy = Dense(2048,activation='relu',name=model_name+"_"+'trans_fc_xy')(trans_flat)     
    pose_xy = Dense(2,name=model_name+"_"+'pose_xy')(trans_fc_xy)


    rot_L1 = Conv2D(128,1,padding='same',activation='relu',name=model_name+"_"+'rot_L1')(input)
    rot_L2 = Conv2D(128,1,padding='same',activation='relu',name=model_name+"_"+'rot_L2')(rot_L1)
    rot_MP1 = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name=model_name+"_"+'rot_MP1')(rot_L2)
    rot_L3 = Conv2D(128,1,padding='same',activation='relu',name=model_name+"_"+'rot_L3')(rot_MP1)
    rot_L4 = Conv2D(64,1,padding='same',activation='relu',name=model_name+"_"+'rot_L4')(rot_L3)
    rot_MP2 = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name=model_name+"_"+'rot_MP2')(rot_L4)    
    rot_flat = Flatten()(rot_MP2)
    rot_fc_xy = Dense(2048,activation='relu',name=model_name+"_"+'rot_fc_xy')(rot_flat)     
    pose_r = Dense(1,name=model_name+"_"+'pose_r')(rot_fc_xy)
    return [pose_xy, pose_r]


def NasNet_Large(input_layer,model_name,Tune,Custom_Size=False,AuxiliaryLoss=False):
    if(Tune or Custom_Size):
        base_model = NASNetLarge(weights=None, input_tensor=input_layer)
    else:        
        base_model = NASNetLarge(input_tensor=input_layer)

    
    out=base_model.get_layer(index=-3).output

    out1=base_model.get_layer("activation_235").output
    out2=base_model.get_layer("activation_248").output

    #print(base_model.summary())
  

    #First Output
    [pose_xy,pose_r] = regressor(model_name,out)
    
    #First Auxiliary
    [pose_xy1,pose_r1] = auxiliary(model_name,out1,'_a1')

    #Second Auxiliary
    [pose_xy2,pose_r2] = auxiliary(model_name,out1,'_a2')
    

    if(AuxiliaryLoss==True):
        model = Model(input_layer, [pose_xy1,pose_r1, pose_xy2,pose_r2, pose_xy,pose_r])
    else:
        model = Model(input_layer, [pose_xy,pose_r])
    return model


def InceptionResNet_V2(input_layer,model_name,Tune,Custom_Size=False,AuxiliaryLoss=False,BlockFirsts=False):
    if(Tune or Custom_Size):
        base_model = InceptionResNetV2(weights=None, input_tensor=input_layer)
    else:        
        base_model = InceptionResNetV2(input_tensor=input_layer)

    if(BlockFirsts==True):
       for layer in base_model.layers[:-4]:
         layer.trainable = False
       for layer in base_model.layers[:-4]:
         layer.trainable = False
         
    out=base_model.get_layer(index=-3).output
   
    [pose_xy,pose_r] = regressor(model_name,out)
    
    

    if(AuxiliaryLoss==True):
        out1=base_model.get_layer("block8_1_mixed").output
        out2=base_model.get_layer("block8_7_ac").output
        #First Auxiliary
        [pose_xy1,pose_r1] = auxiliary(model_name,out1,'_a1')

        #Second Auxiliary
        [pose_xy2,pose_r2] = auxiliary(model_name,out1,'_a2')
        model = Model(input_layer, [pose_xy1,pose_r1, pose_xy2,pose_r2, pose_xy,pose_r])
    else:
        model = Model(input_layer, [pose_xy,pose_r])
    return model
def GoogLentetV1(input_layer,model_name,Tune,Custom_Size=False,AuxiliaryLoss=False):
        conv1 = Conv2D(64,7,2,padding='same',activation='relu',name=model_name+'_'+'conv1')(input_layer)        
        pool1 = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same',name=model_name+'_'+'pool1')(conv1)
        norm1 = BatchNormalization(axis=3, name=model_name+'_'+'norm1')(pool1)        
        reduction2 = Conv2D(64,1,padding='same',activation='relu',name=model_name+'_'+'reduction2')(norm1)        
        conv2 = Conv2D(192,3,padding='same',activation='relu',name=model_name+'_'+'conv2')(reduction2)
        norm2 = BatchNormalization(axis=3, name=model_name+'_'+'norm2')(conv2)
        pool2 = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='valid',name=model_name+'_'+'pool2')(norm2)
        icp1_reduction1 = Conv2D(96,1,1,padding='same',activation='relu',name=model_name+'_'+'icp1_reduction1')(pool2)
        icp1_out1 = Conv2D(128,3,padding='same',activation='relu',name=model_name+'_'+'icp1_out1')(icp1_reduction1)
        icp1_reduction2 = Conv2D(16,1,padding='same',activation='relu',name=model_name+'_'+'icp1_reduction2')(pool2)
        icp1_out2 = Conv2D(32,5,padding='same',activation='relu',name=model_name+'_'+'icp1_out2')(icp1_reduction2)   
        icp1_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name=model_name+'_'+'icp1_pool')(pool2)
        icp1_out3 = Conv2D(32,1,padding='same',activation='relu',name=model_name+'_'+'icp1_out3')(icp1_pool)       
        icp1_out0 = Conv2D(64,1,padding='same',activation='relu',name=model_name+'_'+'icp1_out0')(pool2)

        
        icp2_in = concatenate([icp1_out0, icp1_out1, icp1_out2, icp1_out3], axis=3, name=model_name+'_'+'icp2_in') 
        icp2_reduction1 = Conv2D(128,1,padding='same',activation='relu',name=model_name+'_'+'icp2_reduction1')(icp2_in)
        icp2_out1 = Conv2D(192,3,padding='same',activation='relu',name=model_name+'_'+'icp2_out1')(icp2_reduction1)     
        icp2_reduction2 = Conv2D(32,1,padding='same',activation='relu',name=model_name+'_'+'icp2_reduction2')(icp2_in)
        icp2_out2 = Conv2D(96,5,padding='same',activation='relu',name=model_name+'_'+'icp2_out2')(icp2_reduction2)
        icp2_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name=model_name+'_'+'icp2_pool')(icp2_in)
        icp2_out3 = Conv2D(64,1,padding='same',activation='relu',name=model_name+'_'+'icp2_out3')(icp2_pool)
        icp2_out0 = Conv2D(128,1,padding='same',activation='relu',name=model_name+'_'+'icp2_out0')(icp2_in)        
        icp2_out = concatenate([icp2_out0, icp2_out1, icp2_out2, icp2_out3],axis=3,name=model_name+'_'+'icp2_out')






        icp3_in = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same',name=model_name+'_'+'icp3_in')(icp2_out)
        icp3_reduction1 = Conv2D(96,1,padding='same',activation='relu',name=model_name+'_'+'icp3_reduction1')(icp3_in)
        icp3_out1 = Conv2D(208,3,padding='same',activation='relu',name=model_name+'_'+'icp3_out1')(icp3_reduction1)
        icp3_reduction2 = Conv2D(16,1,padding='same',activation='relu',name=model_name+'_'+'icp3_reduction2')(icp3_in)
        icp3_out2 = Conv2D(48,5,padding='same',activation='relu',name=model_name+'_'+'icp3_out2')(icp3_reduction2)      
        icp3_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name=model_name+'_'+'icp3_pool')(icp3_in)
        icp3_out3 = Conv2D(64,1,padding='same',activation='relu',name=model_name+'_'+'icp3_out3')(icp3_pool)
        icp3_out0 = Conv2D(192,1,padding='same',activation='relu',name=model_name+'_'+'icp3_out0')(icp3_in)       
        icp3_out = concatenate([icp3_out0, icp3_out1, icp3_out2, icp3_out3],axis=3,name=model_name+'_'+'icp3_out')
        
               
        icp4_reduction1 = Conv2D(112,1,padding='same',activation='relu',name=model_name+'_'+'icp4_reduction1')(icp3_out)
        icp4_out1 = Conv2D(224,3,padding='same',activation='relu',name=model_name+'_'+'icp4_out1')(icp4_reduction1)        
        icp4_reduction2 = Conv2D(24,1,padding='same',activation='relu',name=model_name+'_'+'icp4_reduction2')(icp3_out)
        icp4_out2 = Conv2D(64,5,padding='same',activation='relu',name=model_name+'_'+'icp4_out2')(icp4_reduction2)
        icp4_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name=model_name+'_'+'icp4_pool')(icp3_out)
        icp4_out3 = Conv2D(64,1,padding='same',activation='relu',name=model_name+'_'+'icp4_out3')(icp4_pool)
        icp4_out0 = Conv2D(160,1,padding='same',activation='relu',name=model_name+'_'+'icp4_out0')(icp3_out)
        icp4_out = concatenate([icp4_out0, icp4_out1, icp4_out2, icp4_out3],axis=3,name=model_name+'_'+'icp4_out')


        icp5_reduction1 = Conv2D(128,1,padding='same',activation='relu',name=model_name+'_'+'icp5_reduction1')(icp4_out)
        icp5_out1 = Conv2D(256,3,padding='same',activation='relu',name=model_name+'_'+'icp5_out1')(icp5_reduction1)
        icp5_reduction2 = Conv2D(24,1,padding='same',activation='relu',name=model_name+'_'+'icp5_reduction2')(icp4_out)
        icp5_out2 = Conv2D(64,5,padding='same',activation='relu',name=model_name+'_'+'icp5_out2')(icp5_reduction2)
        icp5_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name=model_name+'_'+'icp5_pool')(icp4_out)
        icp5_out3 = Conv2D(64,1,padding='same',activation='relu',name=model_name+'_'+'icp5_out3')(icp5_pool)
        icp5_out0 = Conv2D(128,1,padding='same',activation='relu',name=model_name+'_'+'icp5_out0')(icp4_out)
        icp5_out = concatenate([icp5_out0, icp5_out1, icp5_out2, icp5_out3],axis=3,name=model_name+'_'+'icp5_out')


        icp6_reduction1 = Conv2D(144,1,padding='same',activation='relu',name=model_name+'_'+'icp6_reduction1')(icp5_out)
        icp6_out1 = Conv2D(288,3,padding='same',activation='relu',name=model_name+'_'+'icp6_out1')(icp6_reduction1)
        icp6_reduction2 = Conv2D(32,1,padding='same',activation='relu',name=model_name+'_'+'icp6_reduction2')(icp5_out)
        icp6_out2 = Conv2D(64,5,padding='same',activation='relu',name=model_name+'_'+'icp6_out2')(icp6_reduction2)     
        icp6_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name=model_name+'_'+'icp6_pool')(icp5_out)
        icp6_out3 = Conv2D(64,1,padding='same',activation='relu',name=model_name+'_'+'icp6_out3')(icp6_pool)
        icp6_out0 = Conv2D(112,1,padding='same',activation='relu',name=model_name+'_'+'icp6_out0')(icp5_out)
        icp6_out = concatenate([icp6_out0, icp6_out1, icp6_out2, icp6_out3],axis=3,name=model_name+'_'+'icp6_out')
       


        icp7_reduction1 = Conv2D(160,1,padding='same',activation='relu',name=model_name+'_'+'icp7_reduction1')(icp6_out)
        icp7_out1 = Conv2D(320,3,padding='same',activation='relu',name=model_name+'_'+'icp7_out1')(icp7_reduction1)
        icp7_reduction2 = Conv2D(32,1,padding='same',activation='relu',name=model_name+'_'+'icp7_reduction2')(icp6_out)
        icp7_out2 = Conv2D(128,5,padding='same',activation='relu',name=model_name+'_'+'icp7_out2')(icp7_reduction2)
        icp7_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name=model_name+'_'+'icp7_pool')(icp6_out)
        icp7_out3 = Conv2D(128,1,padding='same',activation='relu',name=model_name+'_'+'icp7_out3')(icp7_pool)     
        icp7_out0 = Conv2D(256,1,padding='same',activation='relu',name=model_name+'_'+'icp7_out0')(icp6_out)
        icp7_out = concatenate([icp7_out0, icp7_out1, icp7_out2, icp7_out3],axis=3,name=model_name+'_'+'icp7_out')
  

       
        icp8_in = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same',name=model_name+'_'+'icp8_in')(icp7_out)
        icp8_reduction1 = Conv2D(160,1,padding='same',activation='relu',name=model_name+'_'+'icp8_reduction1')(icp8_in)
        icp8_out1 = Conv2D(320,3,padding='same',activation='relu',name=model_name+'_'+'icp8_out1')(icp8_reduction1)
        icp8_reduction2 = Conv2D(32,1,padding='same',activation='relu',name=model_name+'_'+'icp8_reduction2')(icp8_in)
        icp8_out2 = Conv2D(128,5,padding='same',activation='relu',name=model_name+'_'+'icp8_out2')(icp8_reduction2)
        icp8_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name=model_name+'_'+'icp8_pool')(icp8_in)
        icp8_out3 = Conv2D(128,1,padding='same',activation='relu',name=model_name+'_'+'icp8_out3')(icp8_pool)   
        icp8_out0 = Conv2D(256,1,padding='same',activation='relu',name=model_name+'_'+'icp8_out0')(icp8_in)
        icp8_out = concatenate([icp8_out0, icp8_out1, icp8_out2, icp8_out3],axis=3,name=model_name+'_'+'icp8_out')
        



        icp9_reduction1 = Conv2D(192,1,padding='same',activation='relu',name=model_name+'_'+'icp9_reduction1')(icp8_out)
        icp9_out1 = Conv2D(384,3,padding='same',activation='relu',name=model_name+'_'+'icp9_out1')(icp9_reduction1)
        icp9_reduction2 = Conv2D(48,1,padding='same',activation='relu',name=model_name+'_'+'icp9_reduction2')(icp8_out)
        icp9_out2 = Conv2D(128,5,padding='same',activation='relu',name=model_name+'_'+'icp9_out2')(icp9_reduction2)
        icp9_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name=model_name+'_'+'icp9_pool')(icp8_out)
        icp9_out3 = Conv2D(128,1,padding='same',activation='relu',name=model_name+'_'+'icp9_out3')(icp9_pool)
        icp9_out0 = Conv2D(384,1,padding='same',activation='relu',name=model_name+'_'+'icp9_out0')(icp8_out)
        icp9_out = concatenate([icp9_out0, icp9_out1, icp9_out2, icp9_out3],axis=3,name=model_name+'_'+'icp9_out')
        model = Model([input_layer], [icp6_out, icp3_out, icp9_out])
        return model

def VGG19v1(input_layer,model_name,Tune,Custom_Size=False):
    if(Tune or Custom_Size):
        base_model = keras.applications.vgg19.VGG19(weights=None, input_tensor=input_layer)
    else:        
        base_model = keras.applications.vgg19.VGG19(input_tensor=input_layer)

    out=base_model.get_layer('block5_pool').output  
    
    [pose_xy,pose_r] = regressor(model_name,out)
    

    model = Model([input_layer], [pose_xy,pose_r])
    return model


def Xception(input_layer,model_name,Tune,Custom_Size=False,AuxiliaryLoss=False):
    if(Tune or Custom_Size):
        base_model = keras.applications.Xception(weights=None, input_tensor=input_layer)
    else:        
        base_model = keras.applications.Xception(input_tensor=input_layer)
    #print(base_model.summary())

    out=base_model.get_layer("block14_sepconv2_act").output 
    #out1=base_model.get_layer("conv3_block4_out").output
    #out2=base_model.get_layer("conv4_block6_out").output

    #First Output
    [pose_xy,pose_r] = regressor(model_name,out)
    
    ##First Auxiliary
    #[pose_xy1,pose_r1] = auxiliary(model_name,out1,'_a1')

    ##Second Auxiliary
    #[pose_xy2,pose_r2] = auxiliary(model_name,out1,'_a2')
    

    if(AuxiliaryLoss==True):
        model = Model(input_layer, [pose_xy1,pose_r1, pose_xy2,pose_r2, pose_xy,pose_r])
    else:
        model = Model(input_layer, [pose_xy,pose_r])
    return model
def ResNet_50(input_layer,model_name,Tune,Custom_Size=False,AuxiliaryLoss=False):
    if(Tune or Custom_Size):
        base_model = keras.applications.resnet50.ResNet50(weights=None, input_tensor=input_layer)
    else:        
        base_model = keras.applications.resnet50.ResNet50(input_tensor=input_layer)
    #print(base_model.summary())

    out=base_model.get_layer("conv5_block3_out").output 
    out1=base_model.get_layer("conv3_block4_out").output
    out2=base_model.get_layer("conv4_block6_out").output

    #First Output
    [pose_xy,pose_r] = regressor(model_name,out)
    
    #First Auxiliary
    [pose_xy1,pose_r1] = auxiliary(model_name,out1,'_a1')

    #Second Auxiliary
    [pose_xy2,pose_r2] = auxiliary(model_name,out1,'_a2')
    

    if(AuxiliaryLoss==True):
        model = Model(input_layer, [pose_xy1,pose_r1, pose_xy2,pose_r2, pose_xy,pose_r])
    else:
        model = Model(input_layer, [pose_xy,pose_r])
    return model
def ResNet152V2(input_layer,model_name,Tune,Custom_Size=False,AuxiliaryLoss=False,BlockFirsts=False):
    if(Tune or Custom_Size):
        base_model = keras.applications.ResNet152V2(weights=None, input_tensor=input_layer)
    else:        
        base_model = keras.applications.ResNet152V2(input_tensor=input_layer)
    #print(base_model.summary())

    if(BlockFirsts==True):
       for layer in base_model.layers[:-4]:
          layer.trainable = False


    out=base_model.get_layer("conv5_block3_out").output
    #First Output
    [pose_xy,pose_r] = regressor(model_name,out)



    

    if(AuxiliaryLoss==True):
        out1=base_model.get_layer("conv3_block4_out").output
        out2=base_model.get_layer("conv4_block6_out").output
        #First Auxiliary
        [pose_xy1,pose_r1] = auxiliary(model_name,out1,'_a1')
        #Second Auxiliary
        [pose_xy2,pose_r2] = auxiliary(model_name,out1,'_a2')
        model = Model(input_layer, [pose_xy1,pose_r1, pose_xy2,pose_r2, pose_xy,pose_r])
    else:
        model = Model(input_layer, [pose_xy,pose_r])
    return model
def ResNet_50Transfer(input_layer,model_name,Tune,Custom_Size=False,AuxiliaryLoss=False):
    if(Tune or Custom_Size):
        base_model = keras.applications.resnet50.ResNet50(weights=None, input_tensor=input_layer)
    else:        
        base_model = keras.applications.resnet50.ResNet50(input_tensor=input_layer)
    #print(base_model.summary())
    for layer in base_model.layers[:-4]:
       layer.trainable = False

    out=base_model.get_layer("conv5_block3_out").output 


    out1=base_model.get_layer("conv3_block4_out").output
    out2=base_model.get_layer("conv4_block6_out").output

    #First Output
    [pose_xy,pose_r] = regressor(model_name,out)
    
    #First Auxiliary
    [pose_xy1,pose_r1] = auxiliary(model_name,out1,'_a1')

    #Second Auxiliary
    [pose_xy2,pose_r2] = auxiliary(model_name,out1,'_a2')
    

    if(AuxiliaryLoss==True):
        model = Model(input_layer, [pose_xy1,pose_r1, pose_xy2,pose_r2, pose_xy,pose_r])
    else:
        model = Model(input_layer, [pose_xy,pose_r])
    return model

def GoogLenet(input_layer,model_name,Tune,Custom_Size=False,AuxiliaryLoss=False):
    
    base_model = GoogLentetV1(input_layer,model_name,Tune,Custom_Size=False,AuxiliaryLoss=False)
              
    out=base_model.get_layer(model_name+'_'+'icp9_out').output 
    out1=base_model.get_layer(model_name+'_'+'icp3_out').output
    out2=base_model.get_layer(model_name+'_'+'icp6_out').output

    #First Output
    [pose_xy,pose_r] = regressor(model_name,out)
    
    #First Auxiliary
    [pose_xy1,pose_r1] = auxiliary(model_name,out1,'_a1')

    #Second Auxiliary
    [pose_xy2,pose_r2] = auxiliary(model_name,out2,'_a2')


    if(AuxiliaryLoss==True):
        model = Model([input_layer], [pose_xy1,pose_r1, pose_xy2,pose_r2, pose_xy,pose_r])
    else:
        model = Model([input_layer], [pose_xy,pose_r])
    return model

def InceptionV3(input_layer,model_name,Tune,Custom_Size=False,AuxiliaryLoss=False):
    if(Tune or Custom_Size):
        base_model = keras.applications.InceptionV3(weights=None, input_tensor=input_layer)
    else:        
        base_model = keras.applications.InceptionV3(input_tensor=input_layer)
    
    #print(base_model.summary())

    
    out=base_model.get_layer(index=-3).output
    

    out1=base_model.get_layer("mixed9").output
    out2=base_model.get_layer("average_pooling2d_8").output

    #First Output
    [pose_xy,pose_r] = regressor(model_name,out)
    
    #First Auxiliary
    [pose_xy1,pose_r1] = auxiliary(model_name,out1,'_a1')

    #Second Auxiliary
    [pose_xy2,pose_r2] = auxiliary(model_name,out2,'_a2')
    




    if(AuxiliaryLoss==True):
        model = Model([input_layer], [pose_xy1,pose_r1, pose_xy2,pose_r2, pose_xy,pose_r])
    else:
        model = Model([input_layer], [pose_xy,pose_r])

    return model
def DenseNet_201(input_layer,model_name,Tune,Custom_Size=False,AuxiliaryLoss=False):
    if(Tune or Custom_Size):
        base_model = keras.applications.DenseNet201(weights=None, input_tensor=input_layer)
    else:        
        base_model = keras.applications.DenseNet201(input_tensor=input_layer)
    
    #print(base_model.summary())

    
    out=base_model.get_layer(index=-3).output
    

    out1=base_model.get_layer("conv4_block48_concat").output
    out2=base_model.get_layer("conv5_block29_concat").output

    #First Output
    [pose_xy,pose_r] = regressor(model_name,out)
    
    #First Auxiliary
    [pose_xy1,pose_r1] = auxiliary(model_name,out1,'_a1')

    #Second Auxiliary
    [pose_xy2,pose_r2] = auxiliary(model_name,out2,'_a2')
    




    if(AuxiliaryLoss==True):
        model = Model([input_layer], [pose_xy1,pose_r1, pose_xy2,pose_r2, pose_xy,pose_r])
    else:
        model = Model([input_layer], [pose_xy,pose_r])

    return model
def EfficientNetB4(input_layer1,model_name,Tune,Custom_Size=False,AuxiliaryLoss=False,BlockFirsts=False):
 with tf.device('/gpu:0'):
    if(Tune or Custom_Size):
        base_model = keras.applications.efficientnet.EfficientNetB4(weights=None, input_tensor=input_layer1)
    else:        
        base_model = keras.applications.efficientnet.EfficientNetB4(input_tensor=input_layer1)

    #print(base_model1.summary())
    
    
    for i in range(0,len(base_model.layers)):
        base_model.get_layer(index=i)._name ="Mount_"+base_model.get_layer(index=i).name

    if(BlockFirsts==True):
       for layer in base_model.layers[:-4]:
         layer.trainable = False

    out=base_model.get_layer(index=-4).output  


    #First Output
    [pose_xy,pose_r] = regressor(model_name,out)
    


    if(AuxiliaryLoss==True):
        model = Model([input_layer1,input_layer2], [pose_xy1,pose_r1, pose_xy2,pose_r2, pose_xy,pose_r])
    else:
        model = Model([input_layer1], [pose_xy,pose_r])
    return model
def EfficientNetB5(input_layer1,model_name,Tune,Custom_Size=False,AuxiliaryLoss=False,BlockFirsts=False):
 with tf.device('/gpu:0'):
    if(Tune or Custom_Size):
        base_model = keras.applications.efficientnet.EfficientNetB5(weights=None, input_tensor=input_layer1)
    else:        
        base_model = keras.applications.efficientnet.EfficientNetB5(input_tensor=input_layer1)

    #print(base_model1.summary())
    
    
    for i in range(0,len(base_model.layers)):
        base_model.get_layer(index=i)._name ="Mount_"+base_model.get_layer(index=i).name

    if(BlockFirsts==True):
       for layer in base_model.layers[:-4]:
         layer.trainable = False

    out=base_model.get_layer(index=-4).output  


    #First Output
    [pose_xy,pose_r] = regressor(model_name,out)
    


    if(AuxiliaryLoss==True):
        model = Model([input_layer1,input_layer2], [pose_xy1,pose_r1, pose_xy2,pose_r2, pose_xy,pose_r])
    else:
        model = Model([input_layer1], [pose_xy,pose_r])
    return model
def EfficientNetB7(input_layer1,model_name,Tune,Custom_Size=False,AuxiliaryLoss=False,BlockFirsts=False):
 with tf.device('/gpu:0'):
    if(Tune or Custom_Size):
        base_model = keras.applications.efficientnet.EfficientNetB7(weights=None, input_tensor=input_layer1)
    else:        
        base_model = keras.applications.efficientnet.EfficientNetB7(input_tensor=input_layer1)

    #print(base_model1.summary())
    
    
    for i in range(0,len(base_model.layers)):
        base_model.get_layer(index=i)._name ="Mount_"+base_model.get_layer(index=i).name

    if(BlockFirsts==True):
       for layer in base_model.layers[:-4]:
         layer.trainable = False

    out=base_model.get_layer(index=-4).output  


    #First Output
    [pose_xy,pose_r] = regressor(model_name,out)
    


    if(AuxiliaryLoss==True):
        model = Model([input_layer1,input_layer2], [pose_xy1,pose_r1, pose_xy2,pose_r2, pose_xy,pose_r])
    else:
        model = Model([input_layer1], [pose_xy,pose_r])
    return model



def ResNet152V2Stereo(input_layer1,input_layer2,model_name,Tune,Custom_Size=False,AuxiliaryLoss=False,BlockFirsts=False):
    if(Tune or Custom_Size):
        base_model1 = keras.applications.ResNet152V2(weights=None, input_tensor=input_layer1)
        base_model2 = keras.applications.ResNet152V2(weights=None, input_tensor=input_layer2)
    else:        
        base_model1 = keras.applications.ResNet152V2(input_tensor=input_layer1)
        base_model2 = keras.applications.ResNet152V2(input_tensor=input_layer2)

    if(BlockFirsts==True):
       for layer in base_model1.layers[:-4]:
          layer.trainable = False
       for layer in base_model2.layers[:-4]:
          layer.trainable = False

    for i in range(0,len(base_model1.layers)):
        base_model1.get_layer(index=i)._name ="Above_"+base_model1.get_layer(index=i).name
        base_model2.get_layer(index=i)._name ="Mount_"+base_model2.get_layer(index=i).name



    #print(base_model.summary())
    out_1=base_model1.get_layer("Above_"+"conv5_block3_out").output  
    #out2_1=base_model1.get_layer("Above_"+"conv4_block6_out").output
    #out1_1=base_model1.get_layer("Above_"+"conv3_block4_out").output

    
    out_2=base_model2.get_layer("Mount_"+"conv5_block3_out").output  
    #out2_2=base_model2.get_layer("Mount_"+"conv4_block6_out").output
    #out1_2=base_model2.get_layer("Mount_"+"conv3_block4_out").output
    
    out = concatenate([out_1,out_2])

    #First Output
    [pose_xy,pose_r] = regressor(model_name,out)
    
    #First Auxiliary
    [pose_xy1,pose_r1] = auxiliary(model_name,out_1,'_a1')

    #Second Auxiliary
    [pose_xy2,pose_r2] = auxiliary(model_name,out_2,'_a2')
    
    ##third Auxiliary
    #[pose_xy3,pose_r3] = auxiliary(model_name,out1_2,'_a3')

    ##fourth Auxiliary
    #[pose_xy4,pose_r4] = auxiliary(model_name,out2_2,'_a4')




    if(AuxiliaryLoss==True):
        model = Model([input_layer1,input_layer2], [pose_xy1,pose_r1, pose_xy2,pose_r2, pose_xy,pose_r])
    else:
        model = Model([input_layer1,input_layer2], [pose_xy,pose_r])
    return model
def ResNet50Stereo(input_layer1,input_layer2,model_name,Tune,Custom_Size=False,AuxiliaryLoss=False,BlockFirsts=False):
    if(Tune or Custom_Size):
        base_model1 = keras.applications.resnet50.ResNet50(weights=None, input_tensor=input_layer1)
        base_model2 = keras.applications.resnet50.ResNet50(weights=None, input_tensor=input_layer2)
    else:        
        base_model1 = keras.applications.resnet50.ResNet50(input_tensor=input_layer1)
        base_model2 = keras.applications.resnet50.ResNet50(input_tensor=input_layer2)
    for i in range(0,len(base_model1.layers)):
        base_model1.get_layer(index=i)._name ="Above_"+base_model1.get_layer(index=i).name
        base_model2.get_layer(index=i)._name ="Mount_"+base_model2.get_layer(index=i).name

    if(BlockFirsts==True):
       for layer in base_model1.layers[:-4]:
         layer.trainable = False
       for layer in base_model2.layers[:-4]:
         layer.trainable = False

    #print(base_model.summary())
    out_1=base_model1.get_layer("Above_"+"conv5_block3_out").output  
    out2_1=base_model1.get_layer("Above_"+"conv4_block6_out").output
    out1_1=base_model1.get_layer("Above_"+"conv3_block4_out").output

    
    out_2=base_model2.get_layer("Mount_"+"conv5_block3_out").output  
    out2_2=base_model2.get_layer("Mount_"+"conv4_block6_out").output
    out1_2=base_model2.get_layer("Mount_"+"conv3_block4_out").output
    
    out = concatenate([out_1,out_2])

    #First Output
    [pose_xy,pose_r] = regressor(model_name,out)
    
    #First Auxiliary
    [pose_xy1,pose_r1] = auxiliary(model_name,out1_1,'_a1')

    #Second Auxiliary
    [pose_xy2,pose_r2] = auxiliary(model_name,out2_1,'_a2')
    
    #third Auxiliary
    [pose_xy3,pose_r3] = auxiliary(model_name,out1_2,'_a3')

    #fourth Auxiliary
    [pose_xy4,pose_r4] = auxiliary(model_name,out2_2,'_a4')




    if(AuxiliaryLoss==True):
        model = Model([input_layer1,input_layer2], [pose_xy1,pose_r1, pose_xy2,pose_r2, pose_xy,pose_r])
    else:
        model = Model([input_layer1,input_layer2], [pose_xy,pose_r])
    return model
def InceptResnetStereo(input_layer1,input_layer2,model_name,Tune,Custom_Size=False,AuxiliaryLoss=False,BlockFirsts=False):
    if(Tune or Custom_Size):
        base_model1 = keras.applications.resnet50.ResNet50(weights=None, input_tensor=input_layer1)
        base_model2 = InceptionResNetV2(weights=None, input_tensor=input_layer2)
    else:        
        base_model1 = keras.applications.resnet50.ResNet50(input_tensor=input_layer1)
        base_model2 = InceptionResNetV2(input_tensor=input_layer2)

    for i in range(0,len(base_model1.layers)):
        base_model1.get_layer(index=i)._name = "Above_" + base_model1.get_layer(index=i).name
    for i in range(0,len(base_model2.layers)):
        base_model2.get_layer(index=i)._name = "Mount_" + base_model2.get_layer(index=i).name


    if(BlockFirsts==True):
       for layer in base_model1.layers[:-4]:
         layer.trainable = False
       for layer in base_model2.layers[:-4]:
         layer.trainable = False
         
    #print(base_model.summary())
    out_1=base_model1.get_layer("Above_"+"conv5_block3_out").output  
    out2_1=base_model1.get_layer("Above_"+"conv4_block6_out").output
    out1_1=base_model1.get_layer("Above_"+"conv3_block4_out").output

    
    out_2=base_model2.get_layer(index=-3).output 
    out2_2=base_model2.get_layer("Mount_"+"block8_7_ac").output
    out1_2=base_model2.get_layer("Mount_"+"block8_1_mixed").output
    
    out_1=Conv2D(1024,3)(out_1)
    out_2=Conv2D(1024,1)(out_2)

    out = concatenate([out_1,out_2])

    #First Output
    [pose_xy,pose_r] = regressor(model_name,out)
    
    #First Auxiliary
    [pose_xy1,pose_r1] = auxiliary(model_name,out1_1,'_a1')

    #Second Auxiliary
    [pose_xy2,pose_r2] = auxiliary(model_name,out2_1,'_a2')
    
    #third Auxiliary
    [pose_xy3,pose_r3] = auxiliary(model_name,out1_2,'_a3')

    #fourth Auxiliary
    [pose_xy4,pose_r4] = auxiliary(model_name,out2_2,'_a4')




    if(AuxiliaryLoss==True):
        model = Model([input_layer1,input_layer2], [pose_xy1,pose_r1, pose_xy2,pose_r2, pose_xy,pose_r])
    else:
        model = Model([input_layer1,input_layer2], [pose_xy,pose_r])
    return model
def InceptDenseNetStereo(input_layer1,input_layer2,model_name,Tune,Custom_Size=False,AuxiliaryLoss=False):
    if(Tune or Custom_Size):
        base_model1 = InceptionResNetV2(weights=None, input_tensor=input_layer1)
        base_model2 = keras.applications.DenseNet201(weights=None, input_tensor=input_layer2)
    else:        
        base_model1 = InceptionResNetV2(input_tensor=input_layer1)
        base_model2 = keras.applications.DenseNet201(input_tensor=input_layer2)

    for i in range(0,len(base_model1.layers)):
        base_model1.get_layer(index=i)._name = "Above_" + base_model1.get_layer(index=i).name
    for i in range(0,len(base_model2.layers)):
        base_model2.get_layer(index=i)._name = "Mount_" + base_model2.get_layer(index=i).name



    #print(base_model.summary())
    out_1=base_model1.get_layer(index=-3).output  
    out2_1=base_model1.get_layer("Above_"+"block8_7_ac").output
    out1_1=base_model1.get_layer("Above_"+"block8_1_mixed").output


    out_2=base_model2.get_layer(index=-3).output 
    out2_2=base_model2.get_layer("Mount_"+"conv5_block29_concat").output
    out1_2=base_model2.get_layer("Mount_"+"conv4_block48_concat").output
    
    out_1=Conv2D(1024,1)(out_1)
    out_2=Conv2D(1024,3)(out_2)

    out = concatenate([out_1,out_2])

    #First Output
    [pose_xy,pose_r] = regressor(model_name,out)
    
    #First Auxiliary
    [pose_xy1,pose_r1] = auxiliary(model_name,out1_1,'_a1')

    #Second Auxiliary
    [pose_xy2,pose_r2] = auxiliary(model_name,out2_1,'_a2')
    
    #third Auxiliary
    [pose_xy3,pose_r3] = auxiliary(model_name,out1_2,'_a3')

    #fourth Auxiliary
    [pose_xy4,pose_r4] = auxiliary(model_name,out2_2,'_a4')




    if(AuxiliaryLoss==True):
        model = Model([input_layer1,input_layer2], [pose_xy1,pose_r1, pose_xy2,pose_r2,pose_xy3,pose_r3, pose_xy4,pose_r4, pose_xy,pose_r])
    else:
        model = Model([input_layer1,input_layer2], [pose_xy,pose_r])
    return model
def InceptionResNetV2Stereo(input_layer1,input_layer2,model_name,Tune,Custom_Size=False,AuxiliaryLoss=False,BlockFirsts=False):
    if(Tune or Custom_Size):
        base_model1 = InceptionResNetV2(weights=None, input_tensor=input_layer1)
        base_model2 = InceptionResNetV2(weights=None, input_tensor=input_layer2)
    else:        
        base_model1 = InceptionResNetV2(input_tensor=input_layer1)
        base_model2 = InceptionResNetV2(input_tensor=input_layer2)

    for i in range(0,len(base_model1.layers)):
        base_model1.get_layer(index=i)._name = "Above_" + base_model1.get_layer(index=i).name
    for i in range(0,len(base_model2.layers)):
        base_model2.get_layer(index=i)._name = "Mount_" + base_model2.get_layer(index=i).name


    if(BlockFirsts==True):
       for layer in base_model1.layers[:-4]:
         layer.trainable = False
       for layer in base_model2.layers[:-4]:
         layer.trainable = False
         
    #print(base_model.summary())
    out_1=base_model1.get_layer(index=-3).output  
    #out2_1=base_model1.get_layer("Above_"+"conv4_block6_out").output
    #out1_1=base_model1.get_layer("Above_"+"conv3_block4_out").output

    
    out_2=base_model2.get_layer(index=-3).output 
    #out2_2=base_model2.get_layer("Mount_"+"block8_7_ac").output
    #out1_2=base_model2.get_layer("Mount_"+"block8_1_mixed").output
    
    out_1=Conv2D(1024,3)(out_1)
    out_2=Conv2D(1024,3)(out_2)

    out = concatenate([out_1,out_2])

    #First Output
    [pose_xy,pose_r] = regressor(model_name,out)
    
    #First Auxiliary
    [pose_xy1,pose_r1] = auxiliary(model_name,out_1,'_a1')

    #Second Auxiliary
    [pose_xy2,pose_r2] = auxiliary(model_name,out_2,'_a2')
    
    ##third Auxiliary
    #[pose_xy3,pose_r3] = auxiliary(model_name,out1_2,'_a3')

    ##fourth Auxiliary
    #[pose_xy4,pose_r4] = auxiliary(model_name,out2_2,'_a4')




    if(AuxiliaryLoss==True):
        model = Model([input_layer1,input_layer2], [pose_xy1,pose_r1, pose_xy2,pose_r2, pose_xy,pose_r])
    else:
        model = Model([input_layer1,input_layer2], [pose_xy,pose_r])
    return model
def XceptionStereo(input_layer1,input_layer2,model_name,Tune,Custom_Size=False,AuxiliaryLoss=False,BlockFirsts=False):
    if(Tune or Custom_Size):
        keras.applications.DenseNet121
        base_model1 = keras.applications.Xception(weights=None, input_tensor=input_layer1)
        base_model2 = keras.applications.Xception(weights=None, input_tensor=input_layer2)
    else:        
        base_model1 = keras.applications.Xception(input_tensor=input_layer1)
        base_model2 = keras.applications.Xception(input_tensor=input_layer2)

    print(base_model1.summary())
    
    
    for i in range(0,len(base_model1.layers)):
        base_model1.get_layer(index=i)._name ="Above_"+base_model1.get_layer(index=i).name
        base_model2.get_layer(index=i)._name ="Mount_"+base_model2.get_layer(index=i).name

    if(BlockFirsts==True):
       for layer in base_model1.layers[:-4]:
         layer.trainable = False
       for layer in base_model2.layers[:-4]:
         layer.trainable = False

    out_1=base_model1.get_layer("Above_"+"block14_sepconv2_act").output  
    #out2_1=base_model1.get_layer("Above_"+"conv4_block6_out").output
    #out1_1=base_model1.get_layer("Above_"+"conv3_block4_out").output

    
    out_2=base_model2.get_layer("Mount_"+"block14_sepconv2_act").output  
    #out2_2=base_model2.get_layer("Mount_"+"conv4_block6_out").output
    #out1_2=base_model2.get_layer("Mount_"+"conv3_block4_out").output
    
    out = concatenate([out_1,out_2])

    #First Output
    [pose_xy,pose_r] = regressor(model_name,out)
    
    #First Auxiliary
    [pose_xy1,pose_r1] = auxiliary(model_name,out_1,'_a1')

    #Second Auxiliary
    [pose_xy2,pose_r2] = auxiliary(model_name,out_2,'_a2')
    
    ##third Auxiliary
    #[pose_xy3,pose_r3] = auxiliary(model_name,out1_2,'_a3')

    ##fourth Auxiliary
    #[pose_xy4,pose_r4] = auxiliary(model_name,out2_2,'_a4')




    if(AuxiliaryLoss==True):
        model = Model([input_layer1,input_layer2], [pose_xy1,pose_r1, pose_xy2,pose_r2, pose_xy,pose_r])
    else:
        model = Model([input_layer1,input_layer2], [pose_xy,pose_r])
    return model
def EfficientNetB4Stereo(input_layer1,input_layer2,model_name,Tune,Custom_Size=False,AuxiliaryLoss=False,BlockFirsts=False):
 with tf.device('/gpu:0'):
    if(Tune or Custom_Size):
        base_model1 = keras.applications.efficientnet.EfficientNetB4(weights=None, input_tensor=input_layer1)
        base_model2 = keras.applications.efficientnet.EfficientNetB4(weights=None, input_tensor=input_layer2)
    else:    
        base_model1 = keras.applications.efficientnet.EfficientNetB4(input_tensor=input_layer1)
        base_model2 = keras.applications.efficientnet.EfficientNetB4(input_tensor=input_layer2)

    #print(base_model1.summary())
    
    
    for i in range(0,len(base_model1.layers)):
        base_model1.get_layer(index=i)._name ="Above_"+base_model1.get_layer(index=i).name
        base_model2.get_layer(index=i)._name ="Mount_"+base_model2.get_layer(index=i).name

    if(BlockFirsts==True):
       for layer in base_model1.layers[:-4]:
         layer.trainable = False
       for layer in base_model2.layers[:-4]:
         layer.trainable = False

    out_1=base_model1.get_layer(index=-4).output  
    #out2_1=base_model1.get_layer("Above_"+"conv4_block6_out").output
    #out1_1=base_model1.get_layer("Above_"+"conv3_block4_out").output

    
    out_2=base_model2.get_layer(index=-4).output  
    #out2_2=base_model2.get_layer("Mount_"+"conv4_block6_out").output
    #out1_2=base_model2.get_layer("Mount_"+"conv3_block4_out").output
    
    out = concatenate([out_1,out_2])

    #First Output
    [pose_xy,pose_r] = regressor(model_name,out)
    
    #First Auxiliary
    [pose_xy1,pose_r1] = auxiliary(model_name,out_1,'_a1')

    #Second Auxiliary
    [pose_xy2,pose_r2] = auxiliary(model_name,out_2,'_a2')
    
    ##third Auxiliary
    #[pose_xy3,pose_r3] = auxiliary(model_name,out1_2,'_a3')

    ##fourth Auxiliary
    #[pose_xy4,pose_r4] = auxiliary(model_name,out2_2,'_a4')




    if(AuxiliaryLoss==True):
        model = Model([input_layer1,input_layer2], [pose_xy1,pose_r1, pose_xy2,pose_r2, pose_xy,pose_r])
    else:
        model = Model([input_layer1,input_layer2], [pose_xy,pose_r])
    return model

def EfficientNetB472Stereo(input_layer2,input_layer1,model_name,Tune,Custom_Size=False,AuxiliaryLoss=False,BlockFirsts=False):
 with tf.device('/gpu:0'):
    #above
    base_model1 = EfficientNetB4(input_layer1,model_name,Tune=False,Custom_Size=False,AuxiliaryLoss=False,BlockFirsts=False)
    #mount
    base_model2 = EfficientNetB7(input_layer2,model_name,Tune=False,Custom_Size=False,AuxiliaryLoss=False,BlockFirsts=False)
    
    if(Tune==False):
        base_model1.load_weights('EfficientNetB4Mount_checkpoint_weights.h5')
        base_model2.load_weights('EfficientNetB7Mount_checkpoint_weights.h5')  

    for i in range(0,len(base_model1.layers)):
        base_model1.get_layer(index=i)._name ="C1"+base_model1.get_layer(index=i).name
    for i in range(0,len(base_model2.layers)):
        base_model2.get_layer(index=i)._name ="C2"+base_model2.get_layer(index=i).name


    #print(base_model1.summary())
    #print(base_model2.summary())
    

    #for i in range(0,len(base_model1.layers)):
    #    base_model1.get_layer(index=i).trainable =False
    #for i in range(0,len(base_model2.layers)):
    #    base_model2.get_layer(index=i).trainable =False


    out1_xy = base_model1.output[0]
    out1_r  = base_model1.output[1]

    out2_xy = base_model2.output[0]
    out2_r  = base_model2.output[1]
    
    #xy=concatenate([out1_xy,out2_xy])
    #r=concatenate([out1_r,out2_r])

    #rot_fc_xy = Dense(10,activation='relu',name=model_name+"_"+'rot_fc_xy')(r)     
    #pose_r = Dense(1,name=model_name+"_"+'pose_r')(rot_fc_xy)

    #trans_fc_xy = Dense(20,activation='relu',name=model_name+"_"+'trans_fc_xy')(xy)     
    #pose_xy = Dense(2,name=model_name+"_"+'pose_xy')(trans_fc_xy)



    model = Model([input_layer1,input_layer2], [out1_xy,out2_r])

    return model
def EfficientNetB47Stereo(input_layer1,input_layer2,model_name,Tune,Custom_Size=False,AuxiliaryLoss=False,BlockFirsts=False):
 with tf.device('/gpu:0'):
    #above
    base_model1 = EfficientNetB4(input_layer1,model_name,Tune=False,Custom_Size=False,AuxiliaryLoss=False,BlockFirsts=False)
    #mount
    base_model2 = EfficientNetB7(input_layer2,model_name,Tune=True,Custom_Size=False,AuxiliaryLoss=False,BlockFirsts=False)
    
    if(Tune==False):
        base_model1.load_weights('EfficientNetB4Mount_checkpoint_weights.h5')
        base_model2.load_weights('EfficientNetB7Mount_checkpoint_weights.h5')  


    for i in range(0,len(base_model1.layers)):
        base_model1.get_layer(index=i).trainable =False
    for i in range(0,len(base_model2.layers)-5):
        base_model2.get_layer(index=i).trainable =False

    for i in range(0,len(base_model1.layers)):
        base_model1.get_layer(index=i)._name ="C2"+base_model1.get_layer(index=i).name
    for i in range(0,len(base_model2.layers)):
        base_model2.get_layer(index=i)._name ="C1"+base_model2.get_layer(index=i).name



    

    
    #print(base_model1.summary())
    #print(base_model2.summary())

    out1_xy = base_model1.output[0]
    out1_r  = base_model1.output[1]

    out2_xy = base_model2.output[0]
    out2_r  = base_model2.output[1]

    model = Model([input_layer1,input_layer2], [out2_xy,out2_r])
    return model
    
    #xy=concatenate([out1_xy,out2_xy])
    #r=concatenate([out1_r,out2_r])

    x1=base_model1.get_layer('C1EfficientNetB47Stereo_trans_MP2').output
    r1=base_model1.get_layer('C1EfficientNetB47Stereo_rot_MP2').output
                
    x2=base_model2.get_layer('C2EfficientNetB47Stereo_trans_MP2').output
    r2=base_model2.get_layer('C2EfficientNetB47Stereo_rot_MP2').output

    #x1=base_model1.get_layer('C1EfficientNetB47Stereo_trans_fc_xy').output
    #r1=base_model1.get_layer('C1EfficientNetB47Stereo_rot_fc_xy').output
                
    #x2=base_model2.get_layer('C2EfficientNetB47Stereo_trans_fc_xy').output
    #r2=base_model2.get_layer('C2EfficientNetB47Stereo_rot_fc_xy').output

    xy=concatenate([x1,x2])
    r=concatenate([r1,r2])


    ##trans_fc_xy = Dense(1024*3,name=model_name+"_"+'trans_fc_xy')(xy)     
    ##pose_xy = Dense(2,name=model_name+"_"+'pose_xy')(trans_fc_xy)

    ##rot_fc_xy = Dense(2048,name=model_name+"_"+'rot_fc_xy')(r)     
    ##pose_r = Dense(1,name=model_name+"_"+'pose_r')(rot_fc_xy)
    
    xy=MaxPooling2D(pool_size=(5,5),strides=(1,1),padding='same',name=model_name+"_"+'trans_MP0')(xy)
    r=MaxPooling2D(pool_size=(5,5),strides=(1,1),padding='same',name=model_name+"_"+'rot_MP0')(r)

    trans_L1 = Conv2D(64,2,padding='same',activation='relu',name=model_name+"_"+'trans_L1')(xy)
    trans_L2 = Conv2D(64,1,padding='same',activation='relu',name=model_name+"_"+'trans_L2')(trans_L1)
    trans_MP1 = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name=model_name+"_"+'trans_MP1')(trans_L2)
    trans_L3 = Conv2D(64,1,padding='same',activation='relu',name=model_name+"_"+'trans_L3')(trans_MP1)
    trans_L4 = Conv2D(128,1,padding='same',activation='relu',name=model_name+"_"+'trans_L4')(trans_L3)
    trans_MP2 = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name=model_name+"_"+'trans_MP2')(trans_L4)    
    trans_flat = Flatten()(trans_MP2)
    trans_fc_xy = Dense(2048,activation='relu',name=model_name+"_"+'trans_fc_xy')(trans_flat)     
    pose_xy = Dense(2,name=model_name+"_"+'pose_xy')(trans_fc_xy)


    rot_L1 = Conv2D(128,1,padding='same',activation='relu',name=model_name+"_"+'rot_L1')(r)
    rot_L2 = Conv2D(128,1,padding='same',activation='relu',name=model_name+"_"+'rot_L2')(rot_L1)
    rot_MP1 = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name=model_name+"_"+'rot_MP1')(rot_L2)
    rot_L3 = Conv2D(128,1,padding='same',activation='relu',name=model_name+"_"+'rot_L3')(rot_MP1)
    rot_L4 = Conv2D(64,1,padding='same',activation='relu',name=model_name+"_"+'rot_L4')(rot_L3)
    rot_MP2 = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name=model_name+"_"+'rot_MP2')(rot_L4)    
    rot_flat = Flatten()(rot_MP2)
    rot_fc_xy = Dense(2048,activation='relu',name=model_name+"_"+'rot_fc_xy')(rot_flat)     
    pose_r = Dense(1,name=model_name+"_"+'pose_r')(rot_fc_xy)



    model = Model([input_layer1,input_layer2], [out1_xy,out1_r,out2_xy,out2_r,pose_xy,pose_r])
    return model



def TNet_1Camera(type,Tune=False,Auxiliary_Loss=False,BlockFirsts=False):
    if(type=="VGG19Mount"):
        input = Input(shape=(224, 224, 3))
        model = VGG19v1(input,type,Tune)
        print(model.summary())
        plot_model(model, to_file=type + '_net.png', show_shapes=True, show_layer_names=True,rankdir='LR')
        if(Tune):
            model.load_weights(type+'_checkpoint_weights.h5')
    elif(type=="InceptionV3Mount"):
        input = Input(shape=(224, 224, 3))
        model = InceptionV3(input,type,Tune,AuxiliaryLoss=Auxiliary_Loss)
        print(model.summary())
        plot_model(model, to_file=type + '_net.png', show_shapes=True, show_layer_names=True,rankdir='LR')
        if(Tune):
            model.load_weights(type+'_checkpoint_weights.h5')
    elif(type=="DenseNet201Mount"):
        input = Input(shape=(224, 224, 3))
        model = DenseNet_201(input,type,Tune,AuxiliaryLoss=Auxiliary_Loss)
        print(model.summary())
        #plot_model(model, to_file=type + '_net.png', show_shapes=True, show_layer_names=True,rankdir='LR')
        if(Tune):
            model.load_weights(type+'_checkpoint_weights.h5')
    elif(type=="InceptionResNetV2Mount"):
        input = Input(shape=(224, 224, 3))
        model = InceptionResNet_V2(input,type,Tune,AuxiliaryLoss=Auxiliary_Loss,BlockFirsts=BlockFirsts )
        print(model.summary())
        #plot_model(model, to_file=type + '_net.png', show_shapes=True, show_layer_names=True,rankdir='LR')
        if(Tune):
            model.load_weights(type+'_checkpoint_weights.h5')
    elif(type=="ResNet50Mount"):
        input = Input(shape=(224, 224, 3))
        model = ResNet_50(input,type,Tune,AuxiliaryLoss=Auxiliary_Loss)
        print(model.summary())
        #plot_model(model, to_file=type + '_net.png', show_shapes=True, show_layer_names=True,rankdir='LR')
        if(Tune):
            model.load_weights(type+'_checkpoint_weights.h5')
    elif(type=="ResNet50_HighMount"):
        input = Input(shape=(270, 480, 3))
        model = ResNet_50(input,type,Tune,True)
        print(model.summary())
        #plot_model(model, to_file=type + '_net.png', show_shapes=True, show_layer_names=True,rankdir='LR')
        if(Tune):
            model.load_weights(type+'_checkpoint_weights.h5')
    elif(type=="ResNet152V2Mount"):
        input = Input(shape=(224, 224, 3))
        model = ResNet152V2(input,type,Tune,AuxiliaryLoss=Auxiliary_Loss,BlockFirsts=BlockFirsts)
        print(model.summary())
        #plot_model(model, to_file=type + '_net.png', show_shapes=True, show_layer_names=True,rankdir='LR')
        if(Tune):
            model.load_weights(type+'_checkpoint_weights.h5')
    elif(type=="ResNet50_TransferMount"):
        input = Input(shape=(224, 224, 3))
        model = ResNet_50Transfer(input,type,Tune,AuxiliaryLoss=Auxiliary_Loss)
        print(model.summary())
        #plot_model(model, to_file=type + '_net.png', show_shapes=True, show_layer_names=True,rankdir='LR')
        if(Tune):
            model.load_weights(type+'_checkpoint_weights.h5')
    elif(type=="GoogLenetMount"):
        input = Input(shape=(224, 224, 3))
        model = GoogLenet(input,type,Tune,AuxiliaryLoss=Auxiliary_Loss)
        print(model.summary())
        #plot_model(model, to_file=type + '_net.png', show_shapes=True, show_layer_names=True,rankdir='LR')
        if(Tune):
            model.load_weights(type+'_checkpoint_weights.h5')
    elif(type=="NasNetLargeMount"):
        input = Input(shape=(224, 224, 3))
        model = NasNet_Large(input,type,Tune,AuxiliaryLoss=Auxiliary_Loss)
        print(model.summary())
        #plot_model(model, to_file=type + '_net.png', show_shapes=True, show_layer_names=True,rankdir='LR')
        if(Tune):
            model.load_weights(type+'_checkpoint_weights.h5')
    elif(type=="XceptionMount"):
        input = Input(shape=(224, 224, 3))
        model = Xception(input,type,Tune,AuxiliaryLoss=Auxiliary_Loss)
        print(model.summary())
        #plot_model(model, to_file=type + '_net.png', show_shapes=True, show_layer_names=True,rankdir='LR')
        if(Tune):
            model.load_weights(type+'_checkpoint_weights.h5')
    elif(type=="EfficientNetB4Mount" or type=="EfficientNetB4Above"):
        input = Input(shape=(224, 224, 3))
        model = EfficientNetB4(input,type,Tune,AuxiliaryLoss=Auxiliary_Loss,BlockFirsts=BlockFirsts)
        print(model.summary())
        #plot_model(model, to_file=type + '_net.png', show_shapes=True, show_layer_names=True,rankdir='LR')
        if(Tune):
            model.load_weights(type+'_checkpoint_weights.h5')
    elif(type=="EfficientNetB5Mount" or type=="EfficientNetB5Above"):
        input = Input(shape=(224, 224, 3))
        model = EfficientNetB5(input,type,Tune,AuxiliaryLoss=Auxiliary_Loss,BlockFirsts=BlockFirsts)
        print(model.summary())
        #plot_model(model, to_file=type + '_net.png', show_shapes=True, show_layer_names=True,rankdir='LR')
        if(Tune):
            model.load_weights(type+'_checkpoint_weights.h5')
    elif(type=="EfficientNetB7Mount" or type=="EfficientNetB7Above"):
        input = Input(shape=(224, 224, 3))
        model = EfficientNetB7(input,type,Tune,AuxiliaryLoss=Auxiliary_Loss,BlockFirsts=BlockFirsts)
        print(model.summary())
        #plot_model(model, to_file=type + '_net.png', show_shapes=True, show_layer_names=True,rankdir='LR')
        if(Tune):
            model.load_weights(type+'_checkpoint_weights.h5')
    return model




def TNet_2Camera(type,Tune=False,Auxiliary_Loss=True,BlockFirsts=False):
    if(type=="ResNet152V2Stereo"):
        input1 = Input(shape=(224, 224, 3))
        input2 = Input(shape=(224, 224, 3))
        model = ResNet152V2Stereo(input1,input2,type,Tune,AuxiliaryLoss=Auxiliary_Loss,BlockFirsts=BlockFirsts)
        print(model.summary())
        #plot_model(model, to_file=type + '_net.png', show_shapes=True, show_layer_names=True,rankdir='LR')
        if(Tune):
            model.load_weights(type+'_checkpoint_weights.h5')
    if(type=="ResNet50Stereo"):
        input1 = Input(shape=(224, 224, 3))
        input2 = Input(shape=(224, 224, 3))
        model = ResNet50Stereo(input1,input2,type,Tune,AuxiliaryLoss=Auxiliary_Loss,BlockFirsts=BlockFirsts)
        print(model.summary())
        #plot_model(model, to_file=type + '_net.png', show_shapes=True, show_layer_names=True,rankdir='LR')
        if(Tune):
            model.load_weights(type+'_checkpoint_weights.h5')
    elif(type=="InceotresnetStereo"):
        input1 = Input(shape=(224, 224, 3))
        input2 = Input(shape=(224, 224, 3))
        model = InceptResnetStereo(input1,input2,type,Tune,AuxiliaryLoss=Auxiliary_Loss,BlockFirsts=BlockFirsts)
        print(model.summary())
        #plot_model(model, to_file=type + '_net.png', show_shapes=True, show_layer_names=True,rankdir='LR')
        if(Tune):
            model.load_weights(type+'_checkpoint_weights.h5')
    elif(type=="InceptionDenseNetStereo"):
        input1 = Input(shape=(224, 224, 3))
        input2 = Input(shape=(224, 224, 3))
        model = InceptDenseNetStereo(input1,input2,type,Tune,AuxiliaryLoss=Auxiliary_Loss)
        print(model.summary())
        #plot_model(model, to_file=type + '_net.png', show_shapes=True, show_layer_names=True,rankdir='LR')
        if(Tune):
            model.load_weights(type+'_checkpoint_weights.h5')
    elif(type=="InceptionResNetV2Stereo"):
        input1 = Input(shape=(224, 224, 3))
        input2 = Input(shape=(224, 224, 3))
        model = InceptionResNetV2Stereo(input1,input2,type,Tune,AuxiliaryLoss=Auxiliary_Loss,BlockFirsts=BlockFirsts)
        print(model.summary())
        #plot_model(model, to_file=type + '_net.png', show_shapes=True, show_layer_names=True,rankdir='LR')
        if(Tune):
            model.load_weights(type+'_checkpoint_weights.h5')
    elif(type=="EfficientNetB4Stereo"):
        input1 = Input(shape=(224, 224, 3))
        input2 = Input(shape=(224, 224, 3))
        model = EfficientNetB4Stereo(input1,input2,type,Tune,AuxiliaryLoss=Auxiliary_Loss,BlockFirsts=BlockFirsts)
        print(model.summary())
        #plot_model(model, to_file=type + '_net.png', show_shapes=True, show_layer_names=True,rankdir='LR')
        if(Tune):
            model.load_weights(type+'_checkpoint_weights.h5')
    elif(type=="XceptionStereo"):
        input1 = Input(shape=(224, 224, 3))
        input2 = Input(shape=(224, 224, 3))
        model = XceptionStereo(input1,input2,type,Tune,AuxiliaryLoss=Auxiliary_Loss,BlockFirsts=BlockFirsts)
        print(model.summary())
        #plot_model(model, to_file=type + '_net.png', show_shapes=True, show_layer_names=True,rankdir='LR')
        if(Tune):
            model.load_weights(type+'_checkpoint_weights.h5')
    elif(type=="EfficientNetB47Stereo"):
        input1 = Input(shape=(224, 224, 3))
        input2 = Input(shape=(224, 224, 3))
        model = EfficientNetB47Stereo(input1,input2,type,Tune,AuxiliaryLoss=Auxiliary_Loss,BlockFirsts=BlockFirsts)
        print(model.summary())
        plot_model(model, to_file='model.pdf')
        if(Tune):
            model.load_weights(type+'_checkpoint_weights.h5')
    return model

from dataset_prepare import *
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import CSVLogger

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle
from tnet import euc_loss1r
from tnet import euc_loss2r
from tnet import euc_loss3r
from tnet import euc_loss1x
from tnet import euc_loss2x
from tnet import euc_loss3x
from tnet import TNet_2Camera
from result_plot import plot_loss
from result_csvplot import csvplot_loss
import pandas as pd


def scheduler(epoch, lr):
    print("################################### 0 #######################################")
    print(str(epoch)+"-----"+str(lr))
    print("#############################################################################")
    return lr
    if(epoch==10):
        return 1e-6 
    if(epoch==20):
        return 1e-7 
    if(epoch==100):
        return 1e-5 
    if(epoch==120):
        return 1e-5 
    return lr


def scheduler2(epoch, lr):
  if epoch < 8:
    print("################################### 0 #######################################")
    print(str(epoch)+"-----"+str(lr))
    print("#############################################################################")
    return 1e-4
  elif(epoch >=8 and epoch <25):
    print("################################### 1 #######################################")
    print(str(epoch)+"-----"+str(lr))
    print("#############################################################################")
    return 1e-5
  elif(epoch >=25 and epoch <35):
    print("################################## 2 ########################################")
    print(str(epoch)+"-----"+str(lr))
    print("#############################################################################")
    return 1e-6
  else:
    print("################################## 3 #########################################")
    print(str(epoch)+"-----"+str(lr))
    print("#############################################################################")
    return 1e-7     
if __name__ == "__main__":

    plot=False
    #plot=True

    Auxiliary_loss=True
    Auxiliary_loss=False

    #BlockFirsts=True
    BlockFirsts=False

    #Tune=True
    Tune=False   

    Load_Data=True;
    #Load_Data=False;

    camera='Stereo'
    #model_name="InceptionDenseNet"
    #model_name="Inceotresnet"
    #model_name="ResNet50"
    #model_name="ResNet152V2"
    #model_name="InceptionResNetV2"
    #model_name="Xception"
    #model_name="EfficientNetB4"
    model_name="EfficientNetB47"

    train_size=0.85
    learning_rate=1e-5
    batch_size = 40


    epoch_num=600
    epoch=0
    model_name=model_name + camera

    #model = TNet_2Camera(model_name,Tune,Auxiliary_loss,BlockFirsts) 

    if(Load_Data == False):
        data_prepare_stereo()


    [above,mount1,_,dataset,Y]=pickle.load(open(camera+'.pickle', 'rb'))
   
    size=np.shape(above)[0]
    #size=700
    s1=int(size*train_size)
    s2=size-s1 
    X_Above_train=above[0:s1]
    X_Above_test=above[s1:s1+s2]
    above=0

    X_Mount1_train=mount1[0:s1]
    X_Mount1_test=mount1[s1:s1+s2]
    mount1=0

    #X_Mount2_train=mount2[0:s1]
    #X_Mount2_test=mount2[s1:s1+s2]
    #mount2=0

    y_train=Y[0:s1]
    y_test=Y[s1:s1+s2]
    Y=0
         

    train_input=[X_Above_train,X_Mount1_train] 
    test_input=[X_Above_test,X_Mount1_test]


    if(Auxiliary_loss==False):
        train_out=[np.array(y_train[:,0:2]),np.array(y_train[:,2])]
        test_output=[np.array(y_test[:,0:2]),np.array(y_test[:,2])]

    else:
        train_out=[ np.array(y_train[:,0:2]),np.array(y_train[:,2]) , np.array(y_train[:,0:2]),np.array(y_train[:,2]) , np.array(y_train[:,0:2]),np.array(y_train[:,2])  ]
        test_output=[np.array(y_test[:,0:2]),np.array(y_test[:,2])   , np.array(y_test[:,0:2]),np.array(y_test[:,2])   , np.array(y_test[:,0:2]),np.array(y_test[:,2])    ]



    
    if(plot==True):
       csvplot_loss(model_name)
       exit

    if(Auxiliary_loss==False):
        loss_functions={'C1'+model_name+'_pose_xy': euc_loss3x, 'C1'+model_name+'_pose_r': euc_loss3r}
    else:
        loss_functions={'C1'+model_name+'_pose_xy': euc_loss3x, 'C1'+model_name+'_pose_r': euc_loss3r,
                        'C2'+model_name+'_pose_xy': euc_loss3x, 'C2'+model_name+'_pose_r': euc_loss3r,
                             model_name+'_pose_xy': euc_loss3x,  model_name+'_pose_r': euc_loss3r}


    
    #Two Callbacks for learning rate and checkpointer    
    LRScheduler = LearningRateScheduler(scheduler)
    checkpointer = ModelCheckpoint(filepath=model_name+"_checkpoint_weights.h5", verbose=1, save_best_only=True, save_weights_only=True)
    LRReduce = ReduceLROnPlateau(monitor='val_loss', factor=0.1   ,patience=14, min_lr=1e-11,verbose=1)
    early_stopping = EarlyStopping(patience=20,verbose=1)
    csv_logger = CSVLogger(model_name+"_history_log.csv", append=True)

    

    if(Tune==False):
        try:
            os.remove(model_name+"_history_log.csv")
        except:
            print("file not existed!")
    else:
        try:
            history=pd.read_csv(model_name+"_history_log.csv")
            row_count = sum(1 for row in history['lr'])
            #learning_rate = history['lr'][row_count-1]
            #epoch=history['epoch'][row_count-1]+1
        except Exception as e:
            print("Oops!", e.__class__, "occurred.")

          

    model = TNet_2Camera(model_name,Tune,Auxiliary_loss,BlockFirsts) 
    adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-10,clipvalue=0.5)
    model.compile(optimizer=adam, loss=loss_functions)
    


    try:
       history = model.fit(train_input,train_out, batch_size=batch_size, epochs=epoch_num, validation_data=(test_input,test_output),initial_epoch=epoch, callbacks=[early_stopping, checkpointer,LRReduce,LRScheduler,csv_logger],use_multiprocessing=True)
    except Exception as e:
        print("Oops!", e._message, "occurred.")
    
    
    csvplot_loss(model_name)


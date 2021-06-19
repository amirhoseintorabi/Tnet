import numpy as np
import matplotlib.pyplot as plt
import pickle

def plot_loss(model_name='ResNet50'):
    history = pickle.load(open(model_name+'_training_history.pickle', 'rb'))
    number=3700
    # summarize history for loss
    plt.plot(history[model_name+'_pose_xy_loss'][0:number])
    plt.plot(history['val_'+model_name+'_pose_xy_loss'][0:number])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history[model_name+'_pose_r_loss'][0:number])
    plt.plot(history['val_'+model_name+'_pose_r_loss'][0:number])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    # summarize history for loss
    plt.plot(history['loss'][0:number])
    plt.plot(history['val_loss'][0:number])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

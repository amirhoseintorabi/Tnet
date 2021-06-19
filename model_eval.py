import math
import helper
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
import timeit
from sklearn.model_selection import train_test_split
import tnet
import cv2

if __name__ == "__main__":
 
    
    [X_test_mount1,y_test,mean2]=pickle.load(open('eval.pickle', 'rb'))

            
    X_test1 = np.squeeze(np.array(X_test_mount1))
    #X_test2 = np.squeeze(np.array(test_images_above))
    y_test = np.squeeze(np.array(y_test))
    
    #X_test=X_test[0:50,:]
    #y_test=y_test[0:50,:]


    #model = tnet.googlenet_pose("full_separated_megahighres")
    model =     model = tnet.tnet_mount("full_separated_megahighres")

    model.load_weights('checkpoint_weights.h5')

    start = timeit.default_timer()
    testPredict = model.predict([X_test1])
    stop = timeit.default_timer()
    print('Time: ', (stop - start)/X_test1.shape[0]) 

    results = np.zeros((len(X_test1),2))
    for i in range(len(X_test1)):

        pose_r= np.asarray(y_test[i][2])
        pose_x= np.asarray(y_test[i][0:2])
        
        predicted_x = testPredict[0][i]
        predicted_r = testPredict[1][i]
        predicted_x2 = testPredict[2][i]
        predicted_r2 = testPredict[3][i]
        predicted_x3 = testPredict[4][i]
        predicted_r3 = testPredict[5][i]
        
        #pose_q=np.asarray(pose_q, dtype='float32')
        #pose_x=np.asarray(pose_x, dtype='float32')
        #predicted_q=np.asarray(predicted_q, dtype='float32')
        #predicted_x=np.asarray(predicted_x, dtype='float32')

        #pose_q =np.round(np.squeeze(pose_q),2)
        #pose_x = np.round(np.squeeze(pose_x),2)
        #predicted_q = np.round(np.squeeze(predicted_q),2)
        #predicted_x = np.round(np.squeeze(predicted_x),2)

        #predicted_fl = np.round(np.squeeze(predicted_fl),1)
        #pose_fl = np.round(np.squeeze(pose_fl),1)

        #predicted_x[0] = predicted_x[0]*100
        #predicted_x[1] = predicted_x[1]*100
        #predicted_x[2] = predicted_x[2]*100

        fig, ax = plt.subplots()
        ax.set_xlabel("      true values=>"+str(pose_x)+" ,  "+str(pose_r)+'\n'+"predicted values=>"+str(predicted_x)+" ,  "+str(predicted_r)
                                                                          +'\n'+"predicted values2=>"+str(predicted_x2)+" ,  "+str(predicted_r2) 
                                                                          +'\n'+"predicted values3=>"+str(predicted_x3)+" ,  "+str(predicted_r3) )
        plt.imshow(X_test1[i]) 
        manager = plt.get_current_fig_manager()        
        manager.window.state('zoomed')
        plt.show()
        
        #image = cv2.imread('D:/dataset/data/' + str(pose_x[0]) + "_" + str(pose_x[1]) + "_" + str(pose_r)+'/MountCamera.jpg')
        #plt.imshow(image)
        #plt.show()

        #Compute Individual Sample Error
        #q1 = pose_q / np.linalg.norm(pose_q)
        #q2 = np.round(predicted_q / np.linalg.norm(predicted_q),2)
        #d = abs(np.sum(np.multiply(q1,q2)))
        #theta = 2 * np.arccos(d) * 180/math.pi

        theta = np.linalg.norm(pose_r-predicted_r)
        theta = np.round(theta,3)
        error_x = np.round(np.linalg.norm(pose_x-predicted_x),3)        
        results[i,:] = [error_x,theta]
        print('Iteration:  '+str(i)+'  Error XYZ (cm):  '+str(error_x)+'  Error Q:  '+str(theta))
              
    median_result = np.median(results,axis=0)
    print('Median error ', median_result[0], 'cm  and ', median_result[1], 'degrees.')
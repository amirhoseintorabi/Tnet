import math
import helper
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
import timeit

if __name__ == "__main__":
 
    
    load_variable = True;
    #load_variable = False;
    if(load_variable!=True):
        test_images_mount, test_images_above, poses,testimageraw = helper.Get_TestData()
        with open('test.pickle', 'wb') as f:
            pickle.dump([test_images_mount, test_images_above, poses,testimageraw], f)
    else:
        with open('test.pickle', 'rb') as f:
            test_images_mount, test_images_above, poses,testimageraw = pickle.load(f)


            
    X_test1 = np.squeeze(np.array(test_images_mount))
    X_test2 = np.squeeze(np.array(test_images_above))
    y_test = np.squeeze(np.array(poses))
    
    #X_test=X_test[0:50,:]
    #y_test=y_test[0:50,:]

    import posenet
    from keras.optimizers import Adam


    model = posenet.create_posenet()
    model.load_weights('checkpoint_weights.h5')
    #adam = Adam(lr=0.01, beta_1=0.5, beta_2=0.999, epsilon=1e-06, decay=0.0, clipvalue=2.0)
    #model.compile(optimizer=adam, loss={'cls1_fc_pose_xyz': posenet.euc_loss1x, 'cls1_fc_pose_wpqr': posenet.euc_loss1q,'cls1_fc_fl': posenet.euc_loss1fl,
    #                                    'cls2_fc_pose_xyz': posenet.euc_loss2x, 'cls2_fc_pose_wpqr': posenet.euc_loss2q,'cls2_fc_fl': posenet.euc_loss2fl,
    #                                    'cls3_fc_pose_xyz': posenet.euc_loss3x, 'cls3_fc_pose_wpqr': posenet.euc_loss3q,'cls3_fc_fl': posenet.euc_loss3fl})

    start = timeit.default_timer()
    testPredict = model.predict([X_test1,X_test2])
    stop = timeit.default_timer()
    print('Time: ', (stop - start)/X_test1.shape[0]) 

    valsx = testPredict[4]
    valsq = testPredict[5]
    #valsfl = testPredict[8]
    
# Get results... :/
    results = np.zeros((len(test_images_mount),2))
    for i in range(len(test_images_mount)):

        pose_q= np.asarray(poses[i][3:7])
        pose_fl= np.asarray(poses[i][7:8])
        pose_x= np.asarray(poses[i][0:3])
        predicted_x = valsx[i]
        #predicted_fl = valsfl[i]
        predicted_q = valsq[i]
        
        pose_q=np.asarray(pose_q, dtype='float32')
        pose_x=np.asarray(pose_x, dtype='float32')
        predicted_q=np.asarray(predicted_q, dtype='float32')
        predicted_x=np.asarray(predicted_x, dtype='float32')

        pose_q =np.round(np.squeeze(pose_q),2)
        pose_x = np.round(np.squeeze(pose_x),2)
        predicted_q = np.round(np.squeeze(predicted_q),2)
        predicted_x = np.round(np.squeeze(predicted_x),2)

        #predicted_fl = np.round(np.squeeze(predicted_fl),1)
        #pose_fl = np.round(np.squeeze(pose_fl),1)

        #predicted_x[0] = predicted_x[0]*100
        #predicted_x[1] = predicted_x[1]*100
        #predicted_x[2] = predicted_x[2]*100

        fig, ax = plt.subplots()
        ax.set_xlabel("      true values=>"+str(pose_x)+" ,  "+str(pose_q)+'\n'+"predicted values=>"+str(predicted_x)+" ,  "+str(predicted_q))
        plt.imshow(testimageraw[i]) 
        manager = plt.get_current_fig_manager()        
        manager.window.state('zoomed')
        plt.show()


        #Compute Individual Sample Error
        #q1 = pose_q / np.linalg.norm(pose_q)
        #q2 = np.round(predicted_q / np.linalg.norm(predicted_q),2)
        #d = abs(np.sum(np.multiply(q1,q2)))
        #theta = 2 * np.arccos(d) * 180/math.pi

        theta = np.linalg.norm(pose_q-predicted_q)
        theta = np.round(theta,3)
        error_x = np.round(np.linalg.norm(pose_x-predicted_x),3)        
        #error_LDI = np.round(pose_fl-predicted_fl,2)

        results[i,:] = [error_x,theta]
        print('Iteration:  '+str(i)+'  Error XYZ (cm):  '+str(error_x)+'  Error Q:  '+str(theta))
              
    median_result = np.median(results,axis=0)
    print('Median error ', median_result[0], 'cm  and ', median_result[1], 'degrees.')
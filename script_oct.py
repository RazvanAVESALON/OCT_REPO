from email.mime import image
import os
from sqlite3 import Date
import cv2 as cv
import data
import network
import preprocessing
import GlobalSettings as GS
import tensorflow as tf 
import numpy as np
from time import time

import matplotlib.pyplot as plt
# from GAN.dcgan import DCGAN
# from GAN.pix2pix import Pix2Pix
#from GAN.cyclegan import CycleGAN
from  copy import deepcopy
from multiprocessing import Process
import losses
import focal_tversky_unet as attention_unet
import glob
import json
from sklearn.metrics import confusion_matrix


def dice_coef(y_true, y_pred):
    y_true_f = tf.reshape(tf.dtypes.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.dtypes.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1.)


def reg3simpla(x,y):
  z=x
  for i in range(y):
    z[i][0]=(x[i][0]*1024)/704
    z[i][1]=(x[i][1]*1024)/704
  
  return z
  
def transfom_into_binary(data,j):
    for k in data.keys():
        if k=='plaques':
            for p in range(len(data['plaques'])):
                if (data['plaques'][p]['morphology']=='Calcium nodule'):
                     for s in data['plaques'][p]["contours"]:
                        points = data['plaques'][p]["contours"][s]["control_pts"]
                        pts = np.array(points, np.int32)
                        print(pts.shape)
                        pts=reg3simpla(pts,pts.shape[0])
                        print (pts)
                        
                   
                        if data['plaques'][p]["contours"][s]["closed"]==True:
                            img=np.zeros((1024,1024,3), np.uint8)
                            filled = cv.fillPoly(img, pts = [pts], color =(255,255,255))
                            print(filled)
                            path=r"Imagini"
                            cv.imwrite(os.path.join(path, 'Adnotare_binara'+'_'+str(p)+'_'+str(s)+'.png'),filled)
                            
def overlap(gt,pred):
 print(gt.shape, gt.dtype)
 print(gt.min(), gt.max())
 gt = gt[:, :, 0]
 pred = pred[:, :, 0]
 
 tp = gt & pred
 fp = ~gt & pred
 fn =  gt & ~pred
 tn = ~gt & ~pred

 print(tp.min(), tp.max(),fp.min(),fp.max(),fn.min(),fn.max())

 img=np.zeros((1024,1024,3), np.uint8) 
 img[:,:,1] = tp
 img[:,:,2] = fp
 img [:,:,0]= fn
 
 print(img.min(), img.max())
 plt.imshow(img)
 plt.show()
 
#  cv.destroyAllWindows() 

 
 
 
                                    
                        
                            
                              
if __name__=='__main__':   
  # jsons = glob.glob(r"D:\ai intro\OCT\Adnotari\*")
  # for j in jsons:
  #     print (j)
  #     with open(j) as f:
  #          date = json.load(f)
  #     print (date.keys())
  #     transfom_into_binary(date,j)    
        
  gt=cv.imread(r"D:\ai intro\OCT\OCT_REPO\Imagini\Adnotare_binara_0_64.png")
  pred=cv.imread(r"D:\ai intro\OCT\OCT_REPO\Imagini\Adnotare_binara_0_66.png")
  overlap(gt,pred)
   
   

                            
                            
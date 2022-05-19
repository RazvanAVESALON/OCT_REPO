from email.mime import image
import os
import cv2 
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
  
def transfom_into_binary(jsons ):
    
    
    for j in jsons:
        print (j)
        with open(j) as f:
            data = json.load(f)
        print (data.keys())
        for k in data.keys():
            if k=='plaques':
                for p in range(len(data['plaques'])):
                    if (data['plaques'][p]['morphology']=='Calcium nodule'):
                        for s in data['plaques'][p]["contours"]:
                            points = data['plaques'][p]["contours"][s]["control_pts"]
                            pts = np.array(points, np.int32)
                            print(pts.shape)
                            pts=reg3simpla(pts,pts.shape[0])
                            img=cv2.imread(r"D:\ai intro\OCT\OCT_REPO\Solid_black.svg.png")
                            filled = cv2.fillPoly(img, pts = [pts], color =(255,255,255))
                          
                            cv2.imwrite(f"D:\\ai intro\\OCT\\OCT_REPO\\Imagini binare\\Adnotare_binara{f}_{s}.png",filled)
                            cv2.imshow("filled",filled)
                            cv2.waitKey()
                            cv2.destroyAllWindows()
                        
                            
                              
if __name__=='__main__':   
   jsons = glob.glob(r"D:\ai intro\OCT\Adnotari\*")
   images=glob.glob(r"E:\AchizitiiOctombrieUMF2021OCT\*")
   print(jsons)
   transfom_into_binary(jsons)
   
   

                            
                            
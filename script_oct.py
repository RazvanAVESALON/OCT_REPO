from ast import Slice
from email.mime import image
import os
from re import I
from sqlite3 import Date
import cv2 as cv
import data
import network
import preprocessing
import GlobalSettings as GS
import tensorflow as tf 
import numpy as np
from time import time
import pathlib as pt
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
import pandas as pd 
import keras
import SimpleITK as sitk
import pydicom
from data import multiprocess
os.environ["CUDA_VISIBLE_DEVICES"] = GS.GPU

def get_model():
    if GS.NET == 'ResNet':
        model = network.resnet(filter=16, dropout=0.3)
        model.summary()
        tf.keras.utils.plot_model(model, 'model_ResNet.png')
    elif GS.NET == 'DenseNet':
        model = network.densenet(filter=8, k=32, N=2, dropout=0.3)
        model.summary()
        tf.keras.utils.plot_model(model, 'model_DenseNet.png')
    elif GS.NET == 'UNet':
        model = network.unet(filter=12, dropout=0.1)
        model.summary()
        #optimizer = tf.keras.optimizers.Adam(lr=2**-12)
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=2**-12),
                  loss=jaccard_loss_per_sample, metrics=[iou_score])
        tf.keras.utils.plot_model(model, 'model_UNet.png')
    elif GS.NET == 'VGG16':
        model = network.vgg16(256, 0.0)
        model.summary()
        optimizer = keras.optimizers.Adam(lr=2**-12)
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=2**-12),
                  loss=jaccard_loss_per_sample, metrics=[iou_score])
        tf.keras.utils.plot_model(model, 'model_VGG16.png', show_shapes=True)
    elif GS.NET == 'CombinedNet':
        model = network.get_combined_model(filter=1 ,dropout_rate=0.5)
        model.summary()
        tf.keras.utils.plot_model(model, 'model_CombinedNet.png')
    elif GS.NET == 'ResNet50':
        model = network.resnet_pretrained(filter=64, dropout_rate=0.2)
        model.summary()
        tf.keras.utils.plot_model(model, 'model_ResNet50.png')
    elif GS.NET == 'UNet_pretrained':
        model = network.unet_pretrained(filter=4, dropout_rate=0.1)
        model.summary()
        optimizer = keras.optimizers.Adam(lr=2**-12)
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=2**-12),
                  loss=jaccard_loss_per_sample, metrics=[iou_score])
        tf.keras.utils.plot_model(model, "model_UNet_pretrained.png", show_shapes=True, show_layer_names=True)
    elif GS.NET == 'FocalTverskyUnet':
        adam = keras.optimizers.Adam(lr=2**-12)
        #model = attention_unet.attn_reg_small_3M(adam, (512, 512, 1), losses.focal_tversky)
        model = attention_unet.attn_reg_small_1M(adam, (512, 512, 1), losses.focal_tversky)
        model.summary()
        #tf.keras.utils.plot_model(model, "model_Focal_Tversky_Unet.png", show_shapes=True, show_layer_names=True)
        init_g = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_g)
            sess.run(init_l)

    return model


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
  
def transfom_into_binary(data,j,path_to_save): 
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
                            cv.imwrite(os.path.join(path_to_save, 'Adnotare_binara'+'_'+str(p)+'_'+str(s)+'.png'),filled)
                                
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

def read_dicom_without_annotations(path):
    dicom_list = []
    try:
        p = pt.Path(os.path.join(path))
        print(p.exists())
        OCTDICOM = sitk.ReadImage(os.path.join(path))
        images = sitk.GetArrayFromImage(OCTDICOM)
        dcm = pydicom.read_file(os.path.join(path), stop_before_pixels=True)
                
        tags = dcm[0x0018, 0x6011]
        x0 = tags[0][0x0018, 0x6018].value
        y0 = tags[0][0x0018, 0x601A].value
        x1 = tags[0][0x0018, 0x601C].value
        y1 = tags[0][0x0018, 0x601E].value
                
        images = images[:, y0:y1, x0:x1, :]
                
        images_temp = []
        for count, image in enumerate(images):
            image = preprocessing.remove_lumen(image)
            image = preprocessing.cut_text_from_oct_image(image)
            image = preprocessing.remove_speckle(image)
            image = multiprocess(image, [], None, None, [], [], None)
            #    #plt.imshow(image)
            #    #plt.show()
            #    image = preprocessing.remove_lumen(image)
            #    #plt.imshow(image)
            #    #plt.show()
            #    image = preprocessing.cut_text_from_oct_image(image)
            #    #plt.imshow(image)
            #    #plt.show()
            #    image = preprocessing.remove_speckle(image)

            #    image = cv2.resize(image, (512,512))
          
            images_temp.append(image[2][0])
          
            

            dicom_list.append({'slices':images_temp})
            
                   
            
    except:
        print("Could not read dicom ")
    
    print (len(dicom_list),"aa", len(dicom_list[0]),'bb',len(dicom_list[0]['slices'])) 
    return dicom_list

def test_on_dicom():
  model = get_model()
  model.load_weights(r"D:\ai intro\OCT\CalcifiedPlaqueDetection\models\1556888684.747263_Cartesian_FocalTverskyUnet_Gray.h5")
  csv_adnotari=pd.read_csv(r"D:\ai intro\OCT\OCT_REPO\test.csv")
  caile_imaginii=csv_adnotari['image_path'].unique()


  for index in range(len(caile_imaginii)):
        os.mkdir(f"D:\\ai intro\\OCT\\OCT_REPO\\PREDICTII_IMG{index+1}")
        data_cartesian_per_dicom = read_dicom_without_annotations(caile_imaginii[index])
        
        for slice , image in enumerate(data_cartesian_per_dicom[0]['slices']):
            
                image= np.expand_dims(image, axis=0)
                prediction = model.predict(image)
                prediction = prediction[-1]
                prediction = prediction.reshape((512, 512))
                prediction= cv.resize(prediction,(1024,1024))
                print(prediction.max(),prediction.min())
                prediction[prediction > 0.5] = 1.0
                prediction[prediction <= 0.5] = 0.0
                print(prediction.max(),prediction.min())
                
                #prediction = reg3simpla(prediction,prediction.shape[1])
                       
                # plt.subplot(1,2,1)
                # cv.imshow(image[0,:,:,0])
                
                # cv.imshow(prediction)
                # cv.imwrite(f"D:\\ai intro\\OCT\\OCT_REPO\\models\\PREDICTII_IMG{csv_adnotari['image_index'][index]}"+"\\"+"Suprapunere"+str(slice)+".png")
                
                
                plt.imshow(prediction,cmap='gray')
                plt.savefig(f"D:\\ai intro\\OCT\\OCT_REPO\\PREDICTII_IMG{index+1}"+"\\"+"Predictie"+str(slice)+".png")
                #path=f"D:\\ai intro\\OCT\\OCT_REPO\\PREDICTII_IMG{index+1}"
                #cv.imwrite(os.path.join(path, 'PREDICTIE'+'_'+str(slice)+'.png'),prediction)
                #plt.show() 


    # for i in range( len(data_cartesian_per_dicom)):
    #     for count, image in enumerate(data_cartesian_per_dicom[i]['slices']):
    #         for s in range(data_cartesian_per_dicom[i]['slices']):
    #             if s 
    #             image= np.expand_dims(image, axis=0)
    #             prediction = model.predict(image)
    #             prediction = prediction[-1]
    #             prediction = reg3simpla(prediction.shape[1])
    #             os.mkdir(f"D:\\ai intro\\OCT\\CalcifiedPlaqueDetection\\models\PREDICTII{i}")
    #             # plt.subplot(1,2,1)
    #             plt.imshow(image[0,:,:,0], cmap='gray')
    #             #plt.subplot(1,2,2)
    #             plt.imshow(prediction, alpha=.5, cmap='gray')
    #             plt.savefig(f"D:\\ai intro\\OCT\\CalcifiedPlaqueDetection\\models\PREDICTII{i}"+"\\"+"Suprapunere"+str(count)+".png")


    #             plt.imshow(prediction,cmap='gray')
    #             plt.savefig(f"D:\\ai intro\OCT\CalcifiedPlaqueDetection\models\PREDICTII{i}"+"\\"+"Predictie"+str(count)+".png")
    #             #plt.show()

    
                                    
                        
                            
                              
if __name__=='__main__':   
#   jsons = glob.glob(r"D:\ai intro\OCT\Adnotari\*")
#   for j in jsons:
#       print (j)
#       with open(j) as f:
#            date = json.load(f)
#       print (date.keys())
#       os.mkdir(f"D:\\ai intro\\OCT\\OCT_REPO\\Imagini\\ADNOTARI_BINARE_{os.path.basename(j)}")
#       path_to_save=f"D:\\ai intro\\OCT\\OCT_REPO\\Imagini\\ADNOTARI_BINARE_{os.path.basename(j)}"
#       transfom_into_binary(date,j,path_to_save)    
        
#   gt=cv.imread(r"D:\ai intro\OCT\OCT_REPO\Imagini\Adnotare_binara_0_64.png")
#   pred=cv.imread(r"D:\ai intro\OCT\OCT_REPO\Imagini\Adnotare_binara_0_66.png")
#   overlap(gt,pred)
 test_on_dicom() 
   
   

                            
                            
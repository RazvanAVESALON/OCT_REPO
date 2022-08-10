from __future__ import annotations
from ast import Slice
import csv
from email.mime import image
from hashlib import new
import os
from pickle import TRUE
from re import I
import re
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
from sklearn.metrics  import  f1_score , confusion_matrix , precision_score, recall_score  , jaccard_score
from segmentation_models.metrics import iou_score
from scipy.stats import pearsonr
from CalcifiedPlaqueDetection import test , test_dice

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

    y_true_f = y_true.astype('float32')
    print (y_pred)
    y_pred_f = y_pred.astype('float32')
    y_true_f=y_true_f.flatten()
    y_pred_f=y_pred_f.flatten()
    union = np.sum(y_true_f) + np.sum(y_pred_f)
    if union==0: return 1
    intersection = np.sum(y_true_f * y_pred_f)
    return 2. * intersection / union

    # y_true_f = tf.reshape(tf.dtypes.cast(y_true, tf.float32), [-1])
    # y_pred_f = tf.reshape(tf.dtypes.cast(y_pred, tf.float32), [-1])
    # intersection = tf.reduce_sum(y_true_f * y_pred_f)
    # return (2. * intersection + 1.) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1.)


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
                            cv.imwrite(os.path.join(path_to_save, 'Adnotare_binara'+'_'+str(s)+'.png'),filled)
                            
 
 

                                
def overlap(gt,pred):
 print(gt.shape, gt.dtype)
 print(gt.min(), gt.max())

 
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
 #plt.imshow(img)
 #plt.show()
 return img 
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
            
            cv.imwrite(os.path.join(r'D:\ai intro\OCT\OCT_REPO\Before preprocessing', 'IMG_1'+f'_preprocesare{count}'+'.png'),image)
            image = preprocessing.remove_lumen(image)
            
            cv.imwrite(os.path.join(r'D:\ai intro\OCT\OCT_REPO\Remove_lumen', 'IMG_1'+f'_removelumen{count}'+'.png'),image)
            image = preprocessing.cut_text_from_oct_image(image)

            cv.imwrite(os.path.join(r'D:\ai intro\OCT\OCT_REPO\Cut_text', 'IMG_1'+f'_cut_text{count}'+'.png'),image)  

            image = preprocessing.remove_speckle(image)
            cv.imwrite(os.path.join(r'D:\ai intro\OCT\OCT_REPO\remove_spekle', 'IMG_1'+f'_removespeckle{count}'+'.png'),image)

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
    
    return dicom_list

def test_on_dicom():
    model = get_model()
    model.load_weights(r"D:\ai intro\OCT\CalcifiedPlaqueDetection\models\1556888684.747263_Cartesian_FocalTverskyUnet_Gray.h5")
    csv_adnotari=pd.read_csv(r"D:\ai intro\OCT\OCT_REPO\DICE_ADDED_v2.csv")
    caile_imaginii=csv_adnotari['image_path'].unique()
    print(caile_imaginii)
    #caile_imaginii=caile_imaginii[1]
    
    for path in caile_imaginii:
        print (path)
        img_name = os.path.basename(path)
        print(img_name)
        metrics={'dice':[],'jaccard':[],'iou':[]}
        os.mkdir(f"D:\\ai intro\\OCT\\OCT_REPO\\Predictii_v3\\PREDICTII_V3_{img_name}")
        data_cartesian_per_dicom = read_dicom_without_annotations(path)
        if data_cartesian_per_dicom:
            for slice , image in enumerate(data_cartesian_per_dicom[0]['slices']):
                   
                    image= np.expand_dims(image, axis=0)
                    
                    image=cv.resize(image,(512,512), fx=0.5, fy=0.5)
                    prediction = model.predict(image)
                    prediction = prediction[-1]
                    prediction = prediction.reshape((512, 512))
                    dice_idx,jaccard,iou=test_dice(prediction,image)
                    
                    metrics['dice'].append(dice_idx)
                    metrics['jaccard'].append(jaccard)
                    metrics['iou'].append(iou)
                
                    prediction= cv.resize(prediction,(1024,1024))
                    
                    #tp,fp,fn=poz_negs_calculator(image,prediction)
                    
                    prediction[prediction>=0.1]=255
                    prediction[prediction<0.1]=0
                    prediction = reg3simpla(prediction,prediction.shape[1])
                        
                    #plt.subplot(1,2,1)
                    #cv.imshow(image[0,:,:,0])
                    
                    
                    #cv.imwrite(f"D:\\ai intro\\OCT\\OCT_REPO\\models\\PREDICTII_IMG_____{csv_adnotari['image_index'][index]}"+"\\"+"Suprapunere"+str(slice)+".png")
                    
                    
                    #plt.imshow(prediction,cmap='gray')
                    #plt.savefig(f"D:\\ai intro\\OCT\\OCT_REPO\\PREDICTII_IMG{index+1}"+"\\"+"Predictie"+str(slice)+".png")
                    output=f"D:\\ai intro\\OCT\\OCT_REPO\\Predictii_v3\\PREDICTII_V3_{img_name}"
                    cv.imwrite(os.path.join(output, 'PREDICTIE'+'_'+str(slice)+'.png'),prediction)
                    #plt.show()
        df= pd.DataFrame(metrics)
        print(df.head())
        df.to_csv(os.path.join(f"D:\\ai intro\\OCT\\OCT_REPO\\Predictii_v3\\PREDICTII_V3_{img_name}","METRICS"+'_'+str(img_name)+'.csv'), index=False)
                     





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

def metrics_calculation(gt,pred):
    gt=gt.flatten()
    pred=pred.flatten()
    tp,fp,fn,tn=confusion_matrix(gt,pred).ravel()
    #mean_diff = np.mean(np.abs(gt - pred))
    senzitivity=recall_score(gt,pred)
    precision=precision_score(gt,pred)
    f1=f1_score(gt,pred)
    fn_rate=fn/(fn + tp)
    pearson=pearsonr(gt,pred)
    jaccard = jaccard_score(gt,pred)
    #iou= iou_score(gt,pred)
    print ("Senzitivity:",senzitivity,"Precision:",precision,"F1:",f1,"FN_rate:",fn_rate,"Pearson:",pearson[0],"Jaccard:",jaccard)
    
        
    #print("FN rate: ", fn/(fn + tp))    


def adaugare_dice(predictii,adnotari_binare,csv_adnotari):
    
    
    dice=[]
    for index in range(len(csv_adnotari)):
        annotations_name=os.path.basename(csv_adnotari['annotation_path'][index])
        image_name=os.path.basename(csv_adnotari['image_path'][index])
        slice=csv_adnotari['slice_idx'][index]
    
        annotations_images=os.path.join(adnotari_binare,f"ADNOTARI_BINARE.{annotations_name}")
        prediction_images=os.path.join(predictii,f"PREDICTII_V2_{image_name}")
        
        annotations_path=os.path.join(annotations_images,f"Adnotare_binara_{slice}.png")
        prediction_path=os.path.join(prediction_images,f"PREDICTIE_{slice}.png")
        
        pred=cv.imread(prediction_path,cv.IMREAD_GRAYSCALE)
        gt=cv.imread(annotations_path,cv.IMREAD_GRAYSCALE)
        print(annotations_path,prediction_path)
        print (os.path.exists(prediction_path),os.path.exists(annotations_path))
        if os.path.exists(prediction_path)==True and os.path.exists(annotations_path)==True:
            if csv_adnotari['contur'][index]=='Closed':
                gt_bin = (gt / 255).astype(np.uint8)
                pred_bin = (pred / 255).astype(np.uint8)

                dice.append(dice_coef(gt_bin, pred_bin))
                output=r"D:\ai intro\OCT\OCT_REPO\Overlap"
                #metrics_calculation(gt_bin,pred_bin)
                cv.imwrite(os.path.join(output, 'OVERLAP'+'_'+str(image_name)+'_'+str(slice)+'.png'),overlap(gt,pred))
            else:
                dice.append(-1)
        else:
            dice.append(-1)            
        
        
        
        # slice_idx=os.path.basename(dir_ann).split(".")[1]


        # for ann_path in annotations:
        #    print (ann_path)
        #    image_slice=os.path.basename(ann_path).split("_")[3].split(".")[0]
        #    image_ext = os.path.basename(ann_path).split("_")[3].split(".")[1]
       
    csv_adnotari['dice_calculat_de_mine']= dice
    print(dice)
    
    df= pd.DataFrame(csv_adnotari)


    print(df.head())
    df.to_csv(r"D:\ai intro\OCT\OCT_file\DICE_ADDED_v2.csv", index=False)
    
def poz_negs_calculator(gt,prediction):
    
    contours_prediction = cv.findContours(prediction.astype(np.uint8), 0, method=1)
    contours_gt = cv.findContours(gt[count, 0,:, :, 0].astype(np.uint8), 0, method=1)

    cnt_predictions = list(filter(lambda x : len(x) > 5, contours_prediction[1]))
    contours_prediction = (contours_prediction[0], cnt_predictions, contours_prediction[2])
    cnt_gts = list(filter(lambda x : len(x) > 5, contours_gt[1]))
    contours_gt = (contours_gt[0], cnt_gts, contours_gt[2])

    mean_coord_pred = []
    ys_pred = []
    mean_coord_gt = []
    ys_gt = []
    prediction_contours = contours_prediction[1]
    gt_contours = contours_gt[1]

    gts = []
    indexes = []
    for index, cnt in enumerate(gt_contours):
        area = cv.contourArea(cnt)
        box = cv.minAreaRect(cnt)
        box = cv.cv.BoxPoints(box) if imutils.is_cv() else cv.boxPoints(box)
        box = np.array(box, dtype="int")

        box = perspective.order_points(box)

        cX = np.average(box[:, 0])
        cY = np.average(box[:, 1])

        (tl, tr, br, bl) = box
        (tlblX, tlblY) = ((tl[0] + bl[0]) // 2, (tl[1] + bl[1]) // 2)
        (trbrX, trbrY) = ((tr[0] + br[0]) // 2, (tr[1] + br[1]) // 2)

        D = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
        if area > 120:
            gts.append((box, (cX, cY), D, area))
        else:
            indexes.append(index)

    for count, index in enumerate(indexes):
        gt_contours.pop(index - count)

    preds = []
    indexes = []
    for index, cnt in enumerate(prediction_contours):
        area = cv.contourArea(cnt)
        box = cv.minAreaRect(cnt)
        box = cv.cv.BoxPoints(box) if imutils.is_cv2() else cv.boxPoints(box)
        box = np.array(box, dtype="int")

        box = perspective.order_points(box)

        cX = np.average(box[:, 0])
        cY = np.average(box[:, 1])

        (tl, tr, br, bl) = box
        (tlblX, tlblY) = ((tl[0] + bl[0]) // 2, (tl[1] + bl[1]) // 2)
        (trbrX, trbrY) = ((tr[0] + br[0]) // 2, (tr[1] + br[1]) // 2)

        D = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
        if area > 120:
            preds.append((box, (cX, cY), D, area))
        else:
            indexes.append(index)

    for count, index in enumerate(indexes):
        prediction_contours.pop(index - count)

    tp_coords_gt = []
    tp_coords_pred = []
    indexes = []
    for gt in gts:
        (box_gt, (cX_gt, cY_gt), D_gt, area_gt) = gt
        for index, pred in enumerate(preds):
            #if gt not in np.array(tp_coords_gt):
            (box_pred, (cX_pred, cY_pred), D_pred, area_pred) = pred
            diff_box_coord = np.abs(box_pred - box_gt)

            if pred[-1] > gt[-1]:
                min_x = box_pred[:,0].min()
                max_x = box_pred[:,0].max()
                min_y = box_pred[:,1].min()
                max_y = box_pred[:,1].max()

                range_x = np.arange(min_x - 40, max_x + 40)
                range_y = np.arange(min_y - 40, max_y + 40)

                ok = True
                for coord in box_gt:
                    if coord[0] not in range_x or coord[1] not in range_y:
                        ok = False

                if ok == True:
                    tp_coords_pred.append(pred)
                    if gt not in np.array(tp_coords_gt):
                        tp_coords_gt.append(gt)
                    indexes.append(index)
                    print("TP")
                    tp += 1

            elif gt[-1] > pred[-1]:
                min_x = box_gt[:,0].min()
                max_x = box_gt[:,0].max()
                min_y = box_gt[:,1].min()
                max_y = box_gt[:,1].max()

                range_x = np.arange(min_x - 40, max_x + 40)
                range_y = np.arange(min_y - 40, max_y + 40)

                ok = True
                for coord in box_pred:
                    if coord[0] not in range_x or coord[1] not in range_y:
                        ok = False

                if ok == True:
                    tp_coords_pred.append(pred)
                    if gt not in np.array(tp_coords_gt):
                        tp_coords_gt.append(gt)
                    indexes.append(index)
                    print("TP")
                    tp += 1

    if len(preds) - len(tp_coords_pred) > 0:
        fp += len(preds) - len(tp_coords_pred)
        print("FP " * (len(preds) - len(tp_coords_pred)))
    if len(gts) - len(tp_coords_gt) > 0:
        fn += len(gts) - len(tp_coords_gt)
        print("FN " * (len(gts) - len(tp_coords_gt)))
    return tp , fp ,fn 



    
    
                      
if __name__=='__main__':   
    jsons = glob.glob(r"D:\ai intro\OCT\Adnotari\*")
    csv_adnotari=pd.read_csv(r"D:\ai intro\OCT\OCT_REPO\DICE_ADDED.csv")
    predictii=(r"D:\ai intro\OCT\OCT_REPO\Predictii")
    adnotari_binare=(r"D:\ai intro\OCT\OCT_REPO\Imagini")
   
    # # # for j in jsons:
    # # #         print (j)
    # # #         with open(j) as f:
    # # #             date = json.load(f)
    # # #         print (date.keys())
    # # #         os.mkdir(f"D:\\ai intro\\OCT\\OCT_REPO\\Imagini\\ADNOTARI_BINARE.{os.path.basename(j)}")
    # # #         path_to_save=f"D:\\ai intro\\OCT\\OCT_REPO\\Imagini\\ADNOTARI_BINARE.{os.path.basename(j)}"
    # # #         transfom_into_binary(date,j,path_to_save)
      
    #read_dicom_without_annotations(r"E:\AchizitiiOctombrieUMF2021OCT\Patient001\IMG001")
    adaugare_dice(predictii,adnotari_binare,csv_adnotari)
    #test_on_dicom()
        
                                        
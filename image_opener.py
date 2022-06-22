from re import X
from tkinter import image_names
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import numpy as np
import yaml
import pydicom
import json
from pydicom import dcmread

from pydicom.dataset import Dataset
import pickle
import cv2 as cv 
from cv2 import contourArea
import csv
import glob
import os
import pandas as pd 

# with open('config.yml') as f: # reads .yml/.yaml files
#  config = yaml.safe_load(f)

def reg3simpla(x,y):
  z=x
  for i in range(y):
    z[i][0]=(x[i][0]*1024)/704
    z[i][1]=(x[i][1]*1024)/704
  
  return z
  
 

data_info={'image_path':[],'annotation_path':[],'label':[],'slice_idx':[],'n_total_slices':[],'img_size':[],'aria':[],'contur':[]}
jsons = glob.glob(r"D:\ai intro\OCT\Adnotari\*")
images=r"E:\AchizitiiOctombrieUMF2021OCT_2"
idx=0 
print (jsons, images)
for j in jsons:
  img_name = os.path.basename(j).split(".")[0]
  img_path = os.path.join(images,img_name) 

  with open(j) as f:
    data = json.load(f)
    print(data.keys())
  
  
  
  for k in data.keys():
   if k=='plaques':
      ds=dcmread(img_path)
      # img2d=ds.pixel_array
      img2d_shape = ds.pixel_array.shape
      print(len(data['plaques']))
      print(img2d_shape)
    
      
      nr=0
      for p in range(len(data['plaques'])):
       if (data['plaques'][p]['morphology']=='Calcium nodule'):
         n_slices = len(data['plaques'][p]["contours"]) 
         nr=nr+n_slices
         print("# slices", n_slices)
         for s in data['plaques'][p]["contours"]:
           data_info['image_path'].append(img_path)
           data_info['annotation_path'].append(os.path.normpath(j))
           data_info['img_size'].append(img2d_shape)
           print("slice", s)
           points = data['plaques'][p]["contours"][s]["control_pts"]
           print("points", points)
           pts = np.array(points, np.int32)
           print(pts.shape)
           puncte=pts
           pts=reg3simpla(pts,pts.shape[0])
           pts = pts.reshape((-1,1,2))
           print (pts)
           #print(puncte.shape)
           #print (puncte)
           sl = int(s)
           if nr != 0 :  
             data_info['label'].append("Calcium Nodule")
           else:  
             data_info['label'].append("NONE")
           data_info['n_total_slices'].append(nr)
           data_info['slice_idx'].append(s) 
           
      
           if data['plaques'][p]["contours"][s]["closed"]==True:
              data_info['aria'].append(cv.contourArea(pts))
              data_info['contur'].append('Closed')
           else: 
              data_info['aria'].append(cv.contourArea(pts))
              data_info['contur'].append('Open')
          #  cv.polylines(img2d[sl,:,:,:],[pts],data['plaques'][p]["contours"][s]["closed"],(0,255,255))
          #  cv.imshow(f"Slice {sl}", img2d[sl,:,:,2::-1])  
          #  cv.waitKey(0)
 
           
           
      


x=['268','269','271','538','539']
height=[0,0,0,0,0]
print (data_info['img_size'][3])
for info in range(len(data_info['img_size'])):
  if data_info['n_total_slices'][info] != 0:
   if data_info['img_size'][info][0]==268:
      height[0]=height[0]+1
      print (height)
   elif data_info['img_size'][info][0]==269:
      height[1]=height[1]+1
   elif data_info['img_size'][info][0]==271:
      height[2]=height[2]+1
   elif data_info['img_size'][info][0]==538:
      height[3]=height[3]+1
   elif data_info['img_size'][info][0]==539:
      height[4]=height[4]+1

# print (x,height)
plt.bar(x,height)
plt.xlabel("Number of slice from a image")
plt.ylabel("Counter of images ")
plt.savefig(r"D:\ai intro\OCT\OCT_REPOs\Barplot-Number_of_slices") 
# print(data_info)
# df= pd.DataFrame(data_info)


# print(df.head())
# df.to_csv(r"D:\ai intro\OCT\OCT_file\test.csv", index=False)

# with open(r"D:\ai intro\OCT\OCT_FIle\Statusuri.csv", mode='w') as oct_file:
#   oct_writer = csv.writer(oct_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#   for info in range(len(data_info['image_name'])):
#     oct_writer.writerow([data_info['image_name'][info],data_info['label'][info],data_info['n_total_slices'][info],data_info['img_size'][info]])
 
        




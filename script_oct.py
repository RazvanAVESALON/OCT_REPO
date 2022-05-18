import os
import cv2
import data
import network
import preprocessing
import GlobalSettings as GS
import tensorflow as tf 
import numpy as np
from time import time
import tensorflow as tf
import matplotlib.pyplot as plt
# from GAN.dcgan import DCGAN
# from GAN.pix2pix import Pix2Pix
#from GAN.cyclegan import CycleGAN
from  copy import deepcopy
from multiprocessing import Process
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score
import keras
import losses
import focal_tversky_unet as attention_unet

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
import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Conv2DTranspose, Cropping2D
from keras.layers.merge import add
from keras import backend as K
from keras.applications import vgg16
from scipy.io import loadmat

def FCN8s():
    inputData = Input(batch_shape=(None,224,224,3))
    #First Layer
    conv1_1 = Conv2D(64,kernel_size=(3,3),activation='relu',padding='same', name='conv1_1')(inputData)
    conv1_2 = Conv2D(64,kernel_size=(3,3),activation='relu',padding='same', name='conv1_2')(conv1_1)
    pool1 = MaxPooling2D((2,2), strides=(2,2), name='pool1')(conv1_2)
    #Second Convolution
    conv2_1 = Conv2D(128,kernel_size=(3,3),activation='relu',padding='same', name='conv2_1')(pool1)
    conv2_2 = Conv2D(128,kernel_size=(3,3),activation='relu',padding='same', name='conv2_2')(conv2_1)
    pool2 = MaxPooling2D((2,2), strides=(2,2), name='pool2')(conv2_2)
    #Third Convolution
    conv3_1 = Conv2D(256,kernel_size=(3,3),activation='relu',padding='same', name='conv3_1')(pool2)
    conv3_2 = Conv2D(256,kernel_size=(3,3),activation='relu',padding='same', name='conv3_2')(conv3_1)
    conv3_3 = Conv2D(256,kernel_size=(3,3),activation='relu',padding='same', name='conv3_3')(conv3_2)
    pool3 = MaxPooling2D((2,2), strides=(2,2), name='pool3')(conv3_3)
    #Fourth Convolution
    conv4_1 = Conv2D(512,kernel_size=(3,3),activation='relu',padding='same', name='conv4_1')(pool3)
    conv4_2 = Conv2D(512,kernel_size=(3,3),activation='relu',padding='same', name='conv4_2')(conv4_1)
    conv4_3 = Conv2D(512,kernel_size=(3,3),activation='relu',padding='same', name='conv4_3')(conv4_2)
    pool4 = MaxPooling2D((2,2), strides=(2,2), name='pool4')(conv4_3)
    #Fifth Convolution
    conv5_1 = Conv2D(512,kernel_size=(3,3),activation='relu',padding='same', name='conv5_1')(pool4)
    conv5_2 = Conv2D(512,kernel_size=(3,3),activation='relu',padding='same', name='conv5_2')(conv5_1)
    conv5_3 = Conv2D(512,kernel_size=(3,3),activation='relu',padding='same', name='conv5_3')(conv5_2)
    pool5 = MaxPooling2D((2,2), strides=(2,2), name='pool5')(conv5_3)
    #Fully Convolutional Layers 32s
    fc6 = Conv2D(4096,kernel_size=(7,7),activation='relu',padding='same', name='fc6')(pool5)
    drop6 = Dropout(0.5)(fc6)
    fc7 = Conv2D(4096,kernel_size=(1,1),activation='relu',padding='same', name='fc7')(drop6)
    drop7 = Dropout(0.5)(fc7)
    score_fr = Conv2D(21, kernel_size=(1,1), padding='valid', name='score_fr')(drop7)
    #Deconv Layer
    score2 = Conv2DTranspose(21, kernel_size=(4,4),strides=(2,2), name='score2')(score_fr)
    upscore2 = Cropping2D(cropping=1, name='crop2')(score2)
    #Merge with 4th Convolution
    score_pool4 = Conv2D(21, kernel_size=(1,1), padding='valid', name='score_pool4')(pool4)
    fuse = add([upscore2,score_pool4], name='fuse')
    score4 = Conv2DTranspose(21, kernel_size=(4,4),strides=(2,2), name='score4', use_bias=False)(fuse)
    upscore4 = Cropping2D(cropping=1, name='crop4')(score4)
    #Merge with 3rd Convolution
    score_pool3 = Conv2D(21, kernel_size=(1,1), padding='valid', name='score_pool3')(pool3)
    fusex = add([upscore4,score_pool3], name='fusex')
    #Score
    upscore16 = Conv2DTranspose(21, kernel_size=(16,16),strides=(8,8), name='upsample', use_bias=False)(fusex)
    score = Cropping2D(cropping=4, name='score')(upscore16)
    model = Model(inputs=inputData, outputs=[score])
    return model

def FCN16s():
    inputData = Input(batch_shape=(None,224,224,3))
    #First Layer
    conv1_1 = Conv2D(64,kernel_size=(3,3),activation='relu',padding='same', name='conv1_1')(inputData)
    conv1_2 = Conv2D(64,kernel_size=(3,3),activation='relu',padding='same', name='conv1_2')(conv1_1)
    pool1 = MaxPooling2D((2,2), strides=(2,2), name='pool1')(conv1_2)
    #Second Convolution
    conv2_1 = Conv2D(128,kernel_size=(3,3),activation='relu',padding='same', name='conv2_1')(pool1)
    conv2_2 = Conv2D(128,kernel_size=(3,3),activation='relu',padding='same', name='conv2_2')(conv2_1)
    pool2 = MaxPooling2D((2,2), strides=(2,2), name='pool2')(conv2_2)
    #Third Convolution
    conv3_1 = Conv2D(256,kernel_size=(3,3),activation='relu',padding='same', name='conv3_1')(pool2)
    conv3_2 = Conv2D(256,kernel_size=(3,3),activation='relu',padding='same', name='conv3_2')(conv3_1)
    conv3_3 = Conv2D(256,kernel_size=(3,3),activation='relu',padding='same', name='conv3_3')(conv3_2)
    pool3 = MaxPooling2D((2,2), strides=(2,2), name='pool3')(conv3_3)
    #Fourth Convolution
    conv4_1 = Conv2D(512,kernel_size=(3,3),activation='relu',padding='same', name='conv4_1')(pool3)
    conv4_2 = Conv2D(512,kernel_size=(3,3),activation='relu',padding='same', name='conv4_2')(conv4_1)
    conv4_3 = Conv2D(512,kernel_size=(3,3),activation='relu',padding='same', name='conv4_3')(conv4_2)
    pool4 = MaxPooling2D((2,2), strides=(2,2), name='pool4')(conv4_3)
    #Fifth Convolution
    conv5_1 = Conv2D(512,kernel_size=(3,3),activation='relu',padding='same', name='conv5_1')(pool4)
    conv5_2 = Conv2D(512,kernel_size=(3,3),activation='relu',padding='same', name='conv5_2')(conv5_1)
    conv5_3 = Conv2D(512,kernel_size=(3,3),activation='relu',padding='same', name='conv5_3')(conv5_2)
    pool5 = MaxPooling2D((2,2), strides=(2,2), name='pool5')(conv5_3)
    #Fully Convolutional Layers 32s
    fc6 = Conv2D(4096,kernel_size=(7,7),activation='relu',padding='same', name='fc6')(pool5)
    drop6 = Dropout(0.5)(fc6)
    fc7 = Conv2D(4096,kernel_size=(1,1),activation='relu',padding='same', name='fc7')(drop6)
    drop7 = Dropout(0.5)(fc7)
    score_fr = Conv2D(21, kernel_size=(1,1), padding='valid', name='score_fr')(drop7)
    #Deconv Layer
    score2 = Conv2DTranspose(21, kernel_size=(4,4),strides=(2,2), name='score2')(score_fr)
    upscore2 = Cropping2D(cropping=1, name='crop')(score2)
    #Merge with 4th Convolution
    score_pool4 = Conv2D(21, kernel_size=(1,1), padding='valid', name='score_pool4')(pool4)
    fuse = add([upscore2,score_pool4], name='fuse')
    #Score
    upscore16 = Conv2DTranspose(21, kernel_size=(32,32),strides=(16,16), name='upsample_new')(fuse)
    score = Cropping2D(cropping=8, name='score')(upscore16)
    model = Model(inputs=inputData, outputs=[score])
    return model

def FCN32s():
    inputData = Input(batch_shape=(None,224,224,3))
    #First Layer
    conv1_1 = Conv2D(64,kernel_size=(3,3),activation='relu',padding='same', name='conv1_1')(inputData)
    conv1_2 = Conv2D(64,kernel_size=(3,3),activation='relu',padding='same', name='conv1_2')(conv1_1)
    pool1 = MaxPooling2D((2,2), strides=(2,2), name='pool1')(conv1_2)
    #Second Convolution
    conv2_1 = Conv2D(128,kernel_size=(3,3),activation='relu',padding='same', name='conv2_1')(pool1)
    conv2_2 = Conv2D(128,kernel_size=(3,3),activation='relu',padding='same', name='conv2_2')(conv2_1)
    pool2 = MaxPooling2D((2,2), strides=(2,2), name='pool2')(conv2_2)
    #Third Convolution
    conv3_1 = Conv2D(256,kernel_size=(3,3),activation='relu',padding='same', name='conv3_1')(pool2)
    conv3_2 = Conv2D(256,kernel_size=(3,3),activation='relu',padding='same', name='conv3_2')(conv3_1)
    conv3_3 = Conv2D(256,kernel_size=(3,3),activation='relu',padding='same', name='conv3_3')(conv3_2)
    pool3 = MaxPooling2D((2,2), strides=(2,2), name='pool3')(conv3_3)
    #Fourth Convolution
    conv4_1 = Conv2D(512,kernel_size=(3,3),activation='relu',padding='same', name='conv4_1')(pool3)
    conv4_2 = Conv2D(512,kernel_size=(3,3),activation='relu',padding='same', name='conv4_2')(conv4_1)
    conv4_3 = Conv2D(512,kernel_size=(3,3),activation='relu',padding='same', name='conv4_3')(conv4_2)
    pool4 = MaxPooling2D((2,2), strides=(2,2), name='pool4')(conv4_3)
    #Fifth Convolution
    conv5_1 = Conv2D(512,kernel_size=(3,3),activation='relu',padding='same', name='conv5_1')(pool4)
    conv5_2 = Conv2D(512,kernel_size=(3,3),activation='relu',padding='same', name='conv5_2')(conv5_1)
    conv5_3 = Conv2D(512,kernel_size=(3,3),activation='relu',padding='same', name='conv5_3')(conv5_2)
    pool5 = MaxPooling2D((2,2), strides=(2,2), name='pool5')(conv5_3)
    #Fully Convolutional Layers 32s
    fc6 = Conv2D(4096,kernel_size=(7,7),activation='relu',padding='same', name='fc6')(pool5)
    drop6 = Dropout(0.5)(fc6)
    fc7 = Conv2D(4096,kernel_size=(1,1),activation='relu',padding='same', name='fc7')(drop6)
    drop7 = Dropout(0.5)(fc7)
    score_fr = Conv2D(21, kernel_size=(1,1), padding='valid', name='score_fr')(drop7)
    #Deconv Layer
    bil = Conv2DTranspose(21, kernel_size=(64,64),strides=(32,32), name='upsample')(score_fr)
    crop = Cropping2D(cropping=16)(bil)
    model = Model(inputs=inputData, outputs=[crop])
    return model

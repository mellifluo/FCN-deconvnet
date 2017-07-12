import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Conv2DTranspose, Cropping2D
from keras import backend as K

#doesn't work
def FCN(stop=False, input_shape=None):
    inputData = Input(batch_shape=(None,512,512,3))
    #First Layer
    conv1_1 = Conv2D(64,kernel_size=(3,3),activation='relu',padding='same', name='block1_conv1')(inputData)
    conv1_2 = Conv2D(64,kernel_size=(3,3),activation='relu',padding='same', name='block1_conv2')(conv1_1)
    pool1 = MaxPooling2D((2,2), strides=(2,2), name='block1_pool')(conv1_2)
    #Second Convolution
    conv2_1 = Conv2D(128,kernel_size=(3,3),activation='relu',padding='same', name='block2_conv1')(pool1)
    conv2_2 = Conv2D(128,kernel_size=(3,3),activation='relu',padding='same', name='block2_conv2')(conv2_1)
    pool2 = MaxPooling2D((2,2), strides=(2,2), name='block2_pool')(conv2_2)
    #Third Convolution
    conv3_1 = Conv2D(256,kernel_size=(3,3),activation='relu',padding='same', name='block3_conv1')(pool2)
    conv3_2 = Conv2D(256,kernel_size=(3,3),activation='relu',padding='same', name='block3_conv2')(conv3_1)
    conv3_3 = Conv2D(256,kernel_size=(3,3),activation='relu',padding='same', name='block3_conv3')(conv3_2)
    pool3 = MaxPooling2D((2,2), strides=(2,2), name='block3_pool')(conv3_3)
    #Fourth Convolution
    conv4_1 = Conv2D(512,kernel_size=(3,3),activation='relu',padding='same', name='block4_conv1')(pool3)
    conv4_2 = Conv2D(512,kernel_size=(3,3),activation='relu',padding='same', name='block4_conv2')(conv4_1)
    conv4_3 = Conv2D(512,kernel_size=(3,3),activation='relu',padding='same', name='block4_conv3')(conv4_2)
    pool4 = MaxPooling2D((2,2), strides=(2,2), name='block4_pool')(conv4_3)
    #Fifth Convolution
    conv5_1 = Conv2D(512,kernel_size=(3,3),activation='relu',padding='same', name='block5_conv1')(pool4)
    conv5_2 = Conv2D(512,kernel_size=(3,3),activation='relu',padding='same', name='block5_conv2')(conv5_1)
    conv5_3 = Conv2D(512,kernel_size=(3,3),activation='relu',padding='same', name='block5_conv3')(conv5_2)
    pool5 = MaxPooling2D((2,2), strides=(2,2), name='block5_pool')(conv5_3)
    #Fully Convolutional Layers 32s
    fc6 = Conv2D(4096,kernel_size=(7,7),activation='relu',padding='same', name='fc1')(pool5)
    drop6 = Dropout(0.5)(fc6)
    fc7 = Conv2D(4096,kernel_size=(1,1),activation='relu',padding='same', name='fc2')(drop6)
    drop7 = Dropout(0.5)(fc7)
    #score_fr = Conv2D(21,kernel_size=(1,1),padding='valid')(drop7)
    score_fr = Conv2D(1000, kernel_size=(1,1), padding='valid', name='predictions')(drop7)
    #Deconv Layer
    #flat = Flatten()(upscore)
    #pred32 = Dense(21,activation='softmax')(flat)
    #upscore = Conv2DTranspose(3,kernel_size=(64,64),strides=(32,32),use_bias=False)(score_fr)

    #bil = Conv2DTranspose(1000, kernel_size=(64,64),strides=(32,32), use_bias=False)(score_fr)
    #crop = Cropping2D(cropping=16)(upscore)
    #bilinear_lambda = Lambda(bilfunction, output_shape=(224,224,3))(score_fr)
    #bu = Lambda(lambda x:theano.tensor.nnet.abstract_conv.bilinear_upsampling(x,32), output_shape=(224,224,3))(score_fr)
    model = Model(inputs=inputData, outputs=[score_fr])
    model.load_weights('/home/afagnani/keras-deconvnet/fcn.h5', by_name=True)
    #bilW = bilinear_upsample_weights(32,1000)
    #model.layers[-1].set_weights([bilW])
    return model

def bilfunction(x):
    x = K.permute_dimensions(x, [0, 3, 1, 2])
    x = theano.tensor.nnet.abstract_conv.bilinear_upsampling(x,32)
    x = K.permute_dimensions(x, [0, 2, 3, 1])
    return x

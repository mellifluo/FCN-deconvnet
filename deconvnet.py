import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Conv2DTranspose, Cropping2D
from keras import backend as K

#an example of Unpooling usign a mask
#based on: https://stackoverflow.com/questions/43657825/create-an-unpooling-mask-from-output-layers-in-keras
def Mask(x, strides=(2,2)):
    orig = x #Save output x
    x = MaxPooling2D(strides=strides)(x)
    x = UpSampling2D()(x)
    img_h = orig._keras_shape[1]
    img_w = orig._keras_shape[2]
    num_channels = orig._keras_shape[3]
    #here we're going to reshape the data for a concatenation:
    #xReshaped and origReshaped are now split branches
    xReshaped = Reshape((1, img_h, img_w ,num_channels))(x)
    origReshaped = Reshape((1, img_h, img_w ,num_channels))(orig)
    #concatenation - here, you unite both branches again
        #normally you don't need to reshape or use the axis var,
        #but here we want to keep track of what was x and what was orig.
    together = Concatenate(axis=1)([origReshaped,xReshaped])
    bool_mask = Lambda(lambda t: K.greater_equal(t[:,0], t[:,1]),
        output_shape=(img_h, img_w, num_channels))(together)
    mask = Lambda(lambda t: K.cast(t, dtype='float32'),
        output_shape=(img_h, img_w, num_channels))(bool_mask)
    return mask

def Unpooling2D(mask,x):
    x = UpSampling2D()(x)
    return Multiply()([mask, x])


def deconvnet():
    inputData = Input(batch_shape=(None,224,224,3))
    #First Layer
    conv1_1 = Conv2D(64,kernel_size=(3,3),activation='relu',padding='same', name='block1_conv1')(inputData)
    conv1_2 = Conv2D(64,kernel_size=(3,3),activation='relu',padding='same', name='block1_conv2')(conv1_1)
    pool1 = MaxPooling2D((2,2), strides=(2,2), name='block1_pool')(conv1_2)
    #224x224
    conv2_1 = Conv2D(128,kernel_size=(3,3),activation='relu',padding='same', name='block2_conv1')(pool1)
    conv2_2 = Conv2D(128,kernel_size=(3,3),activation='relu',padding='same', name='block2_conv2')(conv2_1)
    pool2 = MaxPooling2D((2,2), strides=(2,2), name='block2_pool')(conv2_2)
    #112x112
    conv3_1 = Conv2D(256,kernel_size=(3,3),activation='relu',padding='same', name='block3_conv1')(pool2)
    conv3_2 = Conv2D(256,kernel_size=(3,3),activation='relu',padding='same', name='block3_conv2')(conv3_1)
    conv3_3 = Conv2D(256,kernel_size=(3,3),activation='relu',padding='same', name='block3_conv3')(conv3_2)
    pool3 = MaxPooling2D((2,2), strides=(2,2), name='block3_pool')(conv3_3)
    #56x56
    conv4_1 = Conv2D(512,kernel_size=(3,3),activation='relu',padding='same', name='block4_conv1')(pool3)
    conv4_2 = Conv2D(512,kernel_size=(3,3),activation='relu',padding='same', name='block4_conv2')(conv4_1)
    conv4_3 = Conv2D(512,kernel_size=(3,3),activation='relu',padding='same', name='block4_conv3')(conv4_2)
    pool4 = MaxPooling2D((2,2), strides=(2,2), name='block4_pool')(conv4_3)
    #14x14
    conv5_1 = Conv2D(512,kernel_size=(3,3),activation='relu',padding='same', name='block5_conv1')(pool4)
    conv5_2 = Conv2D(512,kernel_size=(3,3),activation='relu',padding='same', name='block5_conv2')(conv5_1)
    conv5_3 = Conv2D(512,kernel_size=(3,3),activation='relu',padding='same', name='block5_conv3')(conv5_2)
    pool5 = MaxPooling2D((2,2), strides=(2,2), name='block5_pool')(conv5_3)
    #7x7
    fc6 = Conv2D(4096,kernel_size=(7,7),activation='relu')(pool5)
    #1x1
    fc7 = Conv2D(4096,kernel_size=(1,1),activation='relu')(fc6)

    fc6_deconv = Conv2DTranspose(512,kernel_size=(7,7),activation='relu')(fc7)
    #7x7
    m5 = Mask(conv5_3)
    unpool5 = Unpooling2D(m5,fc6_deconv)
    #14x14
    deconv5_1 = Conv2DTranspose(512,kernel_size=(3,3),padding='same',activation='relu', name='Xblock5_conv3')(unpool5)
    deconv5_2 = Conv2DTranspose(512,kernel_size=(3,3),padding='same',activation='relu', name='Xblock5_conv2')(deconv5_1)
    deconv5_3 = Conv2DTranspose(512,kernel_size=(3,3),padding='same',activation='relu', name='Xblock5_conv1')(deconv5_2)
    m4 = Mask(conv4_3)
    unpool4 = Unpooling2D(m4,deconv5_3)
    #28x28
    deconv4_1 = Conv2DTranspose(512,kernel_size=(3,3),padding='same',activation='relu', name='Xblock4_conv3')(unpool4)
    deconv4_2 = Conv2DTranspose(512,kernel_size=(3,3),padding='same',activation='relu', name='Xblock4_conv2')(deconv4_1)
    deconv4_3 = Conv2DTranspose(256,kernel_size=(3,3),padding='same',activation='relu', name='Xblock4_conv1')(deconv4_2)
    m3 = Mask(conv3_3)
    unpool3 = Unpooling2D(m3,deconv4_3)
    #56x56
    deconv3_1 = Conv2DTranspose(256,kernel_size=(3,3),padding='same',activation='relu', name='Xblock3_conv3')(unpool3)
    deconv3_2 = Conv2DTranspose(256,kernel_size=(3,3),padding='same',activation='relu', name='Xblock3_conv2')(deconv3_1)
    deconv3_3 = Conv2DTranspose(128,kernel_size=(3,3),padding='same',activation='relu', name='Xblock3_conv1')(deconv3_2)
    m2 = Mask(conv2_2)
    unpool2 = Unpooling2D(m2,deconv3_3)
    #112x112
    deconv2_1 = Conv2DTranspose(128,kernel_size=(3,3),padding='same',activation='relu', name='Xblock2_conv2')(unpool2)
    deconv2_2 = Conv2DTranspose(64,kernel_size=(3,3),padding='same',activation='relu', name='Xblock2_conv1')(deconv2_1)
    m1 = Mask(conv1_2)
    unpool1 = Unpooling2D(m1,deconv2_2)
    #224x224
    deconv1_1 = Conv2DTranspose(64,kernel_size=(3,3),padding='same',activation='relu', name='Xblock1_conv2')(unpool1)
    deconv1_2 = Conv2DTranspose(64,kernel_size=(3,3),padding='same',activation='relu', name='Xblock1_conv1')(deconv1_1)
    score_fr = Conv2D(3,kernel_size=(1,1),padding='same')(deconv1_2)
    #pred32 = Dense(21,activation='softmax')(score_fr)
    modelD = Model(inputs=inputData, outputs=[score_fr])
    modelD.load_weights('/home/afagnani/keras-deconvnet/fcn.h5', by_name=True)
    modelD.load_weights('deconv_weights.h5', by_name=True)

    return modelD

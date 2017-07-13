import keras
from keras.layers import MaxPooling2D, UpSampling2D, Input
from keras.layers import Multiply, Lambda, Concatenate, Reshape, Lambda
from PIL import Image
import keras.backend as K
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from keras.preprocessing import image
from imagenet_utils import preprocess_input, decode_predictions
from scipy.io import loadmat


#2 simple functions to visualize an image predicted by the model
#https://github.com/erikreppel/visualizing_cnns/blob/master/visualize_cnns.ipynb
def visualize_image(model, myimage, dim=None, compare=0, figure=None):
    myimage_batch = np.expand_dims(myimage,axis=0)
    conv_myimage = model.predict(myimage_batch)
    conv_myimage = np.squeeze(conv_myimage, axis=0)
    if dim:
        conv_myimage = sp.misc.imresize(conv_myimage, dim)
    print conv_myimage.shape
    if compare == 1:
        two_images(myimage, conv_myimage, figure)
    if compare == 0:
        plt.imshow(conv_myimage)
    return conv_myimage

def keras_predict(model, img):
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)
    return model.predict(x)

def keras_visualize(img):
    img = np.squeeze(img,axis=0)
    plt.imshow(img)
    return img

def keras_open_image(img_path, target_size=(224,224)):
    #rgbImage = Image.open(myimage).convert('RGB')
    #ret_image = np.array(rgbImage)
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    return x

def open_image(myimage, resize=None):
    rgbImage = Image.open(myimage).convert('RGB')
    if resize != None:
        rgbImage = sp.misc.imresize(rgbImage, resize)
    return np.array(rgbImage)

#look at differences between two carved images
def same_output(model1, model2, image):
    figure = plt.figure()
    figure.add_subplot(1,2,1)
    im1 = visualize_image(model1, image)
    figure.add_subplot(1,2,2)
    im2 = visualize_image(model2, image)
    return np.array_equal(im1,im2)

def two_images(image1, image2, figure=None):
    if figure == None:
        figure = plt.figure()
    figure.add_subplot(1,2,1)
    plt.imshow(image1)
    figure.add_subplot(1,2,2)
    plt.imshow(image2)

def res_visualize_image(model, myimage, dim):
    myimage_batch = np.expand_dims(myimage,axis=0)
    conv_myimage = model.predict(myimage_batch)
    conv_myimage = np.squeeze(conv_myimage, axis=0)
    print conv_myimage.shape
    plt.imshow(conv_myimage)
    return conv_myimage

# Function to nicely print segmentation results with
# colorbar showing class names
def discrete_matshow(data, labels_names=[], title=""):

    fig_size = [7, 6]
    plt.rcParams["figure.figsize"] = fig_size

    #get discrete colormap
    cmap = plt.get_cmap('Paired', np.max(data)-np.min(data)+1)

    # set limits .5 outside true range
    mat = plt.imshow(data,
                      cmap=cmap,
                      vmin = np.min(data)-.5,
                      vmax = np.max(data)+.5)
    #tell the colorbar to tick at integers
    cax = plt.colorbar(mat,
                       ticks=np.arange(np.min(data),np.max(data)+1))

    # The names to be printed aside the colorbar
    if labels_names:
        cax.ax.set_yticklabels(labels_names)

    if title:
        plt.suptitle(title, fontsize=15, fontweight='bold')
def get_kernel_size(factor):
    """
    Find the kernel size given the desired factor of upsampling.
    """
    return 2 * factor - factor % 2


def upsample_filt(size):
    """
    Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
    """
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)


def bilinear_upsample_weights(factor, number_of_classes):
    """
    Create weights matrix for transposed convolution with bilinear filter
    initialization.
    """

    filter_size = get_kernel_size(factor)

    weights = np.zeros((filter_size,
                        filter_size,
                        number_of_classes,
                        number_of_classes), dtype=np.float32)

    upsample_kernel = upsample_filt(filter_size)

    for i in xrange(number_of_classes):
        weights[:, :, i, i] = upsample_kernel
    return weights

def copy_mat_to_keras(kmodel):
    kerasnames = [lr.name for lr in kmodel.layers]
    for i in range(0, p.shape[1]-1, 2):
        matname = '_'.join(p[0,i].name[0].split('_')[0:-1])
        if matname in kerasnames:
            kindex = kerasnames.index(matname)
            print 'found : ', (str(matname), kindex)
            l_weights = p[0,i].value
            l_bias = p[0,i+1].value
            kmodel.layers[kindex].set_weights([l_weights, l_bias[:,0]])
        else:
            print 'not found : ', str(matname)


def prepareim(im):
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    im = np.expand_dims(im, axis=0)
    return im

def predvgg():
    im = keras_open_image('cat.png')
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    im = np.expand_dims(im, axis=0)

    # Test pretrained model
    model = vgg()
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    out = model.predict(im)
    return out

"""
utili per il dataset
        data = loadmat('pascal-fcn32s-dag.mat', matlab_compatible=False, struct_as_record=False)
        l = data['layers']
        p = data['params']
        description = data['meta'][0,0].classes[0,0].description
"""

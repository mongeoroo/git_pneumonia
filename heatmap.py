import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import cv2
from Utils import min_max_normalization, label
from keras.preprocessing.image import ImageDataGenerator
from scipy.ndimage import gaussian_filter
from model import model_1
from model2 import model_2
import tensorflow as tf

def custom_cam(model,X_data):
    class_output = model.output[0, 0]

    conv = model.layers[-3].output
    iterate = K.function([model.input], [conv,class_output])

    conv_output, output = iterate([X_data])
    fc = model.get_layer('fc3').get_weights()[0]

    conv_output = np.squeeze(conv_output,axis=0)

    for i in range(conv_output.shape[-1]):
        if i == 0:
            cam = conv_output[:,:,i]*fc[i,0]
        else:
            cam += conv_output[:, :, i] * fc[i, 0]

    cam = (cam > 0) * cam
    cam = cv2.resize(cam, (224, 224))

    return fc, cam, output

def custom_grad_cam(model,X_data):

    class_output = model.output[0,0]

    conv = model.layers[-4].output

    grads = K.gradients(class_output, conv)[0]

    iterate = K.function([model.input], [grads,conv,class_output])

    grad_value, conv_output, output = iterate([X_data])

    grad_value = np.squeeze(grad_value, axis=0) ## (14,14,512)

    grad_value = np.average(grad_value, axis=(0,1))## (512,)

    conv_output = np.squeeze(conv_output,axis=0)

    for i,alpha in enumerate(grad_value):

        if i == 0:
            grad_cam = alpha*conv_output[:,:,i]
        else:
            grad_cam += alpha*conv_output[:,:,i]

    grad_cam = (grad_cam>0)*grad_cam
    grad_cam = cv2.resize(grad_cam, (224, 224))

    return grad_cam, grad_value, output

def saliencymap(model,X_data):
    class_output = model.output[0, 0]

    grads = K.gradients(class_output, model.input)[0]

    iterate = K.function([model.input], [grads,class_output])

    saliency, output = iterate([X_data])

    saliency = np.max(np.abs(saliency[0]),axis=-1)

    return saliency, output

def show_heatmap(image_path):
    image = load_img(image_path, target_size=(224, 224))
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    test_datagen = ImageDataGenerator(rescale=1.0/255, samplewise_center=True,samplewise_std_normalization=True)
    gen = test_datagen.flow(image_array,batch_size=1)
    X_data = gen.next()

    plt.subplot(1, 4, 1)
    plt.imshow(min_max_normalization(X_data[0]),cmap='gray')
    plt.title('Original image')
    plt.axis('off')

    model = model_1
    model.load_weights('./weights_model1.h5')

    '''cam'''
    fc, cam, output = custom_cam(model,X_data)
    plt.subplot(1, 4, 2)
    plt.imshow(min_max_normalization(cam), cmap='jet')
    plt.imshow(min_max_normalization(X_data[0]),cmap='gray',alpha=.6)
    plt.title('CAM: {}'.format(label[int(round(output))]))
    plt.axis('off')

    model = model_2
    model.load_weights('./weights_model2.h5')

    '''grad cam'''
    grad_cam, grad_value, output = custom_grad_cam(model,X_data)
    plt.subplot(1, 4, 3)
    plt.imshow(min_max_normalization(grad_cam), cmap='jet')
    plt.imshow(min_max_normalization(X_data[0]),cmap='gray',alpha=.6)
    plt.title('Grad-CAM: {}'.format(label[int(round(output))]))
    plt.axis('off')


    '''saliency map'''
    saliency, output = saliencymap(model,X_data)
    plt.subplot(1, 4, 4)
    plt.imshow(min_max_normalization(gaussian_filter(saliency,sigma=1)), cmap='jet')
    plt.imshow(min_max_normalization(X_data[0]),cmap='gray',alpha=.5)
    plt.title('Saliency Map: {}'.format(label[int(round(output))]))
    plt.axis('off')
    plt.show()

if __name__=='__main__':
    with tf.device('/cpu:0'):
        image_path = ".\childhood-pneumonia-1.jpg"
        show_heatmap(image_path)

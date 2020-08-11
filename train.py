from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import numpy as np
from train_model import model_1
from train_model2 import model_2
from tensorflow import keras
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

np.random.seed(55)


train_path = 'C:\mong\Korea radiology\pneumonia/chest_xray/train'
test_path = 'C:\mong\Korea radiology\pneumonia/chest_xray/test'

model = model_1
#model = model_2
otim = Adam(lr=0.0001, beta_1=0.9,beta_2=0.999,epsilon=1e-07)
model.compile(optimizer = otim, loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary(line_length=120)


train_datagen = ImageDataGenerator(rescale=1.0/255,
                                   samplewise_center=True,
                                   samplewise_std_normalization=True,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,fill_mode='constant',cval=0
                                   )

test_datagen = ImageDataGenerator(rescale=1.0/255,
                                   samplewise_center=True,
                                   samplewise_std_normalization=True
                                    )

batch_size = 8


print('generator flow')
training_set = train_datagen.flow_from_directory(train_path,
                                                 target_size = (224, 224),
                                                 batch_size = batch_size,
                                                 class_mode='binary',seed=44,shuffle=True
                                                 )

test_set = test_datagen.flow_from_directory(test_path,
                                            target_size = (224, 224),
                                            batch_size = batch_size,
                                            class_mode='binary',seed=44,shuffle=False
                                            )

print('train start')
cp_callback = keras.callbacks.ModelCheckpoint('./weights.h5', save_weights_only=True, verbose=1, save_best_only=True)
hist = model.fit_generator(training_set,steps_per_epoch=5219//batch_size,epochs=300,validation_data=test_set,callbacks=[cp_callback],
                           validation_steps=624//batch_size,verbose=1)

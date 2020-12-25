from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
#from keras.datasets import cifar10
import numpy as np
import os



data_augmentation = True
num_classes = 10
depth=8


subtract_pixel_mean = True

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
input_shape = x_train.shape[1:]
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

x_train_mean = np.mean(x_train, axis=0)
x_train -= x_train_mean
x_test -= x_train_mean


y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


def lr_schedule(epoch):

    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def resnet_layer(inputs, num_filters=16,kernel_size=3, strides=1,
                 activation='relu', batch_normalization=True,conv_first=True):

    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x



num_filters = 16
num_res_blocks = 1
inputs = Input(shape=input_shape)
x = resnet_layer(inputs=inputs)
for stack in range(3):
    for res_block in range(num_res_blocks):
        strides = 1
        if stack > 0 and res_block == 0:
            strides = 2
        y = resnet_layer(inputs=x, num_filters=num_filters, strides=strides)
        y = resnet_layer(inputs=y,num_filters=num_filters,activation=None)
        if stack > 0 and res_block == 0:
                x = resnet_layer(inputs=x, num_filters=num_filters,kernel_size=1,
                                 strides=strides, activation=None, batch_normalization=False)
        x = keras.layers.add([x, y])
        x = Activation('relu')(x)
    num_filters *= 2

x = AveragePooling2D(pool_size=8)(x)
y = Flatten()(x)
outputs = Dense(num_classes, activation='softmax',
                    kernel_initializer='he_normal')(y)

model = Model(inputs=inputs, outputs=outputs)
model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])
model.summary()

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'cifar10.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)
checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_acc',  verbose=1,save_best_only=True)
lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),cooldown=0, patience=5,min_lr=0.5e-6)

callbacks = [checkpoint, lr_reducer, lr_scheduler]
datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        zca_epsilon=1e-06,
        rotation_range=0,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.,
        zoom_range=0.,
        channel_shift_range=0.,
        fill_mode='nearest',
        cval=0.,
        horizontal_flip=True,
        vertical_flip=False,
        rescale=None,
        preprocessing_function=None,
        data_format=None,
        validation_split=0.10)

datagen.fit(x_train)
model.fit_generator(datagen.flow(x_train, y_train, batch_size=100),validation_data=(x_test, y_test),
                        epochs=10, verbose=1, workers=4,callbacks=callbacks)

scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss = ', scores[0])
print('Test accuracy = ', scores[1])




import os
import tensorflow as tf
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras.backend as K
from keras.models import Model
from keras.layers import (multiply, Conv2D, Input, MaxPooling2D, Conv2DTranspose,
        Concatenate, Dropout, UpSampling2D, BatchNormalization, Cropping2D, Lambda, Activation)
from keras.regularizers import l2
from keras.activations import relu

def ConvBlock(x, filters=64, expanding_path=False):

    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same',
            kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Activation(relu)(x)
    if expanding_path:
        x = Conv2D(filters=filters // 2, kernel_size=3, strides=1, padding='same',
            kernel_regularizer=l2(0.01))(x)
    else:
        x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same',
            kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    return Activation(relu)(x)

_epsilon = tf.convert_to_tensor(K.epsilon(), tf.float32)

def unet(input_shape, initial_exp=6, n_classes=2):
     
    features = Input(shape=input_shape)
    _power = initial_exp
    exp = 2

    c1 = ConvBlock(features, exp**_power)
    mp1 = MaxPooling2D(pool_size=2, strides=2)(c1)

    _power += 1

    c2 = ConvBlock(mp1, exp**_power)
    mp2 = MaxPooling2D(pool_size=2, strides=2)(c2)

    _power += 1

    c3 = ConvBlock(mp2, exp**_power)
    mp3 = MaxPooling2D(pool_size=2, strides=2)(c3)

    _power += 1 

    c4 = ConvBlock(mp3, exp**_power)
    mp4 = MaxPooling2D(pool_size=2, strides=2)(c4)

    _power += 1

    # 1024 filters
    c5 = Conv2D(filters=exp**_power, kernel_size=3, strides=1, padding='same',
            kernel_regularizer=l2(0.01))(mp4)
    _power -= 1
    c5 = Conv2D(filters=exp**_power, kernel_size=3, strides=1, padding='same',
            kernel_regularizer=l2(0.01))(c5)

    u1 = UpSampling2D(size=(2, 2))(c5)

    u1_c4 = Concatenate()([u1, c4])

    c6 = ConvBlock(u1_c4, filters=exp**_power, expanding_path=True)

    u2 = UpSampling2D(size=(2, 2))(c6)

    u2_c3 = Concatenate()([u2, c3])

    _power -= 1 
    c7 = ConvBlock(u2_c3, filters=exp**_power, expanding_path=True)

    u3 = UpSampling2D(size=(2, 2))(c7)

    u3_c2 = Concatenate()([u3, c2])

    _power -= 1 
    c8 = ConvBlock(u3_c2, filters=exp**_power, expanding_path=True)

    u4 = UpSampling2D(size=(2, 2))(c8)

    u4_c1 = Concatenate()([u4, c1])

    _power -= 1 
    c9 = ConvBlock(u4_c1, filters=exp**_power)
    last_conv = Conv2D(filters=1, kernel_size=1, padding='same', activation='sigmoid')(c9)
    return Model(inputs=[features], outputs=[last_conv])



if __name__ == '__main__':
    import keras.preprocessing as kp

    data_gen_args = dict( rotation_range=30, width_shift_range=0.1, height_shift_range=0.1,
    zoom_range=0.3,horizontal_flip=True)
    image_datagen = kp.image.ImageDataGenerator(preprocessing_function=lambda x: x/255.,**data_gen_args)
    mask_datagen = kp.image.ImageDataGenerator(**data_gen_args)

    target_size = (240, 320)
    batch_size = 8

    image_generator = image_datagen.flow_from_directory(
        './imgs_train',
        class_mode=None,
        seed=0,target_size=target_size,batch_size=batch_size)

    mask_generator = mask_datagen.flow_from_directory(
        './masks_train',
        class_mode=None,
        seed=0,target_size=target_size,color_mode='grayscale',batch_size=batch_size)

    image_generator_test = image_datagen.flow_from_directory(
        './imgs_test',
        class_mode=None,
        seed=0,target_size=target_size,batch_size=batch_size)

    mask_generator_test = mask_datagen.flow_from_directory(
        './masks_test',
        class_mode=None,
        seed=0,target_size=target_size,color_mode='grayscale',batch_size=batch_size)

    train_generator = zip(image_generator, mask_generator)
    test_generator = zip(image_generator_test,mask_generator_test)

    model = unet(input_shape=(240, 320, 3))

    import keras.optimizers as ko
    model.compile(optimizer = ko.Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics =
        ['accuracy'])
    model.summary()

    import keras.callbacks as kc
    filepath = './checkpoints_2.h5'

    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = kc.ModelCheckpoint(filepath=filepath,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True)

    def lr_schedule(epoch):
        lr = 1e-4
        if epoch > 80:
            lr /=16.
        elif epoch > 40:
            lr /= 8.
        elif epoch > 25:
            lr /= 4.
        elif epoch > 10:
            lr /= 2.
        print('Learning rate: ', lr)
        return lr

    lr_scheduler = kc.LearningRateScheduler(lr_schedule)

    model.fit_generator(train_generator, steps_per_epoch=(3041-305)//batch_size,epochs=200,validation_data=test_generator,validation_steps=301//8,callbacks=[checkpoint,lr_scheduler])

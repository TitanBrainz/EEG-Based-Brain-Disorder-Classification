import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, Dense, Flatten, MaxPooling1D, 
                                   GlobalAveragePooling1D, Dropout, BatchNormalization,
                                   Add, Concatenate)
from tensorflow.keras.regularizers import l2

# Add L2 regularization factor
L2_FACTOR = 0.01

def create_lenet(input_shape, num_classes):
    """LeNet-5 architecture adapted for 1D signals"""
    inputs = Input(shape=input_shape)
    
    # C1: First convolutional layer
    x = Conv1D(6, kernel_size=5, activation='tanh')(inputs)
    x = MaxPooling1D(pool_size=2, strides=2)(x)
    
    # C3: Second convolutional layer
    x = Conv1D(16, kernel_size=5, activation='tanh')(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x)
    
    # C5: Third convolutional layer
    x = Conv1D(120, kernel_size=5, activation='tanh')(x)
    
    # Fully connected layers
    x = Flatten()(x)
    x = Dense(84, activation='tanh')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs, outputs, name='LeNet-5-1D')

def create_alexnet(input_shape, num_classes):
    """AlexNet architecture adapted for 1D signals"""
    inputs = Input(shape=input_shape)
    
    # First convolutional block
    x = Conv1D(96, kernel_size=11, strides=4, activation='relu')(inputs)
    x = MaxPooling1D(pool_size=3, strides=2)(x)
    x = BatchNormalization()(x)
    
    # Second convolutional block
    x = Conv1D(256, kernel_size=5, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=3, strides=2)(x)
    x = BatchNormalization()(x)
    
    # Third convolutional block
    x = Conv1D(384, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv1D(384, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv1D(256, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=3, strides=2)(x)
    
    # Fully connected layers
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs, outputs, name='AlexNet-1D')

def create_resnet_block(inputs, filters, kernel_size=3, strides=1):
    """Basic ResNet block"""
    x = Conv1D(filters, kernel_size, strides=strides, padding='same', kernel_regularizer=l2(L2_FACTOR))(inputs)
    x = BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = Conv1D(filters, kernel_size, padding='same', kernel_regularizer=l2(L2_FACTOR))(x)
    x = BatchNormalization()(x)
    
    if strides != 1:
        inputs = Conv1D(filters, 1, strides=strides, padding='same', kernel_regularizer=l2(L2_FACTOR))(inputs)
    
    x = Add()([x, inputs])
    return tf.keras.activations.relu(x)

def create_resnet(input_shape, num_classes):
    """ResNet architecture adapted for 1D signals"""
    inputs = Input(shape=input_shape)
    
    x = Conv1D(64, kernel_size=7, strides=2, padding='same', kernel_regularizer=l2(L2_FACTOR))(inputs)
    x = BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = MaxPooling1D(pool_size=3, strides=2, padding='same')(x)
    
    # ResNet blocks
    x = create_resnet_block(x, 64)
    x = create_resnet_block(x, 64)
    x = create_resnet_block(x, 128, strides=2)
    x = create_resnet_block(x, 128)
    x = create_resnet_block(x, 256, strides=2)
    x = create_resnet_block(x, 256)
    
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs, outputs, name='ResNet-1D')

def inception_module(x, filters):
    """GoogLeNet Inception module adapted for 1D"""
    path1 = Conv1D(filters[0], 1, padding='same', activation='relu', kernel_regularizer=l2(L2_FACTOR))(x)
    
    path2 = Conv1D(filters[1][0], 1, padding='same', activation='relu', kernel_regularizer=l2(L2_FACTOR))(x)
    path2 = Conv1D(filters[1][1], 3, padding='same', activation='relu', kernel_regularizer=l2(L2_FACTOR))(path2)
    
    path3 = Conv1D(filters[2][0], 1, padding='same', activation='relu', kernel_regularizer=l2(L2_FACTOR))(x)
    path3 = Conv1D(filters[2][1], 5, padding='same', activation='relu', kernel_regularizer=l2(L2_FACTOR))(path3)
    
    path4 = MaxPooling1D(3, strides=1, padding='same')(x)
    path4 = Conv1D(filters[3], 1, padding='same', activation='relu', kernel_regularizer=l2(L2_FACTOR))(path4)
    
    return Concatenate(axis=-1)([path1, path2, path3, path4])

def create_googlenet(input_shape, num_classes):
    """GoogLeNet architecture adapted for 1D signals"""
    inputs = Input(shape=input_shape)
    
    # Initial convolutions
    x = Conv1D(64, 7, strides=2, padding='same', activation='relu', kernel_regularizer=l2(L2_FACTOR))(inputs)
    x = MaxPooling1D(3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    
    x = Conv1D(64, 1, activation='relu', kernel_regularizer=l2(L2_FACTOR))(x)
    x = Conv1D(192, 3, padding='same', activation='relu', kernel_regularizer=l2(L2_FACTOR))(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(3, strides=2, padding='same')(x)
    
    # Inception modules
    x = inception_module(x, [64, (96, 128), (16, 32), 32])
    x = inception_module(x, [128, (128, 192), (32, 96), 64])
    x = MaxPooling1D(3, strides=2, padding='same')(x)
    
    x = inception_module(x, [192, (96, 208), (16, 48), 64])
    x = inception_module(x, [160, (112, 224), (24, 64), 64])
    
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs, outputs, name='GoogLeNet-1D')

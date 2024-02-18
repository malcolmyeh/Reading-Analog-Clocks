from keras.models import Model
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout, Input, BatchNormalization, Activation

def load_model(img_size=150, CNN_KERNEL_SIZE=[5, 3, 3], CNN_STRIDES=[2, 2, 1], CNN_FILTERS=[32, 64, 128], POOL_SIZE=[(2, 2)], POOL_STRIDES=[2], DROPOUT_P=[0.3], HOUR_UNITS=[512], MINUTE_UNITS=[512]):
    # Input layer
    model_input = Input(shape=(img_size, img_size, 1))

    time = Conv2D(CNN_FILTERS[0], kernel_size=CNN_KERNEL_SIZE[0],
                  strides=CNN_STRIDES[0], activation='relu', padding='same')(model_input)

    # Hidden convolutional layers
    time = Conv2D(CNN_FILTERS[1], kernel_size=CNN_KERNEL_SIZE[1],
                  strides=CNN_STRIDES[1], activation='relu', padding='same')(time)
    time = Conv2D(CNN_FILTERS[2], kernel_size=CNN_KERNEL_SIZE[2],
                  strides=CNN_STRIDES[2], activation='relu', padding='same')(time)

    # Pooling layer
    time = MaxPooling2D(pool_size=POOL_SIZE[0], strides=POOL_STRIDES[0])(time)
    time = BatchNormalization()(time)

    # Dropout layer to help prevent overfitting by turning off nodes with probability p
    time = Dropout(DROPOUT_P[0])(time)

    # classification for hour, regression for minute

    # fully connected layers
    time = Flatten()(time)

    hour = Dense(HOUR_UNITS[0], activation='relu')(time)
    # hour output layer

    hour = Dense(12)(hour)
    hour = Activation('softmax', name='hour')(hour)


    minute = Dense(MINUTE_UNITS[0], activation='relu')(time)
    # minute output layer
    minute = Dense(1)(minute)
    minute = Activation('linear', name='minute')(minute)
    # minute = Dense(60)(minute)
    # minute = Activation('softmax', name='minute')(minute)

    return Model(inputs=model_input, outputs=[hour, minute])

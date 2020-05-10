from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Activation, Flatten, Input, BatchNormalization, Dropout

def VGG_16(img_size, f_maps, dropout = 0):
    inputs = Input((img_size[0], img_size[1], img_size[2]))

    # First block
    conv_1_1 = Conv2D(f_maps, kernel_size=(3,3), padding='same')(inputs)
    conv_1_1 = BatchNormalization()(conv_1_1)
    conv_1_1 = Activation('relu')(conv_1_1)

    conv_1_2 = Conv2D(f_maps, kernel_size=(3,3), padding='same')(conv_1_1)
    conv_1_2 = BatchNormalization()(conv_1_2)
    conv_1_2 = Activation('relu')(conv_1_2)

    pool_1 = MaxPooling2D(pool_size=(2,2))(conv_1_2)

    # Second block
    conv_2_1 = Conv2D(2*f_maps, kernel_size=(3,3), padding='same')(pool_1)
    conv_2_1 = BatchNormalization()(conv_2_1)
    conv_2_1 = Activation('relu')(conv_2_1)

    conv_2_2 = Conv2D(2*f_maps, kernel_size=(3,3), padding='same')(conv_2_1)
    conv_2_2 = BatchNormalization()(conv_2_2)
    conv_2_2 = Activation('relu')(conv_2_2)

    pool_2 = MaxPooling2D(pool_size=(2,2))(conv_2_2)

    # Third block
    conv_3_1 = Conv2D(4*f_maps, kernel_size=(3,3), padding='same')(pool_2)
    conv_3_1 = BatchNormalization()(conv_3_1)
    conv_3_1 = Activation('relu')(conv_3_1)

    conv_3_2 = Conv2D(4*f_maps, kernel_size=(3,3), padding='same')(conv_3_1)
    conv_3_2 = BatchNormalization()(conv_3_2)
    conv_3_2 = Activation('relu')(conv_3_2)

    conv_3_3 = Conv2D(4*f_maps, kernel_size=(3,3), padding='same')(conv_3_2)
    conv_3_3 = BatchNormalization()(conv_3_3)
    conv_3_3 = Activation('relu')(conv_3_3)

    pool_3 = MaxPooling2D(pool_size=(2,2))(conv_3_3)

    # Forth block
    conv_4_1 = Conv2D(8*f_maps, kernel_size=(3,3), padding='same')(pool_3)
    conv_4_1 = BatchNormalization()(conv_4_1)
    conv_4_1 = Activation('relu')(conv_4_1)

    conv_4_2 = Conv2D(8*f_maps, kernel_size=(3,3), padding='same')(conv_4_1)
    conv_4_2 = BatchNormalization()(conv_4_2)
    conv_4_2 = Activation('relu')(conv_4_2)

    conv_4_3 = Conv2D(8*f_maps, kernel_size=(3,3), padding='same')(conv_4_2)
    conv_4_3 = BatchNormalization()(conv_4_3)
    conv_4_3 = Activation('relu')(conv_4_3)

    pool_4 = MaxPooling2D(pool_size=(2,2))(conv_4_3)

    # Fifth block
    conv_5_1 = Conv2D(8*f_maps, kernel_size=(3,3), padding='same')(pool_4)
    conv_5_1 = BatchNormalization()(conv_5_1)
    conv_5_1 = Activation('relu')(conv_5_1)

    conv_5_2 = Conv2D(8*f_maps, kernel_size=(3,3), padding='same')(conv_5_1)
    conv_5_2 = BatchNormalization()(conv_5_2)
    conv_5_2 = Activation('relu')(conv_5_2)

    conv_5_3 = Conv2D(8*f_maps, kernel_size=(3,3), padding='same')(conv_5_2)
    conv_5_3 = BatchNormalization()(conv_5_3)
    conv_5_3 = Activation('relu')(conv_5_3)

    pool_5 = MaxPooling2D(pool_size=(2,2))(conv_5_3)

    # Last
    flat = Flatten()(pool_5)

    d1 = Dense(128)(flat)
    d1 = Activation('relu')(d1)
    if dropout != 0:
        d1 = Dropout(dropout)(d1)

    d2 = Dense(128)(d1)
    d2 = Activation('relu')(d2)
    if dropout != 0:
        d2 = Dropout(dropout)(d2)

    out = Dense(4)(d2)
    out = Activation('softmax')(out)

    model = Model(inputs=[inputs],outputs=[out])

    return model

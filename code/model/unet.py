from tensorflow import keras

class UNet:
    def __init__(self,
                 input_shape=(224,224,3),
                 batch_size=32,
                 filters=(64, 128, 256, 512, 1024),
                 activation='sigmoid',
                 dropout=0.5):
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.filters = filters
        self.activation = activation
        self.dropout = dropout
        self.model = None

    def summary(self):
        return self.model.summary()

    def load_weights(self, weights):
        self.model.load_weights(weights)

    def down_block(self, x, filters, kernel_size=(3, 3), padding="same", strides=1, dropout=False):
        c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
        c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
        if dropout:
            c = keras.layers.Dropout(0.5)(c)
        p = keras.layers.MaxPool2D((2, 2), (2, 2))(c)
        return c, p

    def up_block(self, x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
        us = keras.layers.UpSampling2D((2, 2))(x)
        concat = keras.layers.Concatenate()([us, skip])
        c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(concat)
        c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
        return c

    def bottleneck(self, x, filters, kernel_size=(3, 3), padding="same", strides=1, dropout=False):
        c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
        c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
        if dropout:
            c = keras.layers.Dropout(0.5)(c)
        return c

    def build_unet_model(self):
        #TO TRY : Try with smaller kernels and smaller image size ==> 128*128 and f = [16, 32, 64, 128, 256]
        f = self.filters
        inputs = keras.layers.Input(shape=self.input_shape)

        p0 = inputs
        c1, p1 = self.down_block(p0, f[0], dropout=True)
        c2, p2 = self.down_block(p1, f[1], dropout=True)
        c3, p3 = self.down_block(p2, f[2], dropout=True)
        c4, p4 = self.down_block(p3, f[3], dropout=True)

        bn = self.bottleneck(p4, f[4], dropout=True)

        u1 = self.up_block(bn, c4, f[3])
        u2 = self.up_block(u1, c3, f[2])
        u3 = self.up_block(u2, c2, f[1])
        u4 = self.up_block(u3, c1, f[0])

        #TO TRY : Check if this layer is helpful
        #u4 = keras.layers.Conv2D(2, 3, padding="same", activation="relu")(u4)

        outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(u4)
        self.model = keras.models.Model(inputs, outputs)

        return self.model

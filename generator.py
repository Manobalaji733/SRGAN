import tensorflow as tf
from tensorflow.keras import layers

def residual_block(x_in):
    x = layers.Conv2D(64, 3, padding='same')(x_in)
    x = layers.BatchNormalization()(x)
    x = layers.PReLU(shared_axes=[1,2])(x)
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    return layers.Add()([x_in, x])

def upsample_block(x_in):
    x = layers.Conv2D(256, 3, padding='same')(x_in)
    x = layers.Lambda(lambda t: tf.nn.depth_to_space(t, 2))(x)
    x = layers.PReLU(shared_axes=[1,2])(x)
    return x

def build_generator(num_res_blocks=16):
    # (None, None, 3) means "Any Height, Any Width, 3 Channels"
    inputs = layers.Input(shape=(None, None, 3)) 
    
    # ... rest of code remains exactly the same
##def build_generator(num_res_blocks=16):
    ##inputs = layers.Input(shape=(24,24,3))
    x = layers.Conv2D(64, 9, padding='same')(inputs)
    x = layers.PReLU(shared_axes=[1,2])(x)
    skip = x
    for _ in range(num_res_blocks):
        x = residual_block(x)
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, skip])
    x = upsample_block(x)
    x = upsample_block(x)
    outputs = layers.Conv2D(3, 9, padding='same', activation='tanh')(x)
    return tf.keras.Model(inputs, outputs, name='generator')

if __name__ == "__main__":
    m = build_generator()
    m.summary()

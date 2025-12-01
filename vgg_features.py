import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

# Input shape must match HR input
HR_SIZE = 96

def build_vgg():
    """
    Returns a VGG19 model that outputs feature maps from layer block5_conv4.
    """
    # Pre-trained VGG19 from ImageNet
    vgg = VGG19(include_top=False, weights='imagenet', input_tensor=Input(shape=(HR_SIZE, HR_SIZE, 3)))
    
    # Use block5_conv4 output
    model = Model(inputs=vgg.input, outputs=vgg.get_layer("block5_conv4").output)

    # Freeze VGG weights
    model.trainable = False

    return model


if __name__ == "__main__":
    model = build_vgg()
    model.summary()

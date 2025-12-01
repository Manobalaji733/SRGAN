import tensorflow as tf
from tensorflow.keras.applications.vgg19 import preprocess_input

# ========================
# 1. PIXEL LOSS (Structure)
# ========================
def pixel_loss(hr, sr):
    """
    Calculates the L1 distance between the High-Res and Super-Res images.
    L1 (Mean Absolute Error) produces sharper images than MSE.
    """
    return tf.reduce_mean(tf.abs(hr - sr))


# ========================
# 2. CONTENT LOSS (Texture)
# ========================
def content_loss(vgg, hr, sr):
    """
    Calculates perceptual similarity using VGG19 features.
    
    Inputs:
        vgg: The frozen VGG feature extractor model.
        hr: High-Res images (scaled [-1, 1])
        sr: Super-Res images (scaled [-1, 1])
    """
    # 1. Denormalize from [-1, 1] to [0, 255] for VGG
    # VGG expects pixels roughly in the 0-255 range (centered), not -1 to 1.
    hr_rescaled = (hr + 1.0) * 127.5
    sr_rescaled = (sr + 1.0) * 127.5

    # 2. Preprocess specifically for VGG19 (subtracts mean RGB)
    hr_pre = preprocess_input(hr_rescaled)
    sr_pre = preprocess_input(sr_rescaled)

    # 3. Extract Features
    hr_features = vgg(hr_pre)
    sr_features = vgg(sr_pre)

    # 4. Compute Mean Squared Error between features
    return tf.reduce_mean(tf.square(hr_features - sr_features))


# ========================
# 3. ADVERSARIAL LOSS (Generator)
# ========================
def adversarial_loss(fake_logits):
    """
    The Generator wants the Discriminator to believe the fake images are Real (1.0).
    """
    # Generator aims for label 1.0
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return loss_fn(tf.ones_like(fake_logits), fake_logits)


# ========================
# 4. DISCRIMINATOR LOSS (Classifier)
# ========================
def discriminator_loss(real_logits, fake_logits):
    """
    The Discriminator distinguishes Real (1.0) vs Fake (0.0).
    
    Includes LABEL SMOOTHING (0.9 instead of 1.0) to prevent 
    the discriminator from becoming too strong too fast.
    """
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # Real Loss: Target is 0.9 (smoothed) instead of 1.0
    # This is the fix for the "D_Loss: 0.000" problem.
    real_loss = loss_fn(tf.ones_like(real_logits) * 0.9, real_logits)

    # Fake Loss: Target is 0.0
    fake_loss = loss_fn(tf.zeros_like(fake_logits), fake_logits)

    return real_loss + fake_loss
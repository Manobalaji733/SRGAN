# src/train_quick.py — quick smoke test
import numpy as np
import tensorflow as tf

from generator import build_generator
from discriminator import build_discriminator
from vgg_features import build_vgg
from losses import pixel_loss, content_loss, adversarial_loss, discriminator_loss

g = build_generator(num_res_blocks=2)
d = build_discriminator(input_shape=(96,96,3))
vgg = build_vgg()

g_opt = tf.keras.optimizers.Adam(1e-4)
d_opt = tf.keras.optimizers.Adam(1e-4)

lr = np.tanh(np.random.randn(2,24,24,3).astype('float32'))
hr = np.tanh(np.random.randn(2,96,96,3).astype('float32'))

with tf.GradientTape() as tape:
    sr = g(lr, training=True)
    p_loss = pixel_loss(hr, sr)
grads = tape.gradient(p_loss, g.trainable_variables)
g_opt.apply_gradients(zip(grads, g.trainable_variables))
print("Pretrain pixel loss:", float(p_loss.numpy()))

with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    sr = g(lr, training=True)
    real_logits = d(hr, training=True)
    fake_logits = d(sr, training=True)
    p_loss = pixel_loss(hr, sr)
    c_loss = content_loss(vgg, hr, sr)
    adv_loss = adversarial_loss(fake_logits)
    g_total = p_loss + c_loss + 0.01 * adv_loss
    d_loss = discriminator_loss(real_logits, fake_logits)
grads_g = gen_tape.gradient(g_total, g.trainable_variables)
grads_d = disc_tape.gradient(d_loss, d.trainable_variables)
g_opt.apply_gradients(zip(grads_g, g.trainable_variables))
d_opt.apply_gradients(zip(grads_d, d.trainable_variables))
print("GAN step losses -> pixel:", float(p_loss.numpy()), "content:", float(c_loss.numpy()), "adv:", float(adv_loss.numpy()), "d_loss:", float(d_loss.numpy()))
print("Quick test finished — Model connections are correct!")

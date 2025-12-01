import os
import time
import tensorflow as tf

# ---------------- GPU Setup ----------------
# Allow memory growth to prevent crashes
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(e)

# ---------------- Imports ----------------
from data_loader import get_dataset_from_folder, HR_SIZE
from generator import build_generator
from discriminator import build_discriminator
from vgg_features import build_vgg
from losses import pixel_loss, content_loss, adversarial_loss, discriminator_loss

# ---------------- Configuration ----------------
# Standard SRGAN Settings
HR_SIZE = 96
BATCH_SIZE = 4           # Keep small for GANs (VGG + Disc + Gen takes memory)
EPOCHS = 20              # Run for longer in GAN phase
STEPS_PER_EPOCH = 500    # Steps per epoch
SAVE_FREQ = 1            # Save image every epoch

# Loss Weights (The "Secret Sauce" of SRGAN)
LAMBDA_PIXEL = 1.0       # Keeps structure
LAMBDA_CONTENT = 0.006   # Adds texture style (VGG)
LAMBDA_ADV = 0.001       # Adds sharpness (Discriminator)

# Learning Rates (Lower is often better for stability)
LR_GEN = 1e-4
LR_DISC = 1e-5  # Slow down the discriminator!

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HR_FOLDER = os.path.join(BASE_DIR, "data", "DIV2K")
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR = "results_gan"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------- Build Models ----------------
print("Building Models...")
generator = build_generator()
discriminator = build_discriminator()
vgg = build_vgg()

# ---------------- Load Pretrained Weights (CRITICAL) ----------------
# This loads the weights you are training RIGHT NOW in Phase 1
ckpt_pre = tf.train.Checkpoint(generator=generator)
manager_pre = tf.train.CheckpointManager(ckpt_pre, CHECKPOINT_DIR, max_to_keep=3)

if manager_pre.latest_checkpoint:
    print(f"✅ Loaded PRETRAINED generator from: {manager_pre.latest_checkpoint}")
    ckpt_pre.restore(manager_pre.latest_checkpoint).expect_partial()
else:
    print("⚠️ WARNING: No pre-trained weights found! Did you run train.py?")

# ---------------- Optimizers ----------------
gen_optimizer = tf.keras.optimizers.Adam(learning_rate=LR_GEN, beta_1=0.9)
disc_optimizer = tf.keras.optimizers.Adam(learning_rate=LR_DISC, beta_1=0.9)

# ---------------- Helper: Save Output ----------------
def save_gan_sample(lr, sr, hr, epoch):
    # Helper to convert tensors to uint8 images
    def process(t):
        return tf.cast(tf.clip_by_value((t + 1.0) * 127.5, 0, 255), tf.uint8)

    # Resize LR to HR size for easy comparison
    lr_resized = tf.image.resize(process(lr[0]), [HR_SIZE, HR_SIZE], method='nearest')
    sr_img = process(sr[0])
    hr_img = process(hr[0])

    # Combine visuals: LR | SR | HR
    grid = tf.concat([lr_resized, sr_img, hr_img], axis=1)
    tf.keras.preprocessing.image.save_img(os.path.join(RESULTS_DIR, f"gan_epoch_{epoch:03d}.png"), grid.numpy())

# ---------------- Training Step (The Core Logic) ----------------
@tf.function
def train_step(lr, hr):
    # --- 1. Train Discriminator ---
    with tf.GradientTape() as tape_d:
        sr = generator(lr, training=True)
        
        real_logits = discriminator(hr, training=True)
        fake_logits = discriminator(sr, training=True)
        
        d_loss = discriminator_loss(real_logits, fake_logits)
        
    grads_d = tape_d.gradient(d_loss, discriminator.trainable_variables)
    disc_optimizer.apply_gradients(zip(grads_d, discriminator.trainable_variables))

    # --- 2. Train Generator ---
    with tf.GradientTape() as tape_g:
        sr = generator(lr, training=True)
        fake_logits = discriminator(sr, training=True)
        
        # Calculate losses
        p_loss = pixel_loss(hr, sr)
        c_loss = content_loss(vgg, hr, sr)
        a_loss = adversarial_loss(fake_logits)
        
        # Combined Loss
        total_gen_loss = (LAMBDA_PIXEL * p_loss) + (LAMBDA_CONTENT * c_loss) + (LAMBDA_ADV * a_loss)

    grads_g = tape_g.gradient(total_gen_loss, generator.trainable_variables)
    gen_optimizer.apply_gradients(zip(grads_g, generator.trainable_variables))

    return d_loss, total_gen_loss, p_loss, c_loss, a_loss

# ---------------- Training Loop ----------------
def train_gan():
    dataset = get_dataset_from_folder(HR_FOLDER, batch_size=BATCH_SIZE)
    print("=== Starting GAN Training (Day 5) ===")
    
    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        
        # Track averages
        d_loss_avg = 0.0
        g_loss_avg = 0.0
        steps = 0
        
        for lr, hr in dataset.take(STEPS_PER_EPOCH):
            d_l, g_l, p_l, c_l, a_l = train_step(lr, hr)
            
            d_loss_avg += float(d_l)
            g_loss_avg += float(g_l)
            steps += 1
            
            if steps % 100 == 0:
                print(f"Ep {epoch} Step {steps} | D_Loss: {d_l:.3f} | G_Loss: {g_l:.3f}")

        # Summary
        print(f"Epoch {epoch} Done | Time: {time.time()-start:.1f}s | D_Loss: {d_loss_avg/steps:.3f} | G_Loss: {g_loss_avg/steps:.3f}")

        # Save Checkpoints & Images
        if epoch % SAVE_FREQ == 0:
            sr = generator(lr, training=False)
            save_gan_sample(lr, sr, hr, epoch)
            
            generator.save_weights(os.path.join(CHECKPOINT_DIR, f"gen_gan_epoch{epoch}.h5"))
            discriminator.save_weights(os.path.join(CHECKPOINT_DIR, f"disc_gan_epoch{epoch}.h5"))
            print(f"Saved checkpoint for epoch {epoch}")

if __name__ == "__main__":
    train_gan()
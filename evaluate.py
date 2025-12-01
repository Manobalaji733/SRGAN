import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
# Path to your HR images (Ground Truth)
HR_FOLDER = "./data/DIV2K/" 
CHECKPOINT_DIR = "checkpoints"
NUM_TEST_IMAGES = 20  # How many images to test?

from generator import build_generator

def preprocess_image(img_path):
    """Loads image and ensures dimensions are multiples of 4 (for clean downscaling)"""
    img = load_img(img_path)
    img = img_to_array(img)
    
    # Crop to make divisible by 4
    h, w, _ = img.shape
    h = h - (h % 4)
    w = w - (w % 4)
    img = img[:h, :w, :]
    
    return img

def evaluate():
    # 1. Setup GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)

    # 2. Load Model
    print("Loading Generator...")
    generator = build_generator()
    ckpt = tf.train.Checkpoint(generator=generator)
    manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_DIR, max_to_keep=3)
    
    if manager.latest_checkpoint:
        ckpt.restore(manager.latest_checkpoint).expect_partial()
        print(f"✅ Loaded weights: {manager.latest_checkpoint}")
    else:
        print("❌ No weights found! Run train.py first.")
        return

    # 3. Get List of Images
    all_files = [os.path.join(HR_FOLDER, f) for f in os.listdir(HR_FOLDER) if f.endswith(('.png', '.jpg'))]
    
    if not all_files:
        print(f"❌ No images found in {HR_FOLDER}")
        return
        
    # Pick random samples
    test_files = np.random.choice(all_files, NUM_TEST_IMAGES, replace=False)
    
    # Metrics Storage
    psnr_srgan, ssim_srgan = [], []
    psnr_bicubic, ssim_bicubic = [], []
    
    print(f"\n📊 Starting Evaluation on {NUM_TEST_IMAGES} images...")
    print("-" * 60)
    print(f"{'Image':<10} | {'PSNR (GAN)':<12} | {'SSIM (GAN)':<12} | {'PSNR (Bicubic)':<15}")
    print("-" * 60)

    for i, img_path in enumerate(test_files):
        # A. Prepare Data
        hr_original = preprocess_image(img_path) # Ground Truth (0-255)
        
        # Create Low Res Input (Downscale 4x)
        h, w, _ = hr_original.shape
        lr_size = (w // 4, h // 4)
        
        # Standard Resize (Bicubic) - creates the "bad" input
        lr_img = tf.image.resize(hr_original, [h // 4, w // 4], method='bicubic')
        
        # B. Generate SRGAN Prediction
        # Normalize LR to [-1, 1] for model
        lr_input = (lr_img / 127.5) - 1.0
        lr_input = tf.expand_dims(lr_input, axis=0)
        
        sr_gan = generator(lr_input, training=False)[0]
        
        # Denormalize SRGAN output to [0, 255]
        sr_gan = (sr_gan + 1.0) * 127.5
        sr_gan = tf.clip_by_value(sr_gan, 0, 255)
        
        # C. Generate Bicubic Baseline (Upscale LR back to HR size)
        sr_bicubic = tf.image.resize(lr_img, [h, w], method='bicubic')
        
        # D. Calculate Metrics
        # PSNR
        p_gan = tf.image.psnr(hr_original, sr_gan, max_val=255)
        p_bic = tf.image.psnr(hr_original, sr_bicubic, max_val=255)
        
        # SSIM
        s_gan = tf.image.ssim(hr_original, sr_gan, max_val=255)
        s_bic = tf.image.ssim(hr_original, sr_bicubic, max_val=255)
        
        # Store
        psnr_srgan.append(p_gan.numpy())
        ssim_srgan.append(s_gan.numpy())
        psnr_bicubic.append(p_bic.numpy())
        ssim_bicubic.append(s_bic.numpy())
        
        print(f"Img {i+1:<6} | {p_gan:.2f} dB      | {s_gan:.4f}       | {p_bic:.2f} dB")

    # 4. Final Report
    print("-" * 60)
    print("\n📈 FINAL RESULTS REPORT 📈")
    print(f"Average PSNR (SRGAN):   {np.mean(psnr_srgan):.2f} dB")
    print(f"Average SSIM (SRGAN):   {np.mean(ssim_srgan):.4f}")
    print("-" * 30)
    print(f"Average PSNR (Bicubic): {np.mean(psnr_bicubic):.2f} dB")
    print(f"Average SSIM (Bicubic): {np.mean(ssim_bicubic):.4f}")
    print("-" * 60)
    print("INTERPRETATION:")
    print("Note: SRGAN often has LOWER PSNR than Bicubic but looks better to human eyes.")
    print("This is because GANs hallucinate details that might not match the original pixels exactly,")
    print("whereas Bicubic is just a blurry average (mathematically 'safer' but uglier).")

if __name__ == "__main__":
    evaluate()
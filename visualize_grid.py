import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
HR_FOLDER = "./data/DIV2K"  # Path to your images
CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR = "final_comparisons"
os.makedirs(OUTPUT_DIR, exist_ok=True)

from generator import build_generator

def preprocess(img_path):
    img = load_img(img_path)
    img = img_to_array(img)
    # Crop to multiple of 4
    h, w, _ = img.shape
    h = h - (h % 4)
    w = w - (w % 4)
    img = img[:h, :w, :]
    return img

def create_comparison_grid():
    # 1. Setup GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError: pass

    # 2. Load Model
    print("Loading Generator...")
    generator = build_generator()
    ckpt = tf.train.Checkpoint(generator=generator)
    manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_DIR, max_to_keep=3)
    if manager.latest_checkpoint:
        ckpt.restore(manager.latest_checkpoint).expect_partial()
    else:
        print("❌ No weights found.")
        return

    # 3. Pick a Random Image
    all_files = [os.path.join(HR_FOLDER, f) for f in os.listdir(HR_FOLDER) if f.endswith(('.png', '.jpg'))]
    if not all_files:
        print("❌ No images found.")
        return
        
    img_path = np.random.choice(all_files)
    print(f"📸 Selected Image: {os.path.basename(img_path)}")

    # 4. Process Images
    hr_img = preprocess(img_path) # High Res (Ground Truth)
    h, w, _ = hr_img.shape
    
    # Create Low Res (Input)
    lr_img = tf.image.resize(hr_img, [h//4, w//4], method='bicubic')
    
    # Create Bicubic Upscale (Baseline)
    sr_bicubic = tf.image.resize(lr_img, [h, w], method='bicubic')
    
    # Create SRGAN Upscale (Our Model)
    lr_input = (lr_img / 127.5) - 1.0
    lr_input = tf.expand_dims(lr_input, axis=0)
    sr_gan = generator(lr_input, training=True)[0]
    sr_gan = (sr_gan + 1.0) * 127.5
    sr_gan = tf.clip_by_value(sr_gan, 0, 255)

    # 5. Plotting
    plt.figure(figsize=(20, 6))
    
    # A. Low Res
    plt.subplot(1, 4, 1)
    plt.title("Low Res Input (1/4 Size)")
    plt.imshow(lr_img.numpy().astype(np.uint8))
    plt.axis('off')

    # B. Bicubic
    plt.subplot(1, 4, 2)
    plt.title("Bicubic Interpolation")
    plt.imshow(sr_bicubic.numpy().astype(np.uint8))
    plt.axis('off')

    # C. SRGAN
    plt.subplot(1, 4, 3)
    plt.title("SRGAN (Ours)")
    plt.imshow(sr_gan.numpy().astype(np.uint8))
    plt.axis('off')

    # D. Original
    plt.subplot(1, 4, 4)
    plt.title("Original High Res")
    plt.imshow(hr_img.astype(np.uint8))
    plt.axis('off')
    
    # Save
    save_path = os.path.join(OUTPUT_DIR, f"grid_{os.path.basename(img_path)}")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✅ Saved Comparison Grid to: {save_path}")
    plt.show()

if __name__ == "__main__":
    create_comparison_grid()
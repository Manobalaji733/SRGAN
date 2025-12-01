import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
# Path to your BEST trained weights
CHECKPOINT_DIR = "checkpoints" 
# Folder to save test results
OUTPUT_DIR = "inference_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Import your model builder
# Make sure generator.py has input_shape=(None, None, 3)!
from generator import build_generator

def resolve_single_image(model, img_path):
    """
    Reads an image, checks its size, resizes if too big (to prevent OOM),
    upscales it using the SRGAN generator, and saves the result.
    """
    if not os.path.exists(img_path):
        print(f"❌ Error: Image not found at {img_path}")
        return

    print(f"🔍 Processing: {img_path}...")

    # 1. Load and Preprocess Image (With Safety Resize)
    try:
        # Load initially to check size
        img_temp = load_img(img_path)
        width, height = img_temp.size
        
        # --- CRITICAL FIX FOR MEMORY CRASH ---
        # If image is larger than 800px, resize it down first.
        # 800x800 input -> 3200x3200 output (Safe for RTX 4060)
        MAX_DIM = 800 
        
        if width > MAX_DIM or height > MAX_DIM:
            print(f"⚠️ Image is huge ({width}x{height}). Resizing to max {MAX_DIM}px to prevent GPU crash...")
            
            # Calculate scale to keep aspect ratio
            scale = MAX_DIM / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # Reload with new size
            lr_img = load_img(img_path, target_size=(new_height, new_width))
        else:
            # Size is safe, use original
            lr_img = img_temp

    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return

    # Convert to array
    lr_arr = img_to_array(lr_img)
    print(f"   Input Shape: {lr_arr.shape}")
    
    # Normalize to [-1, 1] (Generator expects this)
    lr_input = (lr_arr / 127.5) - 1.0
    
    # Add Batch Dimension: (1, Height, Width, 3)
    lr_input = np.expand_dims(lr_input, axis=0)

    # 2. Generate Super Resolution Image
    try:
        sr_output = model.predict(lr_input)
    except Exception as e:
        print(f"❌ Prediction failed (Likely OOM): {e}")
        return

    # 3. Post-process
    # Remove batch dimension
    sr_output = np.squeeze(sr_output, axis=0)
    # Denormalize from [-1, 1] to [0, 255]
    sr_output = (sr_output + 1.0) * 127.5
    sr_output = np.clip(sr_output, 0, 255).astype(np.uint8)

    # 4. Save Result
    filename = os.path.basename(img_path)
    save_path = os.path.join(OUTPUT_DIR, f"SR_{filename}")
    save_img(save_path, sr_output)
    
    print(f"✅ Saved High-Res image to: {save_path}")
    
    # 5. Visualization (Comparison)
    plt.figure(figsize=(12, 6))
    
    # Plot Low Res (Original / Resized Input)
    plt.subplot(1, 2, 1)
    plt.title(f"Input (Low Res)\n{lr_arr.shape}")
    plt.imshow(lr_arr.astype(np.uint8))
    plt.axis('off')
    
    # Plot Super Res (Generated)
    plt.subplot(1, 2, 2)
    plt.title(f"SRGAN Output (4x Upscale)\n{sr_output.shape}")
    plt.imshow(sr_output)
    plt.axis('off')
    
    # Show
    plt.tight_layout()
    plt.show()

def main():
    # 1. Setup GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("🚀 GPU Active for Inference")
        except RuntimeError as e:
            print(e)

    # 2. Build Generator
    # (None, None, 3) allows any image size
    print("Building Generator...")
    generator = build_generator() 

    # 3. Load Weights
    print("Loading Weights...")
    ckpt = tf.train.Checkpoint(generator=generator)
    manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_DIR, max_to_keep=3)

    if manager.latest_checkpoint:
        ckpt.restore(manager.latest_checkpoint).expect_partial()
        print(f"✅ Loaded Weights from: {manager.latest_checkpoint}")
    else:
        print("⚠️ WARNING: No weights found! The image will look like noise.")
        print("   Did you run train.py?")
        return

    # 4. Run Inference Loop
    print("\n--- SRGAN Inference Engine ---")
    while True:
        path = input("\nPaste the file path of an image to upscale (or 'q' to quit): ").strip()
        
        # Remove quotes if user copied path as "C:\path\..."
        path = path.replace('"', '') 
        
        if path.lower() == 'q':
            break
        
        resolve_single_image(generator, path)

if __name__ == "__main__":
    main()
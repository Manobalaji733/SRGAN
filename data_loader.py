# src/data_loader.py - TF efficient loader (low-memory, robust)
import tensorflow as tf
import os

AUTOTUNE = tf.data.AUTOTUNE

# Default settings - change if you want
HR_SIZE = 96
SCALE = 4
LR_SIZE = HR_SIZE // SCALE

def _load_and_preprocess(path):
    """Read image from path and return float32 image in [0,1]"""
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)   # [0,1]
    img.set_shape([None, None, 3])
    return img

def _random_crop_and_downsample(hr_img):
    """
    hr_img: float32 [0,1], arbitrary size >= HR_SIZE
    returns: lr_img, hr_img both in [-1,1]
    """
    hr_crop = tf.image.random_crop(hr_img, size=[HR_SIZE, HR_SIZE, 3])
    lr_img = tf.image.resize(hr_crop, [LR_SIZE, LR_SIZE], method='bicubic')
    hr_out = (hr_crop * 2.0) - 1.0
    lr_out = (lr_img * 2.0) - 1.0
    return lr_out, hr_out

def get_dataset_from_folder(hr_folder, batch_size=2, shuffle=True, shuffle_buffer=512, prefetch=8):
    """
    Returns tf.data.Dataset yielding (lr_batch, hr_batch)
    - hr_folder: path to folder with HR images (string)
    - batch_size: int
    - shuffle_buffer: int (512 default)
    - prefetch: int (8 default)
    """
    # Normalize path
    hr_folder = os.path.abspath(hr_folder)
    pattern = os.path.join(hr_folder, "*")

    # Use tf.io.gfile.glob so it works with local and remote filesystems
    files = tf.io.gfile.glob(pattern)
    if not files:
        raise FileNotFoundError(
            f"No files found for pattern: {pattern}\n"
            "Make sure the folder exists, contains image files (png/jpg), and path is correct.\n"
            "If you want to create test images, run the sample image creation snippet provided in README or ask me."
        )

    # Create dataset of file paths (string tensors)
    ds = tf.data.Dataset.from_tensor_slices(files)

    if shuffle:
        # shuffle files list (small buffer is fine)
        ds = ds.shuffle(buffer_size=min(len(files), shuffle_buffer))

    # Map to decoded images lazily
    ds = ds.map(lambda p: _load_and_preprocess(p), num_parallel_calls=AUTOTUNE)

    # Filter images smaller than HR_SIZE
    ds = ds.filter(lambda img: tf.logical_and(tf.shape(img)[0] >= HR_SIZE, tf.shape(img)[1] >= HR_SIZE))

    # Map to LR/HR pairs
    ds = ds.map(lambda img: _random_crop_and_downsample(img), num_parallel_calls=AUTOTUNE)

    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(prefetch)

    return ds

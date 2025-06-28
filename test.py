import cv2
import numpy as np
from scipy.signal import correlate2d
import argparse
import os
import time
from skimage.restoration import denoise_wavelet

def extract_prnu(image_path, resize_dim=(512, 512)):
    """Extract a more advanced PRNU (noise residual) from a grayscale image using wavelet denoising."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    img = cv2.resize(img, resize_dim).astype(np.float32) / 255.0

    # Apply wavelet denoising to simulate high-frequency removal
    denoised = denoise_wavelet(img, channel_axis=None, rescale_sigma=True)
    denoised = np.clip(denoised, 0, 1)
    noise_residual = img - denoised
    return noise_residual

def compute_prnu_score(noise_residual):
    """Compute the self-correlation PRNU score from the noise residual."""
    corr = correlate2d(noise_residual, noise_residual, mode='same')
    center = corr.shape[0] // 2
    score = corr[center, center] / np.max(corr)
    return float(score)

def classify_by_prnu(score, threshold=0.95):
    """Classify image as real or fake based on PRNU score."""
    if score >= threshold:
        return "REAL (Strong PRNU detected)"
    else:
        return "FAKE or Weak PRNU"

def main():
    parser = argparse.ArgumentParser(description="PRNU-based Real/Fake Image Classifier")
    parser.add_argument("image", help="Path to the image file")
    args = parser.parse_args()

    image_path = args.image
    if not os.path.isfile(image_path):
        print(f"[ERROR] File not found: {image_path}")
        return

    print(f"[INFO] Analyzing: {image_path}")
    start_time = time.time()

    try:
        prnu_residual = extract_prnu(image_path)
        prnu_score = compute_prnu_score(prnu_residual)
        classification = classify_by_prnu(prnu_score)

        print(f"[RESULT] PRNU Score: {prnu_score:.4f}")
        print(f"[RESULT] Classification: {classification}")

    except Exception as e:
        print(f"[ERROR] Failed to process image: {str(e)}")

    print(f"[INFO] Total time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()

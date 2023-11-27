import numpy as np
import math
from scipy.ndimage import gaussian_filter
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

from PIL import Image


# Function to calculate Mean Square Error (MSE)
def psnr(original, new):
    mse = np.mean((original - new) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr_val = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr_val


def ssim(img1, img2):
    """
    Compute the SSIM (Structural Similarity Index) between two images.

    :param img1: First image
    :param img2: Second image
    :return: SSIM value
    """
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mu1 = gaussian_filter(img1, 1.5)
    mu2 = gaussian_filter(img2, 1.5)

    sigma1 = gaussian_filter(img1 ** 2, 1.5)
    sigma2 = gaussian_filter(img2 ** 2, 1.5)
    sigma12 = gaussian_filter(img1 * img2, 1.5)

    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2))

    return ssim_map.mean()


def image_delta_e(file1, file2):
    """
    Calculate the average Delta E (CIE 2000) color difference between two images provided as file objects.

    :param file1: File object of the first image.
    :param file2: File object of the second image.
    :return: Average Delta E 2000 value.
    """
    # Load images from file objects
    image1 = np.array(Image.open(file1))
    image2 = np.array(Image.open(file2))

    # Check if images have the same dimensions
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions")

    # Initialize variables for Delta E calculation
    delta_e_total = 0
    num_pixels = image1.shape[0] * image1.shape[1]

    # Calculate Delta E for each pixel
    for i in range(image1.shape[0]):
        for j in range(image1.shape[1]):
            # Convert RGB to Lab
            color1_rgb = sRGBColor(float(image1[i, j, 0]), float(image1[i, j, 1]), float(image1[i, j, 2]), is_upscaled=True)
            color1_lab = convert_color(color1_rgb, LabColor)

            color2_rgb = sRGBColor(float(image2[i, j, 0]), float(image2[i, j, 1]), float(image2[i, j, 2]), is_upscaled=True)
            color2_lab = convert_color(color2_rgb, LabColor)

            # Compute Delta E for the current pixel
            delta_e = delta_e_cie2000(color1_lab, color2_lab)
            delta_e_total += delta_e

    # Calculate average Delta E
    avg_delta_e = delta_e_total / num_pixels

    return avg_delta_e


def mse(image_a, image_b):
    """
    Compute the mean squared error between two images.

    :param image_a: First image.
    :param image_b: Second image.
    :return: MSE value.
    """
    err = np.sum((image_a.astype("float") - image_b.astype("float")) ** 2)
    err /= float(image_a.shape[0] * image_a.shape[1])

    return err

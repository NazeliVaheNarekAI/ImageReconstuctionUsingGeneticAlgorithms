import math

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.color import rgb2lab, deltaE_ciede2000
import cv2


# options = {'psnr': True, 'ssim': True, 'delta_e': False, 'mse': True}

def combine_metrics(options, image1, image2, max_psnr=60, max_mse=10000):
    """
    Combine normalized values of PSNR, SSIM, Delta E, and MSE based on chosen options.

    :param options: Dictionary with keys as metric names ('psnr', 'ssim', 'delta_e', 'mse') and boolean values.
    :param image1: First image or file object, depending on the metric.
    :param image2: Second image or file object, depending on the metric.
    :param max_psnr: Maximum PSNR value for normalization.
    :param max_mse: Maximum MSE value for normalization.
    :return: Combined metric value.
    """
    combined_value = 1

    image1 = np.array(image1)
    image2 = np.array(image2)

    if options.get('psnr'):
        combined_value *= psnr(image1, image2, max_psnr)
    if options.get('ssim'):
        combined_value *= ssim(image1, image2)
    if options.get('delta_e'):
        combined_value *= average_delta_e(image1, image2)  # Assuming image1 and image2 are file objects here
    if options.get('mse'):
        combined_value *= mse(image1, image2, max_mse)

    return combined_value


def psnr(original, new, max_psnr=60):
    """
    Compute the PSNR (Peak Signal to Noise Ratio) between two images.

    :param original: First image
    :param new: Second image
    :param max_psnr: Maximum PSNR value for normalization.
    """
    mse_val = mse_raw(original, new)
    if mse_val == 0:
        return 1
    max_pixel = 255.0
    psnr_val = 20 * math.log10(max_pixel / math.sqrt(mse_val))
    return 1 - min(psnr_val / max_psnr, 1)
    # return psnr_val


def ssim(img1, img2):
    """
    Compute the SSIM (Structural Similarity Index) between two images.

    :param img1: First image (PIL Image object).
    :param img2: Second image (PIL Image object).
    :return: SSIM value
    """
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    # Convert PIL Images to NumPy arrays and then to float64
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # SSIM calculation
    mu1 = gaussian_filter(img1, 1.5)
    mu2 = gaussian_filter(img2, 1.5)

    sigma1 = gaussian_filter(img1 ** 2, 1.5)
    sigma2 = gaussian_filter(img2 ** 2, 1.5)
    sigma12 = gaussian_filter(img1 * img2, 1.5)

    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2))

    ssim_val = ssim_map.mean()
    return 1 - 0.5 * (1 - ssim_val)


def average_delta_e(image1, image2):
    # Load images
    # Convert images to Lab color space
    image1_lab = rgb2lab(image1)
    image2_lab = rgb2lab(image2)

    # Resize images if they are not the same size
    if image1_lab.shape != image2_lab.shape:
        image2_lab = cv2.resize(image2_lab, (image1_lab.shape[1], image1_lab.shape[0]), interpolation=cv2.INTER_AREA)

    # Calculate Delta E for each pixel
    delta_e = deltaE_ciede2000(image1_lab, image2_lab)

    # Average Delta E
    avg = np.mean(delta_e)

    return 1 - min(avg / 100, 1)


def mse(image_a, image_b, max_mse=10000):
    err = np.mean((image_a.astype("float") - image_b.astype("float")) ** 2)
    return 1 - min(err / max_mse, 1)


def mse_raw(image_a, image_b):
    err = np.mean((image_a.astype("float") - image_b.astype("float")) ** 2)
    return err


# if __name__ == '__main__':
#     # img1 = Image.open('images/mona_lisa.jpg')
#     # img2 = Image.open('images/mona_lisa_0.jpg')
#     # print(combine_metrics({'psnr': True, 'ssim': True, 'delta_e': False, 'mse': True}, img1, img2))
#     # print(combine_metrics({'psnr': True, 'ssim': True, 'delta_e': False, 'mse': True}, img1, img1))
#     # print(combine_metrics({'psnr': True, 'ssim': True, 'delta_e': False, 'mse': True}, img2, img2))
#     #
#     # img1 = Image.open('images/mona_lisa.jpg')
#     # img2 = Image.open('images/mona_lisa_0.jpg')
#     # print(combine_metrics({'psnr': True, 'ssim': True, 'delta_e': False, 'mse': True}, img1, img2))
#     # print(combine_metrics({'psnr': True, 'ssim': True, 'delta_e': False, 'mse': True}, img1, img1))
#     # print(combine_metrics({'psnr': True, 'ssim': True, 'delta_e': False, 'mse': True}, img2, img2))
#
#     img1 = Image.open('images/mona_lisa.jpg')
#     img2 = Image.open('images/mona_lisa_0.jpg')
#     # print(average_delta_e(img1, img2))
#     print(combine_metrics({'psnr': True, 'ssim': True, 'delta_e': True, 'mse': True}, img1, img2))
#     # print(combine_metrics({'psnr': True, 'ssim': True, 'delta_e': False, 'mse': True}, img1, img1))
#     # print(combine_metrics({'psnr': True, 'ssim': True, 'delta_e': False, 'mse': True}, img2, img2))

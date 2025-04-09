import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
from scipy import ndimage as ndi
from skimage.filters import roberts
from skimage.measure import regionprops, label
from skimage.morphology import binary_closing, disk, binary_erosion
from skimage.segmentation import clear_border


def normalize_hu_values(image: np.ndarray, min_bound: int = -1000, max_bound: int = 400) -> np.ndarray:
    """
    归一化HU值到[0,1]范围

    Args:
        image: 输入图像
        min_bound: 最小HU值
        max_bound: 最大HU值

    Returns:
        np.ndarray: 归一化后的图像
    """
    image = (image - min_bound) / (max_bound - min_bound)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image

def get_segmented_lungs(im, plot=False):
    '''
        extract lung ROI from pixel array

    :param im:      a patient's piexl array
    :param plot:    if plot when segment
    :return:
    '''
    # Step 1: Convert into a binary image.
    binary = im < -400
    # Step 2: Remove the blobs connected to the border of the image.
    cleared = clear_border(binary)
    # Step 3: Label the image.
    label_image = label(cleared)
    # Step 4: Keep the labels with 2 largest areas.
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                       label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    # Step 5: Erosion operation with a disk of radius 2. This operation is seperate the lung nodules attached to the blood vessels.
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    # Step 6: Closure operation with a disk of radius 10. This operation is    to keep nodules attached to the lung wall.
    selem = disk(10) # CHANGE BACK TO 10
    binary = binary_closing(binary, selem)
    # Step 7: Fill in the small holes inside the binary mask of lungs.
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    # Step 8: Superimpose the binary mask on the input image.
    get_high_vals = binary == 0
    im[get_high_vals] = -2000
    if plot:
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(binary, cmap='gray')
        plt.title('Lung Mask')
        plt.subplot(1, 2, 2)
        plt.imshow(im, cmap='gray')
        plt.title('Masked Image')
        plt.show()
    return im, binary
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imsave
import PIL.Image
from tqdm import tqdm


def read_rgbval(image_path):
    return np.array(PIL.Image.open(image_path))


def show_image(pixel_values):
    plt.imshow(pixel_values)
    plt.tick_params(which='both',
                    bottom=False,
                    top=False,
                    labelbottom=False,
                    labelleft=False,
                    left=False)
    plt.show()


def save_image(path, rgb_val_image):
    im = PIL.Image.fromarray(rgb_val_image)
    im.save(path)


def baw_onetrh(rgb_val, trh):
    """Single value treshholding"""
    blackandwhite = rgb_val
    for i in range(len(blackandwhite)):
        column = blackandwhite[i]
        for p in range(len(column)):
            if np.sum(blackandwhite[i][p]) <= trh:
                blackandwhite[i][p] = blackandwhite[i][p] * 0
            else:
                blackandwhite[i][p] = blackandwhite[i][p] * 0
                blackandwhite[i][p] = blackandwhite[i][p] + 255

    return blackandwhite


def baw_twotrh(rgb_val, ltrh, htrh):
    """Double value treshholding"""
    blackandwhite = rgb_val
    for i in range(len(blackandwhite)):
        column = blackandwhite[i]
        for p in range(len(column)):
            if (np.sum(blackandwhite[i][p]) > htrh or
                    np.sum(blackandwhite[i][p]) < ltrh):
                blackandwhite[i][p] = blackandwhite[i][p] * 0
            else:
                blackandwhite[i][p] = blackandwhite[i][p] * 0
                blackandwhite[i][p] = blackandwhite[i][p] + 255

    return blackandwhite


def calculate_grayscale_value(rgbv):
    return rgbv[0] * 299/1000 + rgbv[1] * 587/1000 + rgbv[2] * 114/1000


def make_grayscale(rgb_val):
    grayscaled = rgb_val
    for i in range(len(grayscaled)):
        column = grayscaled[i]
        for p in range(len(column)):
            grayscaled[i][p] = np.median(grayscaled[i][p])

    return grayscaled


def equalize_hist_grayscale(rgb_val):
    unique_values = np.bincount(rgb_val.flatten())
    sum_of_pixels = np.sum(unique_values)
    probabilities = unique_values/sum_of_pixels
    occurences_before = np.cumsum(probabilities)
    normalized_values = np.floor(occurences_before * 255).astype(np.uint8)
    equalized_image = [normalized_values[pixel_value] for pixel_value in rgb_val.flatten()]
    return equalized_image


def naive_approach_mask(rgb_val, mask_size):
    mask = rgb_val.copy()
    msk_size = mask_size * mask_size
    half_size = int(mask_size/2)
    for index in tqdm(range(rgb_val.shape[0])):
        for pixel_index in range(rgb_val.shape[1]):
            column_s = 0 if index - half_size < 0 else index - half_size
            column_e = len(mask) if index + half_size > len(mask) else index + half_size + 1
            
            row_s = 0 if pixel_index - half_size < 0 else pixel_index - half_size
            row_e = len(mask[0]) if pixel_index + half_size > len(mask[0]) else pixel_index + half_size + 1

            mask[index][pixel_index] = int((np.sum(rgb_val[column_s: column_e, row_s: row_e, 0])/msk_size))

    return mask.astype(np.uint8)


def sat_table_mask(rgb_val, mask_size):
    mask = rgb_val.copy()
    image = rgb_val.copy()
    mask = (mask.cumsum(axis = 0).cumsum(axis = 1))
    mask = mask.astype(np.int64)
    hf_size = mask_size//2
    
    for i in tqdm(range(hf_size, len(mask[0]) - hf_size)):
        for z in range(hf_size,len(mask) - hf_size):
            A = mask[z - hf_size, i + hf_size, 0]
            B = mask[z + hf_size, i + hf_size, 0]
            C = mask[z - hf_size, i - hf_size, 0]
            D = mask[z + hf_size, i - hf_size, 0]
            image[z][i] = -1*((D - B - C + A)/mask_size**2)

    return image
vals = read_rgbval("roadgrayscale.png")

# show_image(sat_table_mask(vals, 71))
img_1 = sat_table_mask(vals, 71)
img_2 = naive_approach_mask(vals, 71)
save_image("Blurroad.png",img_1)

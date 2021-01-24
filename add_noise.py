'''
Add noise to images in a specific folder. Images will be replaced to noisy version.
'''

import glob
import cv2
import numpy as np
from skimage.util import random_noise


folder_path = './generated_dataset_noise/'


for x in glob.glob(folder_path + '**', recursive=True):

    if "train" in x and x.endswith('jpg'):

        img = cv2.imread(x)
        noise_img = random_noise(img, mode='gaussian')
        noise_img = np.array(255*noise_img, dtype='uint8')

        cv2.imwrite(x, noise_img)

print("Done")

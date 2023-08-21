from PIL import Image
from scipy import signal
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':

    filter = np.asarray([[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]])
    image = Image.open('nice_birb.JPEG').convert('L')

    image_convolved = signal.convolve2d(image, filter)
    plt.imshow(image, cmap='gray')
    plt.show()
    plt.imshow(image_convolved, cmap='gray')
    plt.show()

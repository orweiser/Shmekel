import numpy as np
from api.datasets import MNIST
import matplotlib.pyplot as plt


def plot_im(img):
    plt.figure()
    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    a = MNIST("MNIST with noise", num_classes=11, val_mode=False)
    x = a[0]['inputs']
    y = a[0]['outputs']
    plot_im(x)

import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

im_gray = cv.imread('sunspot.jpeg',cv.IMREAD_GRAYSCALE)
im_shape = im_gray.shape
im_gray = im_gray[:im_shape[0],im_shape[1]-im_shape[0]:im_shape[1]]

mask = create_circular_mask(1532, 1532, radius=5)
masked_img = np.ones([1532, 1532])
masked_img[~mask] = 0

# plt.imshow(im_gray, cmap='YlOrBr_r')
# plt.show()

f = np.fft.fft2(masked_img) #must be complex amplitude going in here
fshift = np.fft.fftshift(f)
intensity_image = (np.abs(fshift))**2

plt.imshow(intensity_image)
plt.show()


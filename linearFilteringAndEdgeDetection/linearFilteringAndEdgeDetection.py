# coding: utf-8

# # Linear filtering and edge detection
# 
# This project will apply multiple linear filters to images. It uses python and numpy
# 
# **Instructions:**
# - Create a Gaussian filter and a box filter and write a program to filter the image with a given filter.
# Do not use an existing filtering function (write the function from scratch).Test these filters at different sizes
# and analyze their effects on the stop sign image.
# 
# - Create an edge filter and filter the image. Can you reliably detect edges?
# Create your own edge detector from scratch – do your best to create a good edge detector.
# 
# - Think of a filter that will respond strongly on the stop sign only. Create it and filter the image. Analyze the
# results and improve your filter to detect as well as possible the sign by using your filter. The goal is to
# automatically draw a bounding box just around the stop sign (and not around other regions in the images).

# Import the needed python libraries

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

#get_ipython().magic('matplotlib inline')

# I will use Python libraries to load the image. The image will be a numpy array with the shape (height, width, 3),
# representing the R, G, B matrices.

# In[2]:


my_image = "stopSign.jpg"  # change this to the name of your image file
fname = "images/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
print(image.shape)
plt.imshow(image)


# I will define the convolution filtering function. convolve3d will apply the kernel over each pixel. This operation
# will be executed for every color of the image (R,G,B).

# In[3]:

def convolve3d(img, kernel):
    # Add padding around the image. I chose the replicate border
    border_size = int(kernel.shape[0] / 2)
    img_replicate = cv2.copyMakeBorder(img, border_size, border_size, border_size, border_size, cv2.BORDER_REPLICATE)

    # We need a copy of the original image to place the results
    img_out = np.zeros(img.shape, dtype=np.uint8)
    img_height = img.shape[0]
    img_width = img.shape[1]

    print('Border size ' + str(border_size) + 'height ' + str(img_height) + 'width ' + str(img_width))

    # Vectorized multiplication for each pixel
    for d in range(3):
        for row in range(border_size, img_height):
            for col in range(border_size, img_width):
                # Python slicing to subtract the matrix
                sliced_image = img_replicate[(row - border_size):(row + border_size + 1),
                               (col - border_size):(col + border_size + 1), d]
                # Vectorized matrix multiplication
                value = np.multiply(kernel, sliced_image)
                img_out[row, col, d] = min(255, max(0, value.sum()))

    return img_out


# In the next step we will generate different box size filters, the box filter values will be the average across
# pixels. Then we will compare the results. The first filter will represent a 3x3 box filter.

# In[14]:


def generate_box_filter(size):
    kernel = np.zeros((size, size), dtype=np.float)
    for row in range(0, size):
        for col in range(0, size):
            kernel[row, col] = round(1 / (size * size), 2)

    return kernel


# def generateGaussianFilter(size):

# This method uses convolution to apply a kernel to 
# each pixel color of an image
testImage = np.array([[[244, 0, 0], [0, 244, 0], [0, 0, 244]],
                      [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                      [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])

testKernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])

# Using this kernel the initial image should be displayed
boxKernel = np.array([
    [0.1, 0.1, 0.1],
    [0.1, 0.2, 0.1],
    [0.1, 0.1, 0.1]])

edgeKernel = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]])

laplacianKernel = np.array([
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]])

# stopSignKernel =
print(generate_box_filter(10))

# Filtered image using different size box filters
# 

# In[116]:


filteredImage = convolve3d(image, generate_box_filter(9))
plt.imshow(filteredImage)

# Filtered image using an edgeKernel

# In[13]:


edgeKernel = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]])
filteredImage = convolve3d(image, edgeKernel)
plt.imshow(filteredImage)

# Filtered image using an Laplacian Kernel:

# In[11]:


laplacianKernel = np.array([
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]])

filteredImage = convolve3d(image, laplacianKernel)
plt.imshow(filteredImage)

# In order to detect the stop sign we need a filter that will respond to those specific pixels.
# We will extract the Stop sign from the image and then filter the image using the Stop sign as a mask

# In[114]:


import matplotlib.patches as patches


def detect_stop_sign(img, template):
    padding_size = int(template.shape[0] / 2)
    img_height = img.shape[0]
    img_width = img.shape[1]

    # We only detect full objects so we can start the search taking this into account
    for row in range(padding_size, img_height - padding_size):
        for col in range(padding_size, img_width - padding_size):
            # Python slicing to subtract the matrix
            sliced_image = img[(row - padding_size):(row + padding_size + 1),
                           (col - padding_size):(col + padding_size + 1), 0]

            # Will apply the dot product between the template and the window in the image
            # We should get the identity matrix if we detect the object
            value = np.dot(template, sliced_image).sum()

            if round(value, 10) - template.shape[0] == 0:
                # Return shape coordinates x, y
                return np.array([col - padding_size, row - padding_size])

    return np.array([0, 0])


# Extract the stop sign template from the original image and calculate the inverse on one color
stopSign = image[60:205, 370:515, :]
invStopSign = np.linalg.inv(stopSign[:, :, 0])

# plt.imshow(image)
# Plot the result
signDetection = detect_stop_sign(image, invStopSign)

# Create figure and axes
fig, ax = plt.subplots(1)
# Display the image
ax.imshow(image)
# Create a Rectangle patch
rect = patches.Rectangle((signDetection[0], signDetection[1]), stopSign.shape[0], stopSign.shape[0],
                         linewidth=1, edgecolor='r', facecolor='none')
# Add the patch to the Axes
ax.add_patch(rect)
plt.show(block=True)

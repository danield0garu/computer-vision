
# coding: utf-8

# # Human skin segmentation
# 
# **Instructions:**
# 1. Download 10 images with human faces from the web.
# 2. Threshold the image based on a certain hue range, in order to get regions containing human skin.
# 3. Remove the smallest, noisy connected components.
# 4. Dilate and erode the remaining blobs.
# 5. Compute regions properties of the remaining blobs (center and main axes) and keep the sufficiently round ones in order to obtain human faces. Fit ellipses to those blobs and show them.

# ### 1. Load and display images ###

# In[120]:


import cv2
import glob
import numpy as np
from random import randint
import matplotlib.pyplot as plt


files = glob.glob("imagesHw2/faces/*.jpg")

filesArray = []

# Select 10 random images from Caltech faces dataset and load into a 4D array
for x in range(0, 10):
    imageIndex = randint(0, 449)
    # Open CV reads images in BGR
    imageBGR = cv2.imread(files[imageIndex])
    imageRGB = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2RGB)
    filesArray.append(imageRGB)

filesArray = np.array(filesArray)
 
# Display images
for idx, el in enumerate(filesArray):
    plt.imshow(filesArray[idx])
    plt.show()


# ### 2. Apply HUE tresholding

# In[121]:


def pixelMask(RGBHSVPixel):
    R, G, B, H, S = RGBHSVPixel[0], RGBHSVPixel[1], RGBHSVPixel[2], RGBHSVPixel[3] * 2, RGBHSVPixel[4] / 255
    if 0 <= H and H <= 50 and 0.23 <= S and S <= 0.68 and R > 95 and G > 40 and B > 20 and R > G and R > B and np.absolute(R-G) > 15:
        return 255
    else:
        return 0
    
   
# We need to obtain a binary image (black and white) where white will be the foreground feature and black the 
# backgound. From a 3D image it will return a 2D image
def filterByHUE(image):
    hsvImage = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    #For HSV, Hue range is [0,179], Saturation range is [0,255] and Value range is [0,255]. 
    print("Started")
    rgbhsvimage = np.concatenate((image, hsvImage), axis = 2)
    binaryImage = np.apply_along_axis(pixelMask, 2, rgbhsvimage)
    print("Finished")
    return binaryImage
    
binaryImages = []

for idx, el in enumerate(filesArray):
    binaryImage = filterByHUE(filesArray[idx])
    binaryImages.append(binaryImage)
    plt.imshow(binaryImage, cmap='gray')
    plt.show()


# ### 3. Remove the small connected components

# In[122]:


binaryImages = np.array(binaryImages)
binaryImagesFiltered = []

for i in range (0, binaryImages.shape[0]):
    imCopy = np.uint8(binaryImages[i])
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(imCopy, 4)
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    blobSizeTreshold = 1000
    img2 = np.zeros((output.shape))
    
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255 
    
    binaryImagesFiltered.append(img2)

binaryImagesFiltered = np.array(binaryImagesFiltered)

#Display images
for idx, el in enumerate(binaryImagesFiltered):
    plt.imshow(binaryImagesFiltered[idx], cmap='gray')
    plt.show()


# ### 4. Dilate and erode

# In[123]:


kernel = np.ones((3,3),np.uint8)
imOpen = []
for i in range (0, binaryImagesFiltered.shape[0]):
    imErode = cv2.erode(binaryImagesFiltered[i], kernel, iterations = 1)
    imDilate = cv2.dilate(imErode, kernel, iterations = 1)
    imOpen.append(imDilate)

#Display images
for idx, el in enumerate(imOpen):
    plt.imshow(imOpen[idx], cmap='gray')
    plt.show()


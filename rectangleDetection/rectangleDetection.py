# TODO figure how to use the function from common location
# from ../util/cvUtil import auto_canny

import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob
import scipy.misc
import os


# This method is useful for applying Canny filter and automatically determining the thresholds.
# TODO move to util
def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


# TODO move to util
def transform_to_edge_images():
    image_paths = glob.glob("images/*.jpg")

    for imagePath in image_paths:
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edge_image = auto_canny(blurred)

        image_name = os.path.split(imagePath)[-1]
        scipy.misc.imsave("edgeImages/" + image_name, edge_image)

#transform_to_edge_images()


# TODO move to util
def plot_image_array(image_array):
    for image in image_array:
        plt.figure()
        plt.imshow(image, cmap="gray")


# TODO move to util
def plot_hough_lines(rho, theta, image):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho

    fig2, ax1 = plt.subplots(ncols=1, nrows=1)
    ax1.imshow(image)

    for i in range(0, len(rho)):
        ax1.plot([x0[i] + 1000 * (-b[i]), x0[i] - 1000 * (-b[i])],
                 [y0[i] + 1000 * (a[i]), y0[i] - 1000 * (a[i])],
                 'xb-', linewidth=3)

    ax1.set_ylim([image.shape[0], 0])
    ax1.set_xlim([0, image.shape[1]])

    plt.show()

def getLength(startPoint,secondPoint):
    ##Inputs:
    #startPoint - [x,y]
    #secondPoint - [x,y]

    ##Outputs:
    #lenv - length between two points

    v1x=secondPoint[0]-startPoint[0]
    v1y=secondPoint[1]-startPoint[1]
    lenv=np.sqrt(v1x*v1x+v1y*v1y)
    return lenv

edgeImagePaths = glob.glob("edgeImages/*.jpg")
edgeImageArray = []
# Read edge images
for edgeImage in edgeImagePaths:
    e_image = cv2.imread(edgeImage)
    gray_e = cv2.cvtColor(e_image, cv2.COLOR_BGR2GRAY)
    edgeImageArray.append(gray_e)
edgeImageArray = np.array(edgeImageArray)
# plot_image_array(edgeImageArray)


def hough_transform(edged, rho_res, theta_res, thresholdVotes, filterMultiple, thresholdPixels=0):
    rows, columns = edged.shape

    theta = np.linspace(-90.0, 0.0, np.ceil(90.0 / theta_res) + 1.0)
    theta = np.concatenate((theta, -theta[len(theta) - 2::-1]))

    # defining empty Matrix in Hough space, where x is for theta and y is x*cos(theta)+y*sin(theta)
    diagonal = np.sqrt((rows - 1) ** 2 + (columns - 1) ** 2)
    q = np.ceil(diagonal / rho_res)
    nrho = 2 * q + 1
    rho = np.linspace(-q * rho_res, q * rho_res, nrho)
    houghMatrix = np.zeros((len(rho), len(theta)))

    # Voting. Each pixel on the edge will vote for the model parameters.
    for rowId in range(rows):  # for each x in edged
        for colId in range(columns):  # for each y in edged
            if edged[rowId, colId] > thresholdPixels:  # edged has values 0 or 255 in our case

                # TODO This can be optimized using orientation (gradient at x,y)
                # for each theta we calculate rhoVal, then locate it in Hough space plane
                for thId in range(len(theta)):
                    rhoVal = colId * np.cos(theta[thId] * np.pi / 180.0) + \
                             rowId * np.sin(theta[thId] * np.pi / 180)
                    rhoIdx = np.nonzero(np.abs(rho - rhoVal) == np.min(np.abs(rho - rhoVal)))[0]
                    houghMatrix[rhoIdx[0], thId] += 1

    if filterMultiple > 0:
        clusterDiameter = filterMultiple
        values = np.transpose(np.array(np.nonzero(houghMatrix > thresholdVotes)))
        filterArray = []
        filterArray.append(0)
        totalArray = []
        for i in range(0, len(values)):
            if i in filterArray[1::]:
                continue
            tempArray = [i]
            for j in range(i + 1, len(values)):
                if j in filterArray[1::]:
                    continue
                for k in range(0, len(tempArray)):
                    if getLength(values[tempArray[k]], values[j]) < clusterDiameter:
                        filterArray.append(j)
                        tempArray.append(j)
                        break
            totalArray.append(tempArray)

        # leave the highest value in each cluster
        for i in range(0, len(totalArray)):
            for j in range(0, len(totalArray[i])):
                if j == 0:
                    highest = houghMatrix[values[totalArray[i][j]][0], values[totalArray[i][j]][1]]
                    ii = i
                    jj = j
                else:
                    if houghMatrix[values[totalArray[i][j]][0], values[totalArray[i][j]][1]] >= highest:
                        highest = houghMatrix[values[totalArray[i][j]][0], values[totalArray[i][j]][1]]
                        houghMatrix[values[totalArray[ii][jj]][0], values[totalArray[ii][jj]][1]] = 0
                        ii = i
                        jj = j
                    else:
                        houghMatrix[values[totalArray[i][j]][0], values[totalArray[i][j]][1]] = 0

    return (np.where(houghMatrix > thresholdVotes)[0] - q) * rho_res, theta[
        np.where(houghMatrix > thresholdVotes)[1]] * np.pi / 180.0

def valid_point(pt, ymax, xmax):
  '''
  @return True/False if pt is with bounds for an xmax by ymax image
  '''
  x, y = pt
  if x <= xmax and x >= 0 and y <= ymax and y >= 0:
    return True
  else:
    return False

def round_tup(tup):
  '''
  @return closest integer for each number in a point for referencing
  a particular pixel in an image
  '''
  x,y = [int(round(num)) for num in tup]
  return (x,y)


def hough_transform_gradient(edge_image, dx, dy, theta_res=1, rho_res=1):
    rows, columns = edge_image.shape
    # These are theta values in degrees
    theta = np.linspace(-90.0, 0.0, np.ceil(90.0 / theta_res) + 1.0)
    theta = np.concatenate((theta, -theta[len(theta) - 2::-1]))

    # Length of diagonal of the image denominates the space for rho values
    image_diagonal = np.sqrt((rows - 1) ** 2 + (columns - 1) ** 2)
    max_rho = np.ceil(image_diagonal / rho_res)

    rho_number = 2 * max_rho + 1
    rho = np.linspace(-max_rho * rho_res, max_rho * rho_res, rho_number)

    hough_matrix = np.zeros((len(rho), len(theta)))

    for i in range(rows):
        for j in range(columns):
            # Edges are white
            if edge_image[i, j]:

                dx_value = dx[i, j]
                dy_value = dy[i, j]

                if dx_value != 0:
                    ang = np.arctan(dy_value/dx_value)
                else:
                    if 0 < dy_value:
                        ang = np.pi / 2
                    else:
                        ang = - np.pi / 2

                rho_val = j * np.cos(ang) + i * np.sin(ang)
                rho_index = np.nonzero(np.abs(rho - rho_val) == np.min(np.abs(rho - rho_val)))[0]
                theta_degree = np.ceil((ang * 180 / np.pi) / theta_res)

                hough_matrix[rho_index[0], theta_degree] += 1

    return rho, theta, H


#rho, theta = hough_transform(edgeImageArray[6], rho_res=1, theta_res=1, thresholdVotes=30, filterMultiple=5, thresholdPixels=0)
#plot_hough_lines(rho, theta, edgeImageArray[6])


def top_n_rho_theta_pairs(ht_acc_matrix, n, rhos, thetas):
    '''
    @param hough transform accumulator matrix H (rho by theta)
    @param n pairs of rho and thetas desired
    @param ordered array of rhos represented by rows in H
    @param ordered array of thetas represented by columns in H
    @return top n rho theta pairs in H by accumulator value
    @return x,y indexes in H of top n rho theta pairs
    '''
    flat = list(set(np.hstack(ht_acc_matrix)))
    flat_sorted = sorted(flat, key = lambda n: -n)
    coords_sorted = [(np.argwhere(ht_acc_matrix == acc_value)) for acc_value in flat_sorted[0:n]]
    rho_theta = []
    x_y = []
    for coords_for_val_idx in range(0, len(coords_sorted), 1):
        coords_for_val = coords_sorted[coords_for_val_idx]
        for i in range(0, len(coords_for_val), 1):
          n,m = coords_for_val[i] # n by m matrix
          rho = rhos[n]
          theta = thetas[m]
          rho_theta.append([rho, theta])
          x_y.append([m, n]) # just to unnest and reorder coords_sorted
    return [rho_theta[0:n], x_y]

def draw_rho_theta_pairs(target_im, pairs):
  '''
  @param opencv image
  @param array of rho and theta pairs
  Has the side-effect of drawing a line corresponding to a rho theta
  pair on the image provided
  '''
  im_y_max, im_x_max = np.shape(target_im)
  for i in range(0, len(pairs), 1):
    point = pairs[i]
    rho = point[0]
    theta = point[1] * np.pi / 180 # degrees to radians
    # y = mx + b form
    m = -np.cos(theta) / np.sin(theta)
    b = rho / np.sin(theta)
    # possible intersections on image edges
    left = (0, b)
    right = (im_x_max, im_x_max * m + b)
    top = (-b / m, 0)
    bottom = ((im_y_max - b) / m, im_y_max)

    pts = [pt for pt in [left, right, top, bottom] if valid_point(pt, im_y_max, im_x_max)]
    if len(pts) == 2:
      cv2.line(target_im, round_tup(pts[0]), round_tup(pts[1]), (0,0,255), 1)


image_paths = glob.glob("images/*.jpg")
image = cv2.imread(image_paths[6])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

blurred = cv2.GaussianBlur(gray, (3, 3), 0)
edge_image = auto_canny(blurred)

# Hough transform on test image
rhos, thetas, H = hough_transform_gradient(edge_image, sobel_x, sobel_y)

rho_theta_pairs, x_y_pairs = top_n_rho_theta_pairs(H, 10, rhos, thetas)
draw_rho_theta_pairs(edge_image, rho_theta_pairs)

plt.imshow(edge_image)
plt.show()

test = 2

# plt.figure()
 # plt.imshow(gray, cmap="gray")
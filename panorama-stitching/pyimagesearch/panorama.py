# import the necessary packages
import numpy as np
import imutils
import cv2
import random

print(cv2.__version__)

class Stitcher:
    def __init__(self):
        # determine if we are using OpenCV v3.X
        self.isv3 = imutils.is_cv3()

    def stitch(self, images, ratio=0.75, reprojThresh=4.0,
               showMatches=False):
        # unpack the images, then detect keypoints and extract
        # local invariant descriptors from them
        (imageB, imageA) = images
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)

        # match features between the two images
        M = self.matchKeypoints(kpsA, kpsB,
                                featuresA, featuresB, ratio, reprojThresh)

        # if the match is None, then there aren't enough matched
        # keypoints to create a panorama
        if M is None:
            return None

        # otherwise, apply a perspective warp to stitch the images
        # together
        (matches, H, status) = M

        result = cv2.warpPerspective(imageA, H,
                                     (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))

        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

        # check to see if the keypoint matches should be visualized
        if showMatches:
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches,
                                   status)

            # return a tuple of the stitched image and the
            # visualization
            return (result, vis)

        # return the stitched image
        return result

    def detectAndDescribe(self, image):
        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # check to see if we are using OpenCV 3.X
        if self.isv3:
            # detect and extract features from the image
            descriptor = cv2.xfeatures2d.SIFT_create()
            (kps, features) = descriptor.detectAndCompute(image, None)

        # otherwise, we are using OpenCV 2.4.X
        else:
            # detect keypoints in the image
            detector = cv2.FeatureDetector_create("SIFT")
            kps = detector.detect(gray)

            # extract features from the image
            extractor = cv2.DescriptorExtractor_create("SIFT")
            (kps, features) = extractor.compute(gray, kps)

        # convert the keypoints from KeyPoint objects to NumPy
        # arrays
        kps = np.float32([kp.pt for kp in kps])

        # return a tuple of keypoints and features
        return (kps, features)

    # Should compute H from any number of points (min4)
    def computeHomographicMatrix(self, p1, p2):
        A = []
        for i in range(0, len(p1)):
            x, y = p1[i][0], p1[i][1]
            u, v = p2[i][0], p2[i][1]
            A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
            A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
        A = np.asarray(A)
        U, S, Vh = np.linalg.svd(A)
        L = Vh[-1, :] / Vh[-1, -1]
        H = L.reshape(3, 3)
        return H

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
                       ratio, reprojThresh):
        # compute the raw matches and initialize the list of actual
        # matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []

        # loop over the raw matches
        for m in rawMatches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # computing a homography requires at least 4 matches
        if len(matches) > 4:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            #RANSAC homographyc computation
            i = 0
            maxInliers = 0
            bestH = 0
            inlierDistanceTreshold = 10

            while i < 50:
                # select 4 random points and compute homographic matrix
                randA = []
                randB = []
                for x in range(4):
                    index = random.randint(0, ptsA.shape[0] - 1)
                    randA.append(ptsA[index])
                    randB.append(ptsB[index])
                H = self.computeHomographicMatrix(randA, randB)

                inliers = 0

                # compute projections of the matches and distance
                for index in range(len(ptsA)):
                    homogenousPoint = np.array((ptsA[index][0], ptsA[index][1], 1)).reshape((3, 1))
                    projectedPoint = H.dot(homogenousPoint)

                    # Cartezian coord of projection
                    projectedPointX = projectedPoint[0] / projectedPoint[2]
                    projectedPointY = projectedPoint[1] / projectedPoint[2]

                    # SSD. Projected is a match?
                    ssd = (ptsB[index][0] - projectedPointX) ** 2 + (ptsB[index][1] - projectedPointY) ** 2

                    if ssd < inlierDistanceTreshold:
                        inliers = inliers + 1

                if inliers > maxInliers:
                    maxInliers = inliers
                    bestH = H

                i = i+1

            # Using the best homography get all inliers and fit the final homography
            fittedInliersA = []
            fittedInliersB = []

            status = [0] * len(ptsA)

            for index in range(len(ptsA)):
                homogenousPoint = np.array((ptsA[index][0], ptsA[index][1], 1)).reshape((3, 1))
                projectedPoint = bestH.dot(homogenousPoint)

                # Cartesian coord of projection
                projectedPointX = projectedPoint[0] / projectedPoint[2]
                projectedPointY = projectedPoint[1] / projectedPoint[2]

                # SSD
                ssd = (ptsB[index][0] - projectedPointX) ** 2 + (ptsB[index][1] - projectedPointY) ** 2

                if ssd < inlierDistanceTreshold:
                    fittedInliersA.append(ptsA[index])
                    fittedInliersB.append(ptsB[index])
                    status[index] = 1

            bestBestH = self.computeHomographicMatrix(fittedInliersA, fittedInliersB)

            # return the matches along with the homograpy matrix
            # and status of each matched point
            return (matches, bestBestH, status)

        # otherwise, no homograpy could be computed
        return None

    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # initialize the output visualization image
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        # loop over the matches
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # only process the match if the keypoint was successfully
            # matched
            if s == 1:
                # draw the match
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        # return the visualization
        return vis

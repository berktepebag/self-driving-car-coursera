import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

queryImg = cv.imread('query-image.png')
testImg = cv.imread('test-image.png')

# Initiate the ORB detector
orb = cv.ORB_create()

# Find the keypoints and descriptors of each image
kp1, des1 = orb.detectAndCompute(queryImg,None)
kp2, des2 = orb.detectAndCompute(testImg,None)

# Create BFMatcher
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck = True)

# Match Descriptors
matches = bf.match(des1,des2)

# Sort them in the order of their distance
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches
result = cv.drawMatches(queryImg,kp1,testImg,kp2,matches[:10],None, flags=2)

plt.imshow(result)

plt.show()
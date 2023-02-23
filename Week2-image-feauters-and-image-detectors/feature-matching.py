import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

queryImg = cv.imread('query-image')
testImg = cv.imread('test-image')

# Init SIFT detc.
sift = cv.xfeatures2d.SIFT_create()

# find the keypoints and descriptors
kp1, des1 = sift.detectAndCompute(queryImg, None)
kp2, des2 = sift.detectAndCompute(testImg, None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test
good = []
distance_ratio = 0.7

for m,n in matches:
    if m.distance / n.distance < distance_ratio:
        good.append(m)

MIN_MATCH_COUNT = 10

# If enough matches are found, we extract the locations of matched keypoints in both the images.
if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    h,w,d = queryImg.shape
    pts = np.float32( [0,0],[0,h-1],[w-1,h-1],[w-1,0]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts,M)

    queryImg = cv.polylines(testImg,[np.int32(dst)],True,255,3, cv.LINE_AA)

else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None




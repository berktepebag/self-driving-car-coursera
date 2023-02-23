import numpy as np
import cv2 as cv

filename = 'chessboard.png'
img = cv.imread(filename)
cv.imshow('dst',img)
if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()
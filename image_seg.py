import cv2
import numpy as np
from IPython.display import Image, display
from matplotlib import pyplot as plt

#Image loading
img = cv2.imread("real_images/images/000000.jpg")
# Show image
cv2.imshow('Original', img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray",gray)


ret, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
cv2.imshow("Binary image", bin_img)


kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
bin_img = cv2.morphologyEx(img, 
                           cv2.MORPH_OPEN,
                           kernel,
                           iterations=2)
cv2.imshow("bin_img", bin_img)

# sure background area
sure_bg = cv2.dilate(bin_img, kernel, iterations=3)
cv2.imshow("Sure Background",sure_bg)

cv2.imshow("Background prova",img - sure_bg)

# # Distance transform
# dist = cv2.distanceTransform(bin_img, cv2.DIST_L2, 5)
# cv2.imshow("Distance transform", dist)

 
# #foreground area
# ret, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, cv2.THRESH_BINARY)
# sure_fg = sure_fg.astype(np.uint8)  
# cv2.imshow("Sure foreground", sure_fg)
 
# unknown area
# unknown = cv2.subtract(sure_bg, sure_fg)
# cv2.imshow("Unknown",unknown)
 
plt.show()

cv2.waitKey(0) 
cv2.destroyAllWindows() 
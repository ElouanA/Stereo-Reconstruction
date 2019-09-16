import numpy as np
import cv2
from matplotlib import pyplot as plt

min_disp=0
num_disp=64
window_size=3
imgL = cv2.imread('IML.jpg',0)
imgR = cv2.imread('IMR.jpg',0)
height,width=imgL.shape
kernel= np.ones((3,3),np.uint8)
stereo = cv2.StereoSGBM_create(minDisparity = min_disp,numDisparities = num_disp,blockSize = window_size,uniquenessRatio = 10,speckleWindowSize = 100,speckleRange = 32,disp12MaxDiff = 5,P1 = 8*3*window_size**2,P2 = 32*3*window_size**2)
stereoR=cv2.ximgproc.createRightMatcher(stereo)
lmbda = 8000
sigma = 1.8
visual_multiplier = 1.0
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)
disp= stereo.compute(imgL,imgR).astype(np.float32)
dispL= disp
dispR= stereoR.compute(imgL,imgR)
dispL= np.int16(dispL)
dispR= np.int16(dispR)
filteredImg= wls_filter.filter(dispL,imgL,None,dispR)
filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
filteredImg = np.uint8(filteredImg)
closing= cv2.morphologyEx(disp,cv2.MORPH_CLOSE, kernel) 
dispc= (closing-closing.min())*255
dispC= dispc.astype(np.uint8)
disp_Color= cv2.applyColorMap(dispC,cv2.COLORMAP_OCEAN)
filt_Color= cv2.applyColorMap(filteredImg,cv2.COLORMAP_OCEAN)
plt.imshow(filteredImg)
cv2.imwrite('filteredimg.png',filteredImg)
plt.show()
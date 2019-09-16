import cv2
import matplotlib.pyplot as plt
import time
import numpy as np

CAMERA_WIDTH=640
CAMERA_HEIGHT=360
left = cv2.VideoCapture(0)
left.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
left.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
left.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
left.set(cv2.CAP_PROP_FPS,60)
right = cv2.VideoCapture(1)
right.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
right.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
right.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
right.set(cv2.CAP_PROP_FPS,60)
print("début de calibration, filmer le damier avec différents angles de vue, appuyer sur P pour prendre une paire du damier, appuyer sur Q pour arréter la calibration ")
board_w = 6
board_h = 9
criteria =(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
criteria_stereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((board_h*board_w,3), np.float32)
objp[:,:2] = np.mgrid[0:board_h,0:board_w].T.reshape(-1,2)
objpoints= []
imgpointsR= [] 
imgpointsL= []
t=0
while (True):
    
    if not(right.grab() and left.grab()):
        print("no frame")
    else: 
        _,leftFrame=left.retrieve()
        _,rightFrame=right.retrieve()
        leftFrame=cv2.cvtColor(leftFrame,cv2.COLOR_BGR2GRAY)
        rightFrame=cv2.cvtColor(rightFrame,cv2.COLOR_BGR2GRAY)
        if cv2.waitKey(1) & 0xFF == ord('p'):  
            retR, cornersR = cv2.findChessboardCorners(rightFrame,(board_h,board_w),None)  
            retL, cornersL = cv2.findChessboardCorners(leftFrame,(board_h,board_w),None) 
            if (True == retR) & (True == retL):
                print('coins trouvés sur la paire '+str(t))
                objpoints.append(objp)
                cv2.cornerSubPix(rightFrame,cornersR,(11,11),(-1,-1),criteria)
                cv2.cornerSubPix(leftFrame,cornersL,(11,11),(-1,-1),criteria)
                imgpointsR.append(cornersR)
                imgpointsL.append(cornersL)
                cv2.imwrite('rightframe.jpg',rightFrame)
                cv2.imwrite('leftframe.jpg',leftFrame)
                cv2.drawChessboardCorners(rightFrame,(board_h,board_w),cornersR,retR)
                cv2.drawChessboardCorners(leftFrame,(board_h,board_w),cornersL,retL)
                
                t+=1
            if (True == retR) & (True == retL):
                time.sleep(0.50)
        cv2.imshow('left',leftFrame)
        cv2.imshow('right',rightFrame)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break     
print("Fin de la prise de vue du damier de calibration, début du calcul des données de calibration")

hR,wR= rightFrame.shape[:2]
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints,imgpointsR,(hR,wR),None,None)
OmtxR, roiR= cv2.getOptimalNewCameraMatrix(mtxR,distR,(wR,hR),1,(wR,hR))
hL,wL= leftFrame.shape[:2]
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints,imgpointsL,(hL,wL),None,None)
OmtxL, roiL= cv2.getOptimalNewCameraMatrix(mtxL,distL,(wL,hL),1,(wL,hL)) 
flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC
retS, MLS, dLS, MRS, dRS, R, T, E, F= cv2.stereoCalibrate(objpoints,imgpointsL,imgpointsR,mtxL,distL,mtxR,distR,(hR,wR),criteria_stereo,flags)
rectify_scale= 1
RL, RR, PL, PR, Q, roiL, roiR= cv2.stereoRectify(MLS, dLS, MRS, dRS,(hR,wR), R, T,rectify_scale,(0,0))  
Left_Stereo_Map= cv2.initUndistortRectifyMap(MLS, dLS, RL, PL,(wR,hR), cv2.CV_16SC2)   
Right_Stereo_Map= cv2.initUndistortRectifyMap(MRS, dRS, RR, PR,(wR,hR), cv2.CV_16SC2)
xd,yd,wd,hd = roiR
xg,yg,wg,hg = roiL
print(roiR) 
print(roiL)
x=np.max([xd,xg])
y=np.max([yg,yd])
w=np.min([wd,wg])
h=np.min([hd,hg])
min_disp=0
num_disp=32
window_size=7
stereo = cv2.StereoSGBM_create(minDisparity = min_disp,numDisparities = num_disp,blockSize = window_size,uniquenessRatio = 10,speckleWindowSize = 100,speckleRange = 32,disp12MaxDiff = 5,P1 = 8*3*window_size**2,P2 = 32*3*window_size**2)
stereoR=cv2.ximgproc.createRightMatcher(stereo)
lmbda = 8000
sigma = 1.5
visual_multiplier = 1.0
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)
kernel= np.ones((3,3),np.uint8)
leftFrame=cv2.imread('leftframe.jpg',0)
rightFrame=cv2.imread('rightFrame.jpg',0)
leftFrame= cv2.remap(leftFrame,Left_Stereo_Map[0],Left_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
rightFrame= cv2.remap(rightFrame,Right_Stereo_Map[0],Right_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
img1 = leftFrame[x:x+w,y:y+h]
img2 = rightFrame[x:x+w, y:y+h]
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)
good = []
pts1 = []
pts2 = []
for i,(m,n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)
pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

def drawlines(img1,img2,lines,pts1,pts2):
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)
plt.subplot(121),plt.imshow(img5)
plt.subplot(122),plt.imshow(img3)
plt.show()
print("Fin du calcul des données de calibration, début de la visualisation des photos undistorted et de la disparité, appuyer sur P pour enregistrer le nuage de points")

def exportPointCloud(list3d,name,format):
    if (format=='ply'):
        fic=open(name + '.ply','w') 
        fic.write('ply\n'+'format ascii 1.0\n'+'comment VTK generated PLY File\n')
        fic.write('obj_info vtkPolyData points and polygons: vtk4.0\n'+'element vertex ' + str(len(list3d)) + '\n')
        fic.write('property float x\n'+'property float y\n'+'property float z\n'+'element face 0\n')
        fic.write( 'property list uchar int vertex_indices\n'+ 'end_header\n')
        for i in range (0,len(list3d)):
            fic.write(str(list3d[i][0]) + ' ' + str(list3d[i][1]) + ' ' + str(list3d[i][2]) + ' ' + '\n')
        fic.close()
        fic=open(name + '.ply','r')
        print (str(fic))
        print('export effectué sous le format ' + format + ' avec le nom ' + name)
    elif (format=='obj'):
        fic=open(name + '.obj','w') 
        fic.write('# OBJ file\n')
        for i in range (0,len(list3d)):
            fic.write('v ' + str(list3d[i][0]) + ' '+ str(list3d[i][1]) + ' ' + str(list3d[i][2]) + '\n')
        for i in range (width,len(list3d)-width):
            if (list3d[i][0]!=0) & (list3d[i][1]!=0) & (list3d[i][2]!=0) & (list3d[i+1][0]!=0) & (list3d[i+1][1]!=0) & (list3d[i+1][2]!=0)& (list3d[i+width][0]!=0) & (list3d[i+width][1]!=0) & (list3d[i+width][2]!=0):
                fic.write('f ' + str(i+1)+ ' ' +str(i-width + 2)+' ' + str(i+2)+ '\n')
            if (list3d[i][0]!=0) & (list3d[i][1]!=0) & (list3d[i][2]!=0) & (list3d[i+1][0]!=0) & (list3d[i+1][1]!=0) & (list3d[i+1][2]!=0)& (list3d[i-width+1][0]!=0) & (list3d[i-width+1][1]!=0) & (list3d[i-width+1][2]!=0):
                fic.write('f ' + str(i+1) +' ' + str(i+2) +' '+ str(i + width + 1)+ '\n')
        fic.close()
        fic=open(name + '.obj','r')
        print (str(fic))
        print('export effectué sous le format ' + format + ' avec le nom ' + name)
    else: 
        print('format incompatible, utilisez le format .obj ou .ply')
    return

def MatToList3D(mat): 
    list3d=[]
    for i in range (0,len(mat)): 
        for j in range (0,len(mat[0])):
            inf = float("inf")
            if (abs(mat[i,j,0])!= inf) & (abs(mat[i,j,1])!= inf) & (abs(mat[i,j,2])!= inf):
                list3d.append(mat[i,j])
            else: 
                list3d.append([0,0,0])
    return(np.array(list3d))
    
while (True):
    if not(right.grab() and left.grab()):
        print("no frame")
    else: 
        _,leftFrame=left.retrieve()
        _,rightFrame=right.retrieve()
        leftFrame= cv2.remap(leftFrame,Left_Stereo_Map[0],Left_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        rightFrame= cv2.remap(rightFrame,Right_Stereo_Map[0],Right_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        leftFrame = leftFrame[x:x+w, y:y+h]
        rightFrame = rightFrame[x:x+w,y:y+h]
        grayR= cv2.cvtColor(rightFrame,cv2.COLOR_BGR2GRAY)
        grayL= cv2.cvtColor(leftFrame,cv2.COLOR_BGR2GRAY)
        disp= stereo.compute(grayL,grayR)
        dispL= disp
        dispR= stereoR.compute(grayR,grayL)
        dispL= np.int16(dispL)
        dispR= np.int16(dispR)
        closing= cv2.morphologyEx(dispL,cv2.MORPH_CLOSE, kernel)
        closing= cv2.morphologyEx(dispR,cv2.MORPH_CLOSE, kernel)
        filteredImg= wls_filter.filter(dispL,grayL,None,dispR)
        filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
        filteredImg = np.uint8(filteredImg)
        closing= cv2.morphologyEx(disp,cv2.MORPH_CLOSE, kernel)
        cv2.imshow('disparité',filteredImg)
        cv2.imshow('left',leftFrame)
        cv2.imshow('right',rightFrame)
        if cv2.waitKey(1) & 0xFF == ord('p'):
            _3DImage=np.array([])
            _3DImage=cv2.reprojectImageTo3D(filteredImg,Q,_3DImage,0)
            list3d=MatToList3D(_3DImage)
            exportPointCloud(list3d,'dst','ply')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break     

   

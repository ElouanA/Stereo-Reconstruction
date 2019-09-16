#On importe l'ensemble des bibliothèques dont nous avons besoin
import numpy as np
import cv2
import matplotlib.pyplot as plt
import EpipolarDisplay

''' Définition des variables globales'''
index=19
#Nombre de prises de vue du patern de calibration
n_boards = 28
#Nombre de coins présents dans le patern de calibration
board_w = 6
board_h = 9
#Paramètres pour calculer la disparité, jouer sur min_disp et num_disp permet de jouer sur la profondeur de focus de la carte de disparité
min_disp=0
num_disp=144
window_size=7
#On définit une matrice 3*3 contenant des uns, utilisées pour le filtrage de la carte de disparité 
kernel= np.ones((3,3),np.uint8)

#Définition des critères de terminaison
criteria =(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
criteria_stereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

#On créée une matrice contenant les coordonnés des coins dans le système de coordonné du patern: 1 unité correspond au côté d'un carré sur le patern
objp = np.zeros((board_h*board_w,3), np.float32)
objp[:,:2] = np.mgrid[0:board_h,0:board_w].T.reshape(-1,2)

'''Début de la calibration'''

#On créée une matrice qui contiendra les vecteurs des points trouvés sur le damier, et une matrice objpoints qui contiendra objp autant de fois que l'on aura de photos du patern de calibration
objpoints= []
imgpointsR= [] 
imgpointsL= []


#On parcourt les images des différentes prises de vue du patern, on localise les coins et on ajoute les coordonés à ipts qui est la localisation des coins vus par la caméra
for i in range(1,n_boards+1):   

    t= str(i)
    #On charge les images en grayscale
    ChessImaR= cv2.imread('D ('+t+').jpg',0)   

    ChessImaL= cv2.imread('G ('+t+').jpg',0) 
    #On cherche les coins du patern sur l'image
    retR, cornersR = cv2.findChessboardCorners(ChessImaR,(board_h,board_w),None)  
    retL, cornersL = cv2.findChessboardCorners(ChessImaL,(board_h,board_w),None) 
     
    if (True == retR) & (True == retL):
        print('coins trouvés sur la paire '+t) #On affiche le succès de la recherche des coins 
        #En cas de succès de la recherche des coins on ajoute les coordonnés dans le système de patern à la liste opts (Vecteur de vecteurs) 
        objpoints.append(objp)
        #En cas de succès on affine les coordonées des coins à l'aide de la fonction cv2.cornercubpix puis on ajoute les coordonnés dans le système du patern à la liste ipts (Vecteur de vecteurs)
        cv2.cornerSubPix(ChessImaR,cornersR,(11,11),(-1,-1),criteria)

        cv2.cornerSubPix(ChessImaL,cornersL,(11,11),(-1,-1),criteria)
        imgpointsR.append(cornersR)

        imgpointsL.append(cornersL)
    else: 
        print('coins non trouvés sur la paire ' + t) #On affiche l'echec de la recherche des coins 
        

#On réalise la calibration des caméras individuelles

#Caméra droite
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints,imgpointsR,ChessImaR.shape[::-1],None,None)
hR,wR= ChessImaR.shape[:2]
OmtxR, roiR= cv2.getOptimalNewCameraMatrix(mtxR,distR,(wR,hR),1,(wR,hR))

#Caméra gauche
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints,imgpointsL,ChessImaL.shape[::-1],None,None)
hL,wL= ChessImaL.shape[:2]
OmtxL, roiL= cv2.getOptimalNewCameraMatrix(mtxL,distL,(wL,hL),1,(wL,hL))

#On réalise la calibration stéréo 
flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC
retS, MLS, dLS, MRS, dRS, R, T, E, F= cv2.stereoCalibrate(objpoints,imgpointsL,imgpointsR,mtxL,distL,mtxR,distR,ChessImaR.shape[::-1],criteria_stereo,flags)
print(dRS)
print(dLS)
'''Fin de la calibration ''' 
'''Début de la rectification des images'''

# On fait appel à la fonction stereoRectify et initUndistortRectifyMap pour calculer les rectifications à appliquer aux deux images 
rectify_scale= 1 #Ce masque indique l'echelle de fermeture des pixels blancs sur l'image, 1 indique qu'on ne crop pas l'image, 0 indique un crop maximal 
RL, RR, PL, PR, Q, roiL, roiR= cv2.stereoRectify(MLS, dLS, MRS, dRS,ChessImaR.shape[::-1], R, T,rectify_scale,(0,0))  
Left_Stereo_Map= cv2.initUndistortRectifyMap(MLS, dLS, RL, PL,ChessImaR.shape[::-1], cv2.CV_16SC2)   
Right_Stereo_Map= cv2.initUndistortRectifyMap(MRS, dRS, RR, PR,ChessImaR.shape[::-1], cv2.CV_16SC2)

# On charge la paire d'images sur laquelle on va effectuer la reconstruction

frameR= cv2.imread('D ('+str(index)+').jpg')
frameL= cv2.imread('G ('+str(index)+').jpg')

#On utilise la fonction remap pour recitifier la paire Stereo et on sauvegarde les photos corrigées 

newimageg= cv2.remap(frameL,Left_Stereo_Map[0],Left_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
newimaged= cv2.remap(frameR,Right_Stereo_Map[0],Right_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

#On calcule l'intersection de roiR et roiL et on crop l'image selon ce rectangle (ROI=region of interest=ensemble des pixels valides) 
print(roiR)
print(roiL)
xd,yd,wd,hd = roiR
xg,yg,wg,hg = roiL
x=np.max([xd,xg])
y=np.max([yg,yd])
w=np.min([wd,wg])
h=np.min([hd,hg])
newimageg = newimageg[y:y+h, x:x+w]
newimaged = newimaged[y:y+h, x:x+w]

cv2.imwrite('newimageG' + str(index)+'.jpg',newimageg)
cv2.imwrite('newimageD'+str(index)+'.jpg',newimaged)


#On display les lignes épipolaires pour vérifier que la rectification s'est bien effectuée, normalement les lignes épipolaires sont horizontales et de même coordonnée Y

EpipolarDisplay.epipolarlines(index)

'''Fin de la rectification'''
'''Début du calcul de la carte de disparité'''

#On instance un objet StereoSGBM avec l'ensemble des paramètres et les objets intermédiaires nécéssaires au filtrage de la carte de disparités 
stereo = cv2.StereoSGBM_create(minDisparity = min_disp,numDisparities = num_disp,blockSize = window_size,uniquenessRatio = 10,speckleWindowSize = 100,speckleRange = 32,disp12MaxDiff = 5,P1 = 8*3*window_size**2,P2 = 32*3*window_size**2)
stereoR=cv2.ximgproc.createRightMatcher(stereo)

#Les paramètres sont les paramètres de filtrage de la caméra Stereo, on peut jouer sur ces paramètres pour permettre d'obtenir une carte de disparité plus précise, ou plus 
lmbda = 8000
sigma = 1.5
visual_multiplier = 1.0
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)
#On les convertit en grayscale pour calculer ensuite la disparité 

grayR= cv2.cvtColor(newimaged,cv2.COLOR_BGR2GRAY)
grayL= cv2.cvtColor(newimageg,cv2.COLOR_BGR2GRAY)

#On calcule la carte de disparité à partir des deux images et de l'objet de l'instance StereoSGBM

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

#On a donc calculé la carte de disparité, on la display et l'enregistre pour permettre de mieux la visualiser
plt.imshow(filteredImg)
cv2.imwrite('filtereImg.png',filteredImg)
plt.show()
'''Fin du calcul de la carte de disparité'''
'''Début de la reconstruction 3D'''

#On va désormais passer de la carte de disparité aux coordonnées réelles dans l'espace, pour cela on utilise la fonction reprojectImageTo3D qui utilise la matrice Q calculée lors de la calibration. On rapelle que le système d'unité est défini tel que le côté d'un carré sur le patern de calibration soit l'unité. 

_3DImage=np.array([])
_3DImage=cv2.reprojectImageTo3D(filteredImg,Q,_3DImage,0)
height, width,dim=_3DImage.shape

#Cette fonction permet simplement de transformer une matrice de coordonnées de points (L,M,3) en liste (L*M,3) de coordonnées des points, et on filtre les points ayants pour coordonées +/- infini 

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
    
list3d=MatToList3D(_3DImage)

# On exporte une liste de coordonnées de points (N,3) sous la forme d'une fichier .ply ou .obj qui permet de visualiser le nuage de point dans n'importe quel logiciel de traitement 3D

'''TODO: ajouter le calcul des normals dans l'export en .obj, j'ai essayé en utilisant le produit vectoriel connaissant trois points du plans mais ca ne fonctionne pas, sinon il existe des fonctions OpenCV qui calculent les normales d'un plan '''

def exportPointCloud(list3d,name,format):
    if (format=='ply'):
        #On ouvre un fichier
        fic=open(name + '.ply','w') 
        #On écrit le header pour le format .ply
        fic.write('ply\n'+'format ascii 1.0\n'+'comment VTK generated PLY File\n')
        fic.write('obj_info vtkPolyData points and polygons: vtk4.0\n'+'element vertex ' + str(len(list3d)) + '\n')
        fic.write('property float x\n'+'property float y\n'+'property float z\n'+'element face 0\n')
        fic.write( 'property list uchar int vertex_indices\n'+ 'end_header\n')
        #On ajoute les coordonnés des points
        for i in range (0,len(list3d)):
            fic.write(str(list3d[i][0]) + ' ' + str(list3d[i][1]) + ' ' + str(list3d[i][2]) + ' ' + '\n')
        #On ferme le fichier
        fic.close()
        print('export effectué sous le format ' + format + ' avec le nom ' + name)
    elif (format=='obj'):
        #On ouvre le fichier 
        fic=open(name + '.obj','w') 
        #On écrit le header pour le format .obj
        fic.write('# OBJ file\n')
        #On commence par ajouter les coordonées des points
        for i in range (0,len(list3d)):
            fic.write('v ' + str(list3d[i][0]) + ' '+ str(list3d[i][1]) + ' ' + str(list3d[i][2]) + '\n')
        #Puis on ajoute les faces qui sont des triangles composés des points adjacents
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

exportPointCloud(list3d,'dst','ply')

#Cette fonction calcule le vecteur normal à un plan dont on connait trois points différents à l'aide du produit vectoriel 

def normals(a,b,c): 
    ab=[b[0]-a[0],b[1]-a[1],b[2]-a[2]]
    ac=[c[0]-a[0],c[1]-a[1],c[2]-a[2]]
    vectprod=[ab[1]*ac[2]-ab[2]*ac[1],ab[0]*ac[2]-ab[2]*ac[0],ab[1]*ac[2]-ab[2]*ac[1]]
    return(vectprod) 
    
#Cette fonction permet de calculer la matrice de rotation (3*3) dans l'espace à partir d'un axe (x,y,z) et d'un angle de rotation theta 

def rotation_matrix(axis, theta):
    axis = axis/np.sqrt(np.dot(axis, axis))
    a = np.cos(theta/2.)
    b, c, d = -axis*np.sin(theta/2.)
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],

                  [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],

                  [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])

#Cette fonction permet de ne garder seulement 1/offset des points d'une liste, les points conservés ont des indices espacés de offset

def samplepoints(list,offset):
    tmp=[]
    for i in range (0,int(len(list)/offset)):
        tmp.append(list[i*offset])
    return(np.array(tmp))
  
#La suite du code permet de tester l'implémentation de l'algorithme ICP
#On commence par effectuer une copie de la reconstruction en nuage de points et on va la translater et la rotationner aléatoirement (on peut jouer sur l'amplitude de cette rotation et de la translation avec les échelles translation et rotation) 
#On va égalment bruiter le nuage de points, on peut régler l'amplitude avec sigma.   
     
B=np.copy(list3d)
N=len(B)
dim = 3           
noise_sigma = .0000                      
translation =2
rotation = 3.14 /2
t = np.random.rand(dim)*translation
B += t
R = rotation_matrix(np.random.rand(dim), np.random.rand()*rotation)
B = np.dot(R, B.T).T
B += np.random.randn(N, dim) * noise_sigma
exportPointCloud(B,'src','ply')

# On applique ensuite l'algorithme ICP avec comme référence list3D et comme source B, l'algorithme renvoie une matrice de transformation en coordonées homogènes donc il est nécéssaire de créer ensuite un nuage vide de dimension 4 
'''print("ICP commence")
T, distances, iterations = ICP.icp(list3d, B, tolerance=0.0000001)
print("ICP FInie")
#On crée un nuage vide de dimension 4 et on copie les coordonées de B dans les 3 premières dimensions 
C = np.ones((N, 4))
C[:,0:3] = np.copy(B)
#On applique la transformation avec le résultat de l'ICP 
C = np.dot(T, C.T).T
C=samplepoints(C,10)
exportPointCloud(C,'icp','ply')
print("icp commence")'''

#La partie suivante est un test d'un package OpenCV qui permet une registration de surface sans correspondance supposée, mais je n'ai pas trouvé d'exemple de code. Voir le readme 

'''icp=cv2.ppf_match_3d_ICP(40, 0.05, 2.5, 8)
retval,residual,pose=icp.registerModelToScene(list3d.astype(np.float32),B.astype(np.float32))
print("icp finie")
print(residual,pose,retval)
C = np.ones((N, 4))
C[:,0:3] = np.copy(B)
C = np.dot(pose, C.T).T
exportPointCloud(C,'icp','ply')'''







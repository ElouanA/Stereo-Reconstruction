import numpy as np
from sklearn.neighbors import NearestNeighbors
import time
def best_fit_transform(A, B):

    #assert A.shape == B.shape
    #On récupère la dimension
    m = A.shape[1]

    #On centre les données 

    centroid_A = np.mean(A, axis=0)

    centroid_B = np.mean(B, axis=0)

    AA = A - centroid_A

    BB = B - centroid_B

    # rotation matrix

    H = np.dot(AA.T, BB)

    U, S, Vt = np.linalg.svd(H)

    R = np.dot(Vt.T, U.T)

    # special reflection case

    if np.linalg.det(R) < 0:

       Vt[m-1,:] *= -1

       R = np.dot(Vt.T, U.T)

    # translation

    t = centroid_B.T - np.dot(R,centroid_A.T)

    T = np.identity(m+1)

    T[:m, :m] = R

    T[:m, m] = t

    return T, R, t

def nearest_neighbor(src, dst):

    #assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)

    neigh.fit(dst)

    distances, indices = neigh.kneighbors(src, return_distance=True)

    return distances.ravel(), indices.ravel()

def icp(A, B, init_pose=None, max_iterations=20, tolerance=0.0001):
    
    #assert A.shape == B.shape

    m = A.shape[1]

    src = np.ones((m+1,A.shape[0]))

    dst = np.ones((m+1,B.shape[0]))

    src[:m,:] = np.copy(A.T)

    dst[:m,:] = np.copy(B.T)

    if init_pose is not None:

        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):

        distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)

        T,_,_ = best_fit_transform(src[:m,:].T, dst[:m,indices].T)

        src = np.dot(T, src)

        mean_error = np.mean(distances)

        if np.abs(prev_error - mean_error) < tolerance:

            break

        prev_error = mean_error

    # calculate final transformation

    T,_,_ = best_fit_transform(A, src[:m,:].T)

    return T, distances, i


N = 1500                         
num_tests = 100                       

dim = 3                          

noise_sigma = 0.01                        

translation = 3                       

rotation = .5                             




def rotation_matrix(axis, theta):

    axis = axis/np.sqrt(np.dot(axis, axis))

    a = np.cos(theta/2.)

    b, c, d = -axis*np.sin(theta/2.)

    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],

                  [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],

                  [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])

def test_icp():

    A = np.random.rand(N,dim)

    total_time = 0
    
    for i in range (0,num_tests):
        B = np.copy(A)
        t = np.random.rand(dim)*translation
    
        B += t
            
        R = rotation_matrix(np.random.rand(dim), np.random.rand() * rotation)
    
        B = np.dot(R, B.T).T
    
        B += np.random.randn(N, dim) * noise_sigma
    
        np.random.shuffle(B)
    
        start = time.time()
    
        T, distances, iterations = icp(B, A, tolerance=0.00000001)
    
        total_time += time.time() - start
    
        C = np.ones((N, 4))
    
        C[:,0:3] = np.copy(B)
    
        C = np.dot(T, C.T).T
    
        '''assert np.mean(distances) < 6*noise_sigma               
    
        assert np.allclose(T[0:3,0:3].T, R, atol=6*noise_sigma)     
    
        assert np.allclose(-T[0:3,3], t, atol=6*noise_sigma) ''' 

    print('icp time: {:.3}'.format(total_time/num_tests))
    return (A,B,C)

def pointCloudtoPly(A,name):
    fic=open(name + '.ply','w') 
    fic.write('ply\n'+'format ascii 1.0\n'+'comment VTK generated PLY File\n'+'obj_info vtkPolyData points and polygons: vtk4.0\n'+'element vertex ' + str(len(A)) + '\n'+'property float x\n'+'property float y\n'+'property float z\n'+'element face 0\n'+ 'property list uchar int vertex_indices\n'+ 'end_header\n')
    for i in range (0,len(A)):
        fic.write(str(A[i][0]) + ' ')
        fic.write(str(A[i][1]) + ' ')
        fic.write(str(A[i][2]) + ' ')
        fic.write('\n')
    fic.close()
    fic=open(name + '.ply','r')
    print (str(fic))

if __name__ == "__main__":
    A,B,C = test_icp()
    pointCloudtoPly(A,'vueA')
    pointCloudtoPly(B,'vueB')
    pointCloudtoPly(C,'vueC')
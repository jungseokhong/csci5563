import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors

def MatchSIFT(loc1, des1, loc2, des2):
    """
    Find the matches of SIFT features between two images
    
    Parameters
    ----------
    loc1 : ndarray of shape (n1, 2)
        Keypoint locations in image 1
    des1 : ndarray of shape (n1, 128)
        SIFT descriptors of the keypoints image 1
    loc2 : ndarray of shape (n2, 2)
        Keypoint locations in image 2
    des2 : ndarray of shape (n2, 128)
        SIFT descriptors of the keypoints image 2

    Returns
    -------
    x1 : ndarray of shape (n, 2)
        Matched keypoint locations in image 1
    x2 : ndarray of shape (n, 2)
        Matched keypoint locations in image 2
    """
    # TODO Your code goes here
    #Nearest neighbor distance ratio:
    NNDR = 0.8
    nbrs = NearestNeighbors(n_neighbors=2).fit(des2)
    distances, indices = nbrs.kneighbors(des1, 2, return_distance=True)
    #print(len(distances))
    #print(indices, distances)
    pairlist = []
    x1, x2= [], []
    for i, distance in enumerate(distances):
        if distance[0]<NNDR*distance[1]:
            x1.append(loc1[i])
            x2.append(loc2[indices[i][0]])
    return np.array(x1), np.array(x2)
    

def EstimateH(x1, x2, ransac_n_iter, ransac_thr):
    """
    Estimate the homography between images using RANSAC
    
    Parameters
    ----------
    x1 : ndarray of shape (n, 2)
        Matched keypoint locations in image 1
    x2 : ndarray of shape (n, 2)
        Matched keypoint locations in image 2
    ransac_n_iter : int
        Number of RANSAC iterations
    ransac_thr : float
        Error threshold for RANSAC

    Returns
    -------
    H : ndarray of shape (3, 3)
        The estimated homography
    inlier : ndarray of shape (k,)
        The inlier indices
    """

    # TODO Your code goes here
    ransac_n_iter=500
    kk = 0
    for r in range(ransac_n_iter):
        x1_s = x1[np.random.randint(x1.shape[0], size=4)]
        x2_s = x2[np.random.randint(x2.shape[0], size=4)]
        A = []
        b = []
        for i in range(len(x1_s)):
            xA, yA = x1_s[i][0], x1_s[i][1]
            xB, yB = x2_s[i][0], x2_s[i][1]
            A.append([xA, yA, 1, 0, 0, 0, -xA*xB, -yA*xB])
            A.append([0, 0, 0, xA, yA, 1, -xA*yB, -yA*yB])
            b.append([xB])
            b.append([yB])

        A = np.array(A)
        b = np.array(b)
        x = np.linalg.inv(np.transpose(A)@A)@np.transpose(A)@b
        H = np.concatenate((x,np.array([[1]])), axis=0).reshape(3,3)
    
        inlier_idx=[]
        for i, (point1, point2) in enumerate(zip(x1,x2)):
        #print(H.shape, np.array([[point1[0],point1[1],1]]).shape)
        #print(np.matmul(H,np.concatenate((point1,np.array([[1]])), axis=0)))
        #print(H@np.array([[point1[0]],[point1[1]],[1]]), np.array([[point2[0]],[point2[1]],[1]]))
        #print(np.linalg.norm(H@np.array([[point1[0]],[point1[1]],[1]]) - np.array([[point2[0]],[point2[1]],[1]])))
            if np.linalg.norm(H@np.array([[point1[0]],[point1[1]],[1]]) - np.array([[point2[0]],[point2[1]],[1]])) < ransac_thr:
                inlier_idx.append(i)
    
        if len(inlier_idx) > kk:
            kk = len(inlier_idx)
            print("here", kk)
    
    #print(A, A.shape, b, b.shape, x, x.shape, x.reshape(3,3))
    return 0, 0
    


def EstimateR(H, K):
    """
    Compute the relative rotation matrix
    
    Parameters
    ----------
    H : ndarray of shape (3, 3)
        The estimated homography
    K : ndarray of shape (3, 3)
        The camera intrinsic parameters

    Returns
    -------
    R : ndarray of shape (3, 3)
        The relative rotation matrix from image 1 to image 2
    """
    
    # TODO Your code goes here
    pass


def ConstructCylindricalCoord(Wc, Hc, K):
    """
    Generate 3D points on the cylindrical surface
    
    Parameters
    ----------
    Wc : int
        The width of the canvas
    Hc : int
        The height of the canvas
    K : ndarray of shape (3, 3)
        The camera intrinsic parameters of the source images

    Returns
    -------
    p : ndarray of shape (Hc, Hc, 3)
        The 3D points corresponding to all pixels in the canvas
    """

    # TODO Your code goes here
    pass


def Projection(p, K, R, W, H):
    """
    Project the 3D points to the camera plane
    
    Parameters
    ----------
    p : ndarray of shape (Hc, Wc, 3)
        A set of 3D points that correspond to every pixel in the canvas image
    K : ndarray of shape (3, 3)
        The camera intrinsic parameters
    R : ndarray of shape (3, 3)
        The rotation matrix
    W : int
        The width of the source image
    H : int
        The height of the source image

    Returns
    -------
    u : ndarray of shape (Hc, Wc, 2)
        The 2D projection of the 3D points
    mask : ndarray of shape (Hc, Wc)
        The corresponding binary mask indicating valid pixels
    """
    
    # TODO Your code goes here
    pass


def WarpImage2Canvas(image_i, u, mask_i):
    """
    Warp the image to the cylindrical canvas
    
    Parameters
    ----------
    image_i : ndarray of shape (H, W, 3)
        The i-th image with width W and height H
    u : ndarray of shape (Hc, Wc, 2)
        The mapped 2D pixel locations in the source image for pixel transport
    mask_i : ndarray of shape (Hc, Wc)
        The valid pixel indicator

    Returns
    -------
    canvas_i : ndarray of shape (Hc, Wc, 3)
        the canvas image generated by the i-th source image
    """
    
    # TODO Your code goes here
    pass


def UpdateCanvas(canvas, canvas_i, mask_i):
    """
    Update the canvas with the new warped image
    
    Parameters
    ----------
    canvas : ndarray of shape (Hc, Wc, 3)
        The previously generated canvas
    canvas_i : ndarray of shape (Hc, Wc, 3)
        The i-th canvas
    mask_i : ndarray of shape (Hc, Wc)
        The mask of the valid pixels on the i-th canvas

    Returns
    -------
    canvas : ndarray of shape (Hc, Wc, 3)
        The updated canvas image
    """
    
    # TODO Your code goes here
    pass

def FeatureMatchingDraw(img1, img2, x1, x2):
    height, width, _ = img1.shape
    img_concate_Hori=np.concatenate((img1,img2),axis=1)
    for point1, point2 in zip(x1, x2):
        color1 = (list(np.random.choice(range(256), size=3)))  
        color =[int(color1[0]), int(color1[1]), int(color1[2])] 
        x11, y11 = point1.ravel()
        x22, y22 = point2.ravel()
        cv2.circle(img_concate_Hori, (round(x11), round(y11)), 3, color, -1)
        cv2.circle(img_concate_Hori, (round(x22)+width, round(y22)), 3, color, -1)
        cv2.line(img_concate_Hori, (round(x11), round(y11)), (round(x22)+width, round(y22)), color, 1)

    cv2.imshow('concatenated_Hori',img_concate_Hori)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    ransac_n_iter = 500
    ransac_thr = 3
    K = np.asarray([
        [320, 0, 480],
        [0, 320, 270],
        [0, 0, 1]
    ])


    # Read all images
    im_list = []
    for i in range(1, 9):
        im_file = '{}.jpg'.format(i)
        im = cv2.imread(im_file)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im_list.append(im)

    rot_list = []
    rot_list.append(np.eye(3))
    for i in range(len(im_list) - 1):
        # Load consecutive images I_i and I_{i+1}
		# TODO Your code goes here
        img1, img2 = im_list[i], im_list[i+1]
		
        # Extract SIFT features
		# TODO Your code goes here
        sift = cv2.xfeatures2d.SIFT_create()
        gray1= cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        kp1, des1 = sift.detectAndCompute(gray1,None)
        gray2= cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        kp2, des2 = sift.detectAndCompute(gray2,None)
        loc1 = np.array([kp1[idx].pt for idx in range(0, len(kp1))]).reshape(-1,2)
        loc2 = np.array([kp2[idx].pt for idx in range(0, len(kp2))]).reshape(-1,2)
        
        # Find the matches between two images (x1 <--> x2)
        x1, x2 = MatchSIFT(loc1, des1, loc2, des2)
        #FeatureMatchingDraw(img1, img2, x1, x2)
        
        # Estimate the homography between images using RANSAC
        H, inlier = EstimateH(x1, x2, ransac_n_iter, ransac_thr)
        break
'''
        # Compute the relative rotation matrix R
        R = EstimateR(H, K)
		
		# Compute R_new (or R_i+1)
		# TODO Your code goes here
		
        rot_list.append(R_new)

    Him = im_list[0].shape[0]
    Wim = im_list[0].shape[1]
    
    Hc = Him
    Wc = len(im_list) * Wim // 2
	
    canvas = np.zeros((Hc, Wc, 3), dtype=np.uint8)
    p = ConstructCylindricalCoord(Wc, Hc, K)

    fig = plt.figure('HW1')
    plt.axis('off')
    plt.ion()
    plt.show()
    for i, (im_i, rot_i) in enumerate(zip(im_list, rot_list)):
        # Project the 3D points to the i-th camera plane
        u, mask_i = Projection(p, K, rot_i, Wim, Him)
        # Warp the image to the cylindrical canvas
        canvas_i = WarpImage2Canvas(im_i, u, mask_i)
        # Update the canvas with the new warped image
        canvas = UpdateCanvas(canvas, canvas_i, mask_i)
        plt.imshow(canvas)
        plt.savefig('output_{}.png'.format(i+1), dpi=600, bbox_inches = 'tight', pad_inches = 0)
'''

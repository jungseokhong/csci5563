import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors


#Nearest neighbor distance ratio:
NNDR = 0.8


des1 = np.array([[0,1],[0,2],[0,3]])
des2 = np.array([[0,1],[0,-1],[0,13],[0,2]])
#nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
nbrs = NearestNeighbors(n_neighbors=2).fit(des1)
distances, indicies = nbrs.kneighbors(des2, 2, return_distance=True)
for i, distance in enumerate(distances):
        if distance[0]>0.8*distance[1]:
            remove
            
            
#distances, indices = nbrs.kneighbors(X)
#print(distances)
#print(indices)

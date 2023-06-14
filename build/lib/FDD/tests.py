import numpy as np
from FDD import FDD
from FDD.SURE import *
from matplotlib import pyplot as plt
import cv2
import os
from types import MethodType

# #------ test 1

def boundaryGridToData(self, J_grid, u, average = False):
    # get the indices of the J_grid where J_grid is 1
    

    k = np.array(np.where(J_grid == 1))

    # Store the average points
    Y_jumpfrom = []
    Y_jumpto = []
    Y_jumpsize = []
    Y_boundary = []
    X_jumpfrom = []
    X_jumpto = []
    

    # Iterate over the boundary points
    for i in range(k.shape[1]):
        
        origin_points = []
        dest_points = []

        # Get the coordinates of the current boundary point
        point = k[:, i]

        # Initialize a list to store the neighboring hypervoxels
        neighbors = []

        # Iterate through all dimensions
        for d in range(k.shape[0]):

            # Calculate the down-right neighbor along the current dimension
            neighbor = point.copy()
            if neighbor[d] < J_grid.shape[d] - 1:
                neighbor[d] += 1

                # Check if the neighbor is not a boundary point
                if J_grid[tuple(neighbor)] != 1:
                    neighbors.append(neighbor)

        # Check if there are any valid neighbors
        if neighbors:
            
            # origin_points
            origin_points = self.grid_x_og[tuple(point)]
            if len(origin_points) == 0:
                origin_points = self.grid_x[tuple(point)] + self.resolution / 2
            Yjumpfrom = float(u[tuple(point)])

                
            # jumpfrom point
            origin_points = np.stack(origin_points).squeeze()
            if origin_points.ndim > 1: # if there are multiple points in the hypervoxel, take the mean
                jumpfrom = np.mean(origin_points, axis = 0)
            else:
                jumpfrom = origin_points
                

            # jumpto point
            pointslist = [self.grid_x_og[tuple(neighbors[j])] if self.grid_x_og[tuple(neighbors[j])] != []  # if grid cell is empty, assign centerpoint
                            else [self.grid_x[tuple(neighbors[j])] + self.resolution / 2] for j in range(len(neighbors))]
            
            if average:
                counts = [len(pointslist[j]) for j in range(len(neighbors))]
                total = sum(counts) # TODO: jump sizes on diagonal boundary sections are off
                Yjumpto = np.sum([(u[tuple(neighbors[j])] * counts[j]) / total for j in range(len(neighbors))]) # proper unweighted average of the y values
                dest_points = np.stack([item for sublist in pointslist for item in sublist]).squeeze()
                if dest_points.ndim > 1: # if there are multiple points in the hypervoxel, take the mean
                    jumpto = np.mean(dest_points, axis = 0)
                else:
                    jumpto = dest_points
            else:
                # get closest point
                dists = [[np.linalg.norm(jumpfrom - point) for point in pointslist[j]] for j in range(len(neighbors))]
                idx = np.argmin([np.argmin(sublist) for sublist in dists])
                closest = tuple(neighbors[idx])
                Yjumpto = u[closest]
                
                dest_points = self.grid_x_og[closest]
                if len(dest_points) == 0:
                    dest_points = self.grid_x[closest] + self.resolution / 2
                dest_points = np.stack(dest_points).squeeze()

                if dest_points.ndim > 1: # if there are multiple points in the hypervoxel, take the mean
                    jumpto = np.mean(dest_points, axis = 0)
                else:
                    jumpto = dest_points
            
            # append to lists
            Y_boundary.append((jumpfrom + jumpto) / 2)
            Y_jumpfrom.append(Yjumpfrom)
            Y_jumpto.append(Yjumpto)
            Y_jumpsize.append(Yjumpto - Yjumpfrom)
            X_jumpfrom.append(jumpfrom)
            X_jumpto.append(jumpto)
    
    if Y_boundary:
        Y_boundary = np.stack(Y_boundary)
        Y_jumpfrom = np.stack(Y_jumpfrom)
        Y_jumpto = np.stack(Y_jumpto)
        Y_jumpsize = np.stack(Y_jumpsize)
        
        # create named array to return
        rays = [Y_boundary[:,d] for d in range(Y_boundary.shape[1])] + [Y_jumpfrom, Y_jumpto, Y_jumpsize]
        names = ["X_" + str(d) for d in range(Y_boundary.shape[1])] + ["Y_jumpfrom", "Y_jumpto", "Y_jumpsize"]
        jumps = np.core.records.fromarrays(rays, names=names)
    else:
        jumps = None
    
    return jumps

#os.chdir("../..")
image = "resources/images/marylin.png"
mIn = cv2.imread(image, (0))

scale_percent = 20 # percent of original size
width = int(mIn.shape[1] * scale_percent / 100)
height = int(mIn.shape[0] * scale_percent / 100)
dim = (width, height)
mIn = cv2.resize(mIn, dim)
mIn = mIn.astype(np.float32)


mIn /= 255


Y = mIn.copy() #.flatten()
#Y = np.stack([Y, Y], axis = 1)
# get labels of grid points associated with Y values in mIn
#X = np.stack(np.meshgrid(*[np.arange(Y.shape[1]), np.arange(Y.shape[0])]), axis = -1)


# # reshuffle Y and X in the same way so that it resembles normal data
Y = Y.flatten()
X = np.stack([np.tile(np.arange(0, mIn.shape[0], 1), mIn.shape[1]), 
              np.repeat(np.arange(0, mIn.shape[0], 1), mIn.shape[1])], axis = 1)

        
def forward_differences(ubar, D : int):

    diffs = []

    for dim in range(int(D)):
        #zeros_shape = torch.jit.annotate(List[int], [])
        zeros_shape = list(ubar.shape)
        zeros_shape[dim] = 1
        # for j in range(ubar.dim()): 
        #     if j == dim:
        #         zeros_shape.append(1)
        #     else:
        #         zeros_shape.append(ubar.shape[j])
        zeros = np.zeros(zeros_shape)
        diff = np.concatenate((np.diff(ubar, axis=dim), zeros), axis=dim)
        diffs.append(diff)
                # Stack the results along a new dimension (first dimension)
    u_star = np.stack(diffs, axis=0)

    return u_star

def boundary(u, nu):
    u_diff = forward_differences(u, D = len(u.shape))
    u_norm = np.linalg.norm(u_diff, axis = 0, ord = 2) # 2-norm
    return (u_norm >= np.sqrt(nu)).astype(int)


# histogram of gradient norm
test = np.linalg.norm(forward_differences(mIn, D = len(mIn.shape)), axis = 0)
out = plt.hist(test)

X1 = np.tile(out[1][1:], out[0].shape[0])
X2 = out[0].reshape(-1,1).squeeze(1)
Z = np.stack([X1, X2], axis = 1)

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, random_state=0).fit(Z)
nu = X1[kmeans.labels_ == 1].max()

resolution = 1/int(np.sqrt(X.size*2/3))
model = FDD(Y, X, level = 16, lmbda = 1, nu = 0.01, iter = 10000, tol = 5e-5, 
            image=False, pick_nu = "MS", resolution=resolution, scaled=True, scripted=False)

model.boundaryGridToData = MethodType(boundaryGridToData, model)

u, jumps, J_grid, nrj, eps, it = model.run()


# cv2.imwrite("result.png",u*255)

# model.pick_nu = "kmeans"
# J_grid, jumps = model.boundary(u)

# plt.imshow(u)
# plt.show()

# plt.imshow(J_grid)
# plt.show()

# # plot image with boundary
# plt.imshow(u, cmap = "gray")
# test = J_grid.copy().astype(np.float32)
# test[test == 0] = np.nan
# plt.imshow(test, cmap='autumn', interpolation='none')
# plt.show()
# plt.savefig("resources/images/marylin_segmented.png")

# # histogram of gradient norm

# test = np.linalg.norm(forward_differences(u, D = len(u.shape)), axis = 0)
# out = plt.hist(test)
# #plt.show()
# plt.savefig("resources/images/hist.png")

# X = np.tile(out[1][1:], out[0].shape[0])
# Y = out[0].reshape(-1,1).squeeze(1)
# Z = np.stack([X, Y], axis = 1)

# from sklearn.cluster import KMeans

# kmeans = KMeans(n_clusters=2, random_state=0).fit(Z)
# nu = X[kmeans.labels_ == 1].max()

# # get new boundary
# J_new = boundary(u, nu = nu**2) # the squared is just cause we're taking the square root

# plt.imshow(J_new)

# # plt.hist(test.reshape(-1,1)[kmeans.labels_ == 2], bins = 100)

    
# # from pomegranate import  *

# # model = GeneralMixtureModel.from_samples(NormalDistribution, n_components=2, X=test.reshape(-1,1))
# # labels = model.predict(test.reshape(-1,1))

# # plt.hist(test.reshape(-1,1)[labels == 0], bins = 100)

# # plt.hist(test.reshape(-1,1),  bins = 500)
# # plt.ylim(0,5)



#------- test2

# # Generate some random data points from a discontinuous function
# np.random.seed(0)
# data = np.random.rand(100, 2) # draw 1000 2D points from a uniform

# # Create the grid
# # Define the grid dimensions and resolution
# xmin, xmax = 0, 1
# ymin, ymax = 0, 1
# resolution = 0.01 # 100 by 100 grid
# x, y = np.meshgrid(np.arange(xmin, xmax, resolution), np.arange(ymin, ymax, resolution))
# grid = np.dstack((x, y))
# grid_f = np.zeros(grid.shape[:2])

# def f(x,y):
#     temp = np.sqrt((x-1/2)**2 + (y-1/2)**2)
#     if temp < 1/4:
#         return temp
#     else:
#         return temp + 1/8

# # Compute the function values on the grid
# for i in range(grid.shape[0]):
#     for j in range(grid.shape[1]):
#         grid_f[i, j] = f(grid[i, j][0], grid[i, j][1])
        
# # now sample the function values on the data points
# grid_sample = np.zeros((data.shape[0],1))
# for i in range(data.shape[0]):
#         grid_sample[i] = f(data[i,0], data[i,1]) + np.random.normal(0, 0.08)

# X = data.copy()
# Y = grid_sample.copy().flatten()
# # and run the FDD command
# model = FDD(Y, X, level = 16, lmbda = 1, nu = 0.02, iter = 5000, tol = 5e-5, qtile = 0.08,
#             pick_nu = "MS", scaled = True, scripted = False)

# import time
# t0 = time.time()
# #res = SURE(model, tuner = True, num_gpus=0)
# u, jumps, J_grid, nrj, eps, it = model.run()
# print(time.time() - t0)

# plt.imshow(u)
# plt.show()


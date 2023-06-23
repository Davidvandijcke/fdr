import numpy as np
from FDD import FDD
from FDD.SURE import *
from matplotlib import pyplot as plt
import cv2
import os
from types import MethodType
from itertools import product, combinations
import pandas as pd

# #------ test 1




# #os.chdir("../..")
# image = "resources/images/marylin.png"
# mIn = cv2.imread(image, (0))

# scale_percent = 20 # percent of original size
# width = int(mIn.shape[1] * scale_percent / 100)
# height = int(mIn.shape[0] * scale_percent / 100)
# dim = (width, height)
# mIn = cv2.resize(mIn, dim)
# mIn = mIn.astype(np.float32)


# mIn /= 255


# Y = mIn.copy() #.flatten()
# #Y = np.stack([Y, Y], axis = 1)
# # get labels of grid points associated with Y values in mIn
# #X = np.stack(np.meshgrid(*[np.arange(Y.shape[1]), np.arange(Y.shape[0])]), axis = -1)


# # # reshuffle Y and X in the same way so that it resembles normal data
# Y = Y.flatten()
# X = np.stack([np.tile(np.arange(0, mIn.shape[0], 1), mIn.shape[1]), 
#               np.repeat(np.arange(0, mIn.shape[0], 1), mIn.shape[1])], axis = 1)

        
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

# def boundary(u, nu):
#     u_diff = forward_differences(u, D = len(u.shape))
#     u_norm = np.linalg.norm(u_diff, axis = 0, ord = 2) # 2-norm
#     return (u_norm >= np.sqrt(nu)).astype(int)


# # histogram of gradient norm
# test = np.linalg.norm(forward_differences(mIn, D = len(mIn.shape)), axis = 0)
# out = plt.hist(test)

# X1 = np.tile(out[1][1:], out[0].shape[0])
# X2 = out[0].reshape(-1,1).squeeze(1)
# Z = np.stack([X1, X2], axis = 1)

# from sklearn.cluster import KMeans

# kmeans = KMeans(n_clusters=2, random_state=0).fit(Z)
# nu = X1[kmeans.labels_ == 1].max()

# resolution = 1/int(np.sqrt(X.size*2/3))
# model = FDD(Y, X, level = 16, lmbda = 1, nu = 0.01, iter = 10000, tol = 5e-5, 
#             image=False, pick_nu = "MS", resolution=resolution, scaled=True, scripted=False)

# model.boundaryGridToData = MethodType(boundaryGridToData, model)
# model.explore = MethodType(explore, model)

# u, jumps, J_grid, nrj, eps, it = model.run()


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

# Generate some random data points from a discontinuous function
np.random.seed(0)
data = np.random.rand(10000, 2) # draw 1000 2D points from a uniform

# Create the grid
# Define the grid dimensions and resolution
xmin, xmax = 0, 1
ymin, ymax = 0, 1
resolution = 0.01 # 100 by 100 grid
x, y = np.meshgrid(np.arange(xmin, xmax, resolution), np.arange(ymin, ymax, resolution))
grid = np.dstack((x, y))
grid_f = np.zeros(grid.shape[:2])

def f(x,y):
    temp = np.sqrt((x-1/2)**2 + (y-1/2)**2)
    if temp < 1/4:
        return temp
    else:
        return temp + 0.1066

# Compute the function values on the grid
for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        grid_f[i, j] = f(grid[i, j][0], grid[i, j][1])
        
# now sample the function values on the data points
grid_sample = np.zeros((data.shape[0],1))
for i in range(data.shape[0]):
        grid_sample[i] = f(data[i,0], data[i,1]) + np.random.normal(0, 0.01)

X = data.copy()
Y = grid_sample.copy().flatten()
# and run the FDD command
resolution = 1/int(np.sqrt(2/3*X.size))
model = FDD(Y, X, level = 16, lmbda = 120, nu = 0.0025, iter = 5000, tol = 5e-5, qtile = 0.08,
            pick_nu = "MS", scaled = True, scripted = False, resolution=resolution)

import time

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
            #neighbors = list(self.explore(point, J_grid))
            neighbors = []  
            for d in range(J_grid.ndim):
                neighbor = point.copy()
                if neighbor[d] < J_grid.shape[d] - 1:
                    neighbor[d] += 1
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
                    # # get point with largest jump size
                    # Yjumptos = [u[tuple(neighbors[j])] for j in range(len(neighbors))]
                    # Yjumpsizes = [abs(Yjumptos[j] - Yjumpfrom) for j in range(len(neighbors))]
                    # idx = np.argmax(Yjumpsizes)
                    # Yjumpto = Yjumptos[idx]
                    # dest_points = self.grid_x_og[tuple(neighbors[idx])]
                    # if len(dest_points) == 0:
                    #     dest_points = self.grid_x[tuple(neighbors[idx])] + self.resolution / 2
                    # dest_points = np.stack(dest_points).squeeze()
                    
                    # if dest_points.ndim > 1: # if there are multiple points in the hypervoxel, take the mean
                    #     jumpto = np.mean(dest_points, axis = 0)
                    # else:
                    #     jumpto = dest_points
                    
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
            jumps = pd.DataFrame(np.core.records.fromarrays(rays, names=names))
        else:
            jumps = None

        return jumps
    
model.boundaryGridToData = MethodType(boundaryGridToData, model)
# model.getNearestPoint = MethodType(getNearestPoint, model)
# model.explore = MethodType(explore, model)
#res = SURE(model, tuner = True, num_gpus=0)

u, jumps, J_grid, nrj, eps, it = model.run()

def is_adjacent(points):
    # A function to check if all points are adjacent to each other
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            if np.count_nonzero(np.abs(np.array(points[i]) - np.array(points[j]))) > 1:
                return False
    return True

def get_thick_boundary_points(J_grid):
    # Get the array dimensions
    dimensions = J_grid.shape

    # Initialize the thick boundary grid
    thick_boundary_grid = np.zeros(dimensions, dtype=int)

    # Generate all shifts for queen's rule adjacency in d dimensions
    shifts = list(product([-1, 0, 1], repeat=J_grid.ndim))
    shifts.remove((0,) * J_grid.ndim)  # Remove the zero-shift

    # Iterate over all points in the grid
    for index in np.ndindex(dimensions):
        
        # Check if current point is a jump point
        if J_grid[index] == 1:
            # check if left or up neighbor is boundary point
            lshifts = [(0,-1), (-1,0)]
            lneighbors=[]
            for shift in lshifts:
                neighbor = [index[d] + shift[d] for d in range(J_grid.ndim)]
                if all(0 <= neighbor[d] < dimensions[d] for d in range(J_grid.ndim)):
                    lneighbors.append(tuple(neighbor))
            if any(J_grid[tuple(lu)]==1 for lu in lneighbors):
                neighbors = []
                for shift in shifts:
                    neighbor = [index[d] + shift[d] for d in range(J_grid.ndim)]
                    if all(0 <= neighbor[d] < dimensions[d] for d in range(J_grid.ndim)):
                        neighbors.append(tuple(neighbor))

                    # Check if there are any d+1 subsets of neighbors which are all jump points and all adjacent to each other
                    for subset in combinations(neighbors, len(index) + 1):
                        if all(J_grid[neighbor] == 1 for neighbor in subset) and is_adjacent(subset):
                            # Current point is a thick boundary point
                            thick_boundary_grid[index] = 1
    return thick_boundary_grid

thick = get_thick_boundary_points(J_grid)

jumps = model.boundaryGridToData(J_grid, u, average=False)


temp = pd.DataFrame(jumps)
temp['Y_jumpsize'].abs().hist(bins=40)
temp['Y_jumpsize'].abs().mean()

np.mean(np.abs(0.125 -  k.abs()))

jsizes = forward_differences(u, D=len(u.shape))
jsizes = np.linalg.norm(jsizes, axis = 0, ord = 2)
k = jsizes[J_grid]

# from itertools import product

test = temp[temp['Y_jumpsize'].abs() > np.sqrt(0.0016)]

# def get_packed_points(J_grid, thick_boundary_points):
#     dimensions = J_grid.shape
#     packed_points = np.zeros(dimensions, dtype=int)
#     shifts = [(0,)*i + (-1,) + (0,)*(len(dimensions)-i-1) for i in range(len(dimensions))] 
#     shifts.extend((0,)*i + (1,) + (0,)*(len(dimensions)-i-1) for i in range(len(dimensions)))  # rook neighbors

#     for index in np.ndindex(*dimensions):
#         if thick_boundary_points[index] == 1:
#             neighbors = [tuple(index[i] + shift[i] for i in range(len(index))) for shift in shifts]
#             valid_neighbors = [neighbor for neighbor in neighbors if all(0 <= neighbor[i] < dimensions[i] for i in range(len(neighbor)))]
#             if all(thick_boundary_points[neighbor] == 1 for neighbor in valid_neighbors if J_grid[neighbor] == 1):
#                 packed_points[index] = 1
#     return packed_points

# def get_eliminate_points(J_grid, packed_points):
#     dimensions = J_grid.shape
#     eliminate_points = np.zeros(dimensions, dtype=int)
#     shifts = list(product([0, 1], repeat=J_grid.ndim))  # only "down-right" neighbors

#     for index in np.ndindex(*dimensions):
#         if packed_points[index] == 1:
#             neighbors = [tuple(index[i] + shift[i] for i in range(len(index))) for shift in shifts]
#             valid_neighbors = [neighbor for neighbor in neighbors if all(0 <= neighbor[i] < dimensions[i] for i in range(len(neighbor)))]
#             if any(J_grid[neighbor] == 0 for neighbor in valid_neighbors):
#                 eliminate_points[index] = 1
#     return eliminate_points

# def process_thick_boundary_points(J_grid):
#     while True:
#         thick_boundary_points = get_thick_boundary_points(J_grid)
#         if np.sum(thick_boundary_points) == 0:
#             break
#         packed_points = get_packed_points(J_grid, thick_boundary_points)
#         eliminate_points = get_eliminate_points(J_grid, packed_points)
#         J_grid[eliminate_points == 1] = 0
#     return J_grid

# pgrid = J_grid.copy()
# thick = process_thick_boundary_points(pgrid)

# test = get_thick_boundary_points(pgrid)

# plt.imshow(J_grid-thick)

# fig, ax = plt.subplots(1,1)
# # plot thick on top of J_grid
# ax.imshow(J_grid, cmap = "gray")
# test = thick.copy().astype(np.float32)
# test[test == 0] = np.nan
# ax.imshow(test, cmap='autumn', interpolation='none')

test = temp[temp['Y_jumpsize'].abs() < 0.04]
plt.scatter(temp['X_0'], temp['X_1'], c="gray")
plt.scatter(test['X_0'], test['X_1'], c="red")

# if there are rows with identical X_0, X_1 in temp, keep the one with the largest Y_jumpsize 
temp = temp.sort_values(by=['X_0', 'X_1', 'Y_jumpsize'], ascending=False)
temp = temp.drop_duplicates(subset=['X_0', 'X_1'], keep='first')

# test = J_grid - (1-thick)
# plt.imshow((thick)[40:50,40:60])

# # print(time.time() - t0)

# # plt.imshow(u)
# # plt.show()


fig, ax = plt.subplots(1,1)
# plot thick on top of J_grid
ax.imshow(u)
ax.imshow(J_grid, alpha=0.2)




fig, ax = plt.subplots(1,1)
# plot thick on top of J_grid
ax.imshow(u)
idx = (jsizes > 0.04) & (jsizes < 0.06)
ax.imshow(idx, alpha = 0.3)
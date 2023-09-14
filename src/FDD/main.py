
import numpy as np
import torch
from .utils import * 
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree
from .primaldual_multi_scaled_tune import PrimalDual
from matplotlib import pyplot as plt
import pandas as pd
import random

class FDD():
    def __init__(self, Y : np.array, X : np.array, pick_nu : str="kmeans", level : int=16, 
                 lmbda : float=1, nu : float=0.01, iter : int=1000, tol : float=5e-5, rectangle : bool=False, 
                 qtile : float=0.05, image : bool=False, grid : bool=False, resolution : float=None,
                 scaled=False, scripted=True, average=False, CI=True, alpha=0.05) -> None:

        self.device = setDevice()
        torch.set_grad_enabled(False)
        
        self.Y_raw = Y.copy() # retain original data
        self.X_raw = X.copy()
        self.Y = Y.copy() # arrays I'll pass to PyTorch
        self.X = X.copy()
        self.N = self.Y.shape[0]

        
        self.image = image
        self.grid = grid
        self.level = level
        self.lmbda = lmbda
        self.iter = iter
        self.tol = tol
        self.rectangle = rectangle
        self.qtile = qtile
        self.resolution = resolution
        
        # confidence intervals
        self.CI = CI
        self.alpha = alpha
        self.R_u = None # u residuals conformal split
        self.R_J = None
        self.u_cs = None # conformal split u estimate
        self.u_diff = None # same for u_diff
        
        # for acceleration
        self.gamma = 1
        self.theta_u = 1 # placeholder
        self.theta_mu = 1
        
        self.scripted = scripted
        self.scaled = scaled
        
        self.average = average

        
        self.X = self.normalizeData(self.rectangle, self.X) # scale data to unit hypercube
        # if X only 1 dimension, add a second dimension
        if self.X.ndim == 1:
            self.X = np.expand_dims(self.X, -1)
        self.grid_x_og, self.grid_x, self.grid_y = self.castDataToGrid(self.X, self.Y)
        self.grid_y = (self.grid_y - np.min(self.Y_raw, axis=0)) / np.max(self.Y_raw, axis=0)
        

        self.nu = nu
        self.pick_nu = pick_nu
        
        if self.scripted:
            # scale gradients?
            if self.scaled:
                script = "scripted_primal_dual_scaled"
            else:
                script = "scripted_primal_dual"
            
            self.model = load_model(script + ".pt", device=self.device) #torch.jit.load(script + ".pt", map_location = self.device)
        else:
            self.model = PrimalDual()
        
        self.model = self.model.to(self.device)
        # TODO: exclude duplicate points (there shouldnt be any cause the variables are assumed to be continuous but anyway)
        
    @staticmethod
    def normalizeData(rectangle, X):
        
        #min_y = np.min(self.Y, axis = 0)
        min_x = np.min(X, axis = 0)
        
        # self.Y = self.Y - min_y # start at 0
        X = X - min_x
        
        # max_y = np.max(self.Y, axis = 0)
        if rectangle: # retain proportions between data -- should be used when units are identical along all axes
            max_x = np.max(X)
            # self.Y = self.Y / max_y
            X = X / max_x
        else: # else scale to square
            max_x = np.max(X, axis = 0)
            # self.Y = self.Y / max_y
            X = X / max_x
            
        return X
            

            
    
    def castDataToGridSmooth(self, X, Y):
        
        if self.resolution is None:
            self.resolution = 1/int(self.X_raw.max(axis=0).min()) # int so we get a natural number of grid cells
        
        xmax = np.max(X, axis=0)
        
        # set up grid
        grid_x = np.meshgrid(*[np.arange(0, xmax[i], self.resolution) for i in reversed(range(X.shape[1]))])
        grid_x = np.stack(grid_x, axis=-1)

        if Y.ndim > 1: # account for vector-valued outcomes
            grid_y = np.zeros(list(grid_x.shape[:-1]) + [Y.shape[1]])
        else:
            grid_y = np.zeros(list(grid_x.shape[:-1]))
        grid_x_og = np.empty(list(grid_x.shape[:-1]), dtype=object) # assign original x values as well for later
        
        # Get the indices of the grid cells for each data point
        indices = [(np.clip(X[:, i] // self.resolution, 0, grid_y.shape[i] - 1)).astype(int) for i in range(X.shape[1])]
        indices = np.array(indices).T

        # Create a count array to store the number of data points in each cell
        counts = np.zeros_like(grid_y)

        # Initialize grid_x_og with empty lists
        for index in np.ndindex(grid_x_og.shape):
            grid_x_og[index] = []
        
        # Iterate through the data points and accumulate their values in grid_y and grid_x_og
        for i, index_tuple in enumerate(indices):
            index = tuple(index_tuple)
            if np.all(index < grid_y.shape):
                grid_y[index] += Y[i]
                counts[index] += 1
                grid_x_og[index].append(X[i]) # TODO: needs to be X_raw
        
        # Divide the grid_y by the counts to get the average values
        grid_y = np.divide(grid_y, counts, where=counts != 0)

        # Find the closest data point for empty grid cells
        empty_cells = np.where(counts == 0)
        empty_cell_coordinates = np.vstack([empty_cells[i] for i in range(X.shape[1])]).T * self.resolution
        if empty_cell_coordinates.size > 0:
            tree = cKDTree(X + self.resolution / 2) # get centerpoints of hypervoxels
            _, closest_indices = tree.query(empty_cell_coordinates, k=1)
            closest_Y_values = Y[closest_indices]

            # Assign the closest data point values to the empty grid cells
            grid_y[empty_cells] = closest_Y_values
        
        # add an extra "channel" dimension if we have a scalar outcome
        if Y.ndim == 1:
            grid_y = grid_y.reshape(grid_y.shape + (1,))

        return (grid_x_og, grid_x, grid_y)

        
        
    def castDataToGrid(self, X, Y):
        
        return self.castDataToGridSmooth(X, Y)
        

        
    @staticmethod
    def interpolate(k, uk0, uk1, l):
        return (k + (0.5 - uk0) / (uk1 - uk0)) / l

    def isosurface(self, u):

        mask = (u[...,:-1] > 0.5) & (u[...,1:] <= 0.5)
        # Find the indices of the first True value along the last dimension, and set all the following ones to False
        mask[..., 1:] = (mask[..., 1:]) & (mask.cumsum(-1)[...,:-1] < 1)

        uk0 = u[...,:-1][mask]
        uk1 = u[...,1:][mask]
        
        # get the indices of the last dimension where mask is True
        k = np.where(mask == True)[-1] + 1
        
        h_img = self.interpolate(k, uk0, uk1, self.level).reshape(self.grid_y.shape[:-1])
        
        return h_img
    
    def k_means_boundary(self):
        # histogram of gradient norm
        test = np.linalg.norm(self.forward_differences(self.mIn, D = len(self.mIn.shape)), axis = 0)
        out = plt.hist(test)

        X1 = np.tile(out[1][1:], out[0].shape[0])
        X2 = out[0].reshape(-1,1).squeeze(1)
        Z = np.stack([X1, X2], axis = 1)


        kmeans = KMeans(n_clusters=2, random_state=0).fit(Z)
        nu = X1[kmeans.labels_ == 1].max()
        
    def explore(self, point, J_grid, visited_points=None):
        
        if visited_points is None:
            visited_points = set()
        
        neighbors = []

        for d in range(J_grid.ndim):
            neighbor = point.copy()
            if neighbor[d] < J_grid.shape[d] - 1:
                neighbor[d] += 1
                if J_grid[tuple(neighbor)] == 0:
                    visited_points.add(tuple(neighbor))
                neighbors.append(neighbor)
        
        # Check if all neighbors are jump points, if so, continue exploring
        if all(J_grid[tuple(neighbor)] == 1 for neighbor in neighbors):
            for neighbor in neighbors:
                if tuple(neighbor) not in visited_points:
                    visited_points.update(self.explore(neighbor, J_grid, visited_points))

        return visited_points

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
            average = False

            # Initialize a list to store the neighboring hypervoxels
            #neighbors = list(self.explore(point, J_grid))
            neighbors = []  
            count = 0
            for d in range(J_grid.ndim):
                neighbor = point.copy()
                if neighbor[d] < J_grid.shape[d] - 1:
                    neighbor[d] += 1
                    neighbors.append(neighbor)
                    count += 1
            if count == 0:
                neighbors.append(point.copy())


            # Check if there are any valid neighbors
            if neighbors:

                # origin_points
                origin_points = self.grid_x_og[tuple(point)]
                if len(origin_points) == 0:
                    origin_points = self.grid_x[tuple(point)] + self.resolution / 2
                Yjumpfrom = float(u[tuple(point)])


                # jumpfrom point
                origin_points = np.stack(origin_points).squeeze()
                if (origin_points.ndim > 1) | ((origin_points.ndim == 1) and (J_grid.ndim == 1)): # if there are multiple points in the hypervoxel, take the mean
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
                    # get point with largest jump size
                    Yjumptos = [u[tuple(neighbors[j])] for j in range(len(neighbors))]
                    Yjumpsizes = [abs(Yjumptos[j] - Yjumpfrom) for j in range(len(neighbors))]
                    idx = np.argmax(Yjumpsizes)
                    Yjumpto = Yjumptos[idx]
                    dest_points = self.grid_x_og[tuple(neighbors[idx])]
                    if len(dest_points) == 0:
                        dest_points = self.grid_x[tuple(neighbors[idx])] + self.resolution / 2
                    dest_points = np.stack(dest_points).squeeze()

                    if (dest_points.ndim > 1 or ((dest_points.ndim == 1) and (J_grid.ndim == 1))): # if there are multiple points in the hypervoxel, take the mean
                        jumpto = np.mean(dest_points, axis = 0)
                    else:
                        jumpto = dest_points

                    # dists = [[np.linalg.norm(jumpfrom - point) for point in pointslist[j]] for j in range(len(neighbors))]
                    # idx = np.argmin([np.argmin(sublist) for sublist in dists])
                    # closest = tuple(neighbors[idx])
                    # Yjumpto = u[closest]

                    # dest_points = self.grid_x_og[closest]
                    # if len(dest_points) == 0:
                    #     dest_points = self.grid_x[closest] + self.resolution / 2
                    # dest_points = np.stack(dest_points).squeeze()

                    # if dest_points.ndim > 1: # if there are multiple points in the hypervoxel, take the mean
                    #     jumpto = np.mean(dest_points, axis = 0)
                    # else:
                    #     jumpto = dest_points
                        

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
            
            if Y_boundary.ndim == 1:
                Y_boundary = np.expand_dims(Y_boundary, -1)
        

            # create named array to return
            rays = [Y_boundary[:,d] for d in range(Y_boundary.shape[1])] + [Y_jumpfrom, Y_jumpto, Y_jumpsize]
            names = ["X_" + str(d) for d in range(Y_boundary.shape[1])] + ["Y_jumpfrom", "Y_jumpto", "Y_jumpsize"]
            jumps = pd.DataFrame(np.core.records.fromarrays(rays, names=names))
        else:
            jumps = None

        return jumps

        
    
    def pickKMeans(self, u_norm):
        plt.ioff()  # Turn off interactive mode to prevent the figure from being displayed
        out = plt.hist(u_norm)
        plt.close()  # Close the figure to free up memory

        X1 = np.tile(out[1][1:], out[0].shape[0])
        X2 = out[0].reshape(-1,1).squeeze(1)
        Z = np.stack([X1, X2], axis = 1)

        # the histogram is "bimodal" (one large cluster and a bunch of smaller ones), so we use k-means to find the edge of the first "jump" cluster
        # TODO: averages instead of closest points
        kmeans = KMeans(n_clusters=2, random_state=0, n_init = "auto").fit(Z)
        
        nu = X1[kmeans.labels_ == 1].max()
        return nu
    
    def adjustBoundary(self, u, J_grid):
        for k in range(1, len(J_grid) - 1):
            if J_grid[k] == 1:
                if (J_grid[k-1] == 0) and (J_grid[k+1]==1):
                    u[k+1] = u[k].copy()
                    J_grid[k] = 0
        return (u, J_grid)
    

        
    def boundary(self, u):

        u_diff = self.forward_differences(u, D = len(u.shape))
        # u_diff = u_diff / self.resolution # scale FD by side length
        u_norm = np.linalg.norm(u_diff, axis = 0, ord = 2) # 2-norm

        if self.pick_nu == "kmeans":
            nu = self.pickKMeans(u_norm)
        else:
            nu = np.sqrt(self.nu)


        # find the boundary on the grid by comparing the gradient norm to the threshold
        J_grid = (u_norm >= nu).astype(int)

        # adjust boundary in 1D
        if J_grid.ndim == 1:
            u, J_grid = self.adjustBoundary(u, J_grid)


        # scale u back to get correct jump sizes
        if not self.image:
            u = u * np.max(self.Y_raw, axis = 0) + np.min(self.Y_raw, axis = 0)

        ## find the boundary on the point cloud
        jumps = self.boundaryGridToData(J_grid, u, self.average)

        # test_grid = np.zeros(self.grid_y.shape)
        # for row in jumps:
        #     test_grid[tuple(row)[:-2]] = 1

        return (u, J_grid, jumps)
    
    #def treatmentEffects(self, u, J):
    
    def conformalSplit(self):

        I = list(range(self.N))
        I1 = random.sample(I, int(self.N/2))
        X_1 = self.X_raw[I1]
        Y_1 = self.Y_raw[I1]

        model = FDD(Y_1, X_1, level = self.level, lmbda = self.lmbda, nu = self.nu, iter = self.iter, tol = self.tol, resolution=self.resolution,
            pick_nu = self.pick_nu, scaled = self.scaled, scripted = self.scripted, rectangle = self.rectangle, average=self.average, CI=False)

        self.u_cs = model.run()['u']

        I2 = [i for i in I if i not in I1]
        X_2 = self.X_raw[I2]
        Y_2 = self.Y_raw[I2]

        model_temp = FDD(Y_2, X_2, level = self.level, lmbda = self.lmbda, nu = self.nu, iter = self.iter, tol = self.tol, resolution=self.resolution,
            pick_nu = self.pick_nu, scaled = self.scaled, scripted = self.scripted, rectangle = self.rectangle, average=self.average, CI=False)


        grid_x_reshaped = model.grid_x.reshape(-1, model.grid_x.shape[-1])[:,::-1]


        # Initializing a list to hold the indices of the closest points
        closest_indices = []

        # normalize X_2
        X_2 = self.normalizeData(self.rectangle, X_2)


        # Looping through each point in X_2 for image function
        for point in X_2:
            # Calculating the Euclidean distance between the point and all points in grid_x
            distances = np.linalg.norm(grid_x_reshaped - point, axis=1)

            # Finding the index of the closest point in grid_x
            closest_index = np.argmin(distances)

            # Converting the index to a tuple representing the location in the original grid
            closest_grid_index = np.unravel_index(closest_index, (self.grid_x.shape[0], self.grid_x.shape[1]))

            closest_indices.append(closest_grid_index)

        y_closest, x_closest = self.castDataToGridPoints(grid_x = model_temp.grid_x, X = X_2, Y = Y_2)
        y_closest = (y_closest) # - np.min(Y_2, axis=0)) / np.max(Y_2, axis=0)
        self.y_diff = model.forward_differences(y_closest, D = len(y_closest.shape))
        #y_closest_norm = np.linalg.norm(y_diff, axis=0, ord=2)

        # # Converting the list to a numpy array
        closest_indices = np.array(closest_indices)

        u_pred = (self.u_cs.copy()) #  - np.min(model.Y_raw,axis=0)) / np.max(model.Y_raw,axis=0)
        self.u_diff = model.forward_differences(u_pred, D = len(u_pred.shape))
        #u_norm =  np.linalg.norm(u_diff, axis = 0, ord = 2) # 2-norm

        # calculate interval length for function
        if X_2.ndim == 1:
            idx = closest_indices
        else:
            idx = tuple(closest_indices.T)
        self.R_u = np.abs(Y_2 - self.u_cs[idx])
        

        # calculate interval length for forward differences
        indexing_tuple = (slice(None),) + idx
        self.R_J = np.abs(y_diff[indexing_tuple] - self.u_diff[indexing_tuple])
        

        return self.conformalSplitBounds(self.alpha)
    
    def conformalSplitBounds(self, alpha):
        k = int(np.ceil((self.N/2 + 1)* (1-alpha)))
        d = sorted(self.R_u.flatten())[k-1]
        
        k = int(np.ceil((self.N/2 + 1)* (1-alpha)))
        d_norm = np.sort(self.R_J, axis=1)[:,k-1]

        # if the upper jump is below 0 or the lower jump above, we have a significant jump
        shp = self.R_J.shape[0]
        J_lower = (self.u_diff -  d_norm.reshape((shp,) + (1,) * (shp)))
        J_upper = (self.u_diff +  d_norm.reshape((shp,) + (1,) * (shp)))
        J_lower = (np.sum((J_lower > 0).astype(int) + (J_upper < 0).astype(int), axis=0) > 0)
        
        return (self.u_cs - d, self.u_cs + d, J_lower)
    
    @staticmethod
    def castDataToGridPoints(grid_x, X, Y):
        
        if X.ndim == 1:
            X = np.expand_dims(X, -1)

        # set up grid
        grid_y = np.zeros(list(grid_x.shape[:-1]))
        grid_x_og = np.empty(list(grid_x.shape[:-1]), dtype = object) # assign original x values as well for later

        # find closest data point for each point on grid and assign value
        # Iterate over the grid cells
        it = np.nditer(grid_x[...,0], flags = ['multi_index'])
        for x in it:
            distances = np.linalg.norm(X - grid_x[it.multi_index], axis=1, ord = 2)
            # Find the closest seed
            closest_seed = np.argmin(distances)
            # Assign the value of the corresponding data point to the grid cell
            grid_y[it.multi_index] = Y[closest_seed] #.min()

            # assign original x value
            grid_x_og[it.multi_index] = tuple(X[closest_seed,:])

        return (grid_y, grid_x_og)
        
        
    def forward_differences(self, ubar, D : int):

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
    
    def arraysToTensors(self, y, iter, level, lmbda, nu, tol):
        f = torch.tensor(y, device = self.device, dtype = torch.float32)
        repeats = torch.tensor(iter, device = self.device, dtype = torch.int32)
        level = torch.tensor(level, device = self.device, dtype = torch.int32)
        lmbda = torch.tensor(lmbda, device = self.device, dtype = torch.float32)
        nu = torch.tensor(nu, device = self.device, dtype = torch.float32)
        tol = torch.tensor(tol, device = self.device, dtype = torch.float32)
        
        return f, repeats, level, lmbda, nu, tol
    
    def processResults(self, results):
        v, nrj, eps, it = results
        v = v.cpu().detach().numpy()
        nrj = nrj.cpu().detach().numpy()
        eps = eps.cpu().detach().numpy()
        
        u = self.isosurface(v) 
        
        
        u, J_grid, jumps = self.boundary(u)
        
        return (u, jumps, J_grid, nrj, eps, it)

    

        
    
    
    def run(self):
        
        f, repeats, level, lmbda, nu, tol = \
            self.arraysToTensors(self.grid_y, self.iter, self.level, self.lmbda, self.nu, self.tol)
        
        if self.scripted:
            results = self.model(f, repeats, level, lmbda, nu, tol)
        else:
            results = self.model.forward(f, repeats, level, lmbda, nu, tol)
        
        u, jumps, J_grid, nrj, eps, it = self.processResults(results)
        
        if self.CI: # confidence intervals
            u_lower, u_upper, J_lower = self.conformalSplit()
        else:
            u_lower, u_upper, J_lower = None, None, None
        
        
        return {"u": u, "u_lower": u_lower, "u_upper": u_upper,
                "jumps": jumps, "J": J_grid, "J_lower": J_lower,  
                "nrj": nrj, "eps": eps, "it": it}
    

        
        
   



        

        

import numpy as np
import torch
from .utils import * 
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree
from .primaldual_multi_scaled_tune import PrimalDual
from matplotlib import pyplot as plt
import pandas as pd
import random
import ray

class FDR():
    r""" Free Discontinuity Regression

    Parameters
    ----------
    Y : np.array
        The outcome variable
    X : np.array
        The feature matrix
    pick_nu : str
        How to pick the threshold for the gradient norm. Options are "kmeans" or "sqrt". Default is "kmeans".
    level : int
        The level of the contour plot. Default is 16.
    lmbda : float
        The data term parameter. Default is 1.
    nu : float
        The parameter on the boundary regularity penalty. Default is 0.01.
    iter : int
        The number of iterations for the primal-dual algorithm. Default is 1000.
    tol : float
        The tolerance for the primal-dual algorithm. Default is 5e-5.
    rectangle : bool
        Whether to relax the scaling of the domain to a unit hypercube, to allow for rectangular domains. Default is False.
    image : bool
        Whether the outcome is an image, in which case we don't scale the function back to the original values. Default is False. TODO: remove argument.
    resolution : float
        The resolution of the grid. Default is None, in which case it is calculated based on the number of data points.
    grid_n : np.array
        An optional number of pre-specified grid points along each dimension, in case the user wants to impose a specific discretization. Default is None.
    scaled : bool
        Whether to scale the gradients in the primal-dual algorithm. Default is False.
    scripted : bool
        Whether to use the scripted version of the primal-dual algorithm. Default is False.
    average : bool
        Whether to average the size of the jump points along each jump direction. Default is False, in which case the direction with the largest jump size is used
        to calculate the overall jump size.
    CI : bool
        Whether to calculate conformal prediction bands for the jump points. Default is True.
    alpha : float
        The significance level for the conformal prediction bands. Default is 0.05.
    num_cpus : int
        The number of CPUs to use for the primal-dual algorithm. Default is 1.
    num_gpus : int
        The number of GPUs to use for the primal-dual algorithm. Default is 1.
    

    Example 
    -------


    
    Note
    ----



    References
    ----------

    .. bibliography::    
            

    """

    def __init__(self, Y : np.array, X : np.array, pick_nu : str="kmeans", level : int=16, 
                 lmbda : float=1, nu : float=0.01, iter : int=1000, tol : float=5e-5, rectangle : bool=False, 
                 image : bool=False, resolution : float=None, grid_n : np.array=None,
                 scaled=True, scripted=False, average=False, CI=True, alpha=0.05, num_cpus=1, num_gpus=1) -> None:

        self.device = setDevice()
        torch.set_grad_enabled(False)
        
        self.Y_raw = Y.copy() # retain original data
        self.X_raw = X.copy()
        self.Y = Y.copy() # arrays I'll pass to PyTorch
        self.X = X.copy()
        self.N = self.Y.shape[0]

        
        self.image = image
        self.level = level
        self.lmbda = lmbda
        self.iter = iter
        self.tol = tol
        self.rectangle = rectangle
        self.resolution = resolution
        self.grid_n = grid_n
        
        # confidence intervals
        self.CI = CI
        self.alpha = alpha
        self.R_u = None # u residuals conformal split
        self.R_J = None
        self.u_cs = None # conformal split u estimate
        self.u_diff = None # same for u_diff
        self.num_gpus = num_gpus
        self.num_cpus = num_cpus
        
        # for acceleration
        self.gamma = 1
        self.theta_u = 1 # placeholder
        self.theta_mu = 1
        
        self.scripted = scripted
        self.scaled = scaled
        
        self.average = average

        
        self.X = self.normalizeData(self.rectangle, self.grid_n, self.X) # scale data to unit hypercube
        # if X only 1 dimension, add a second dimension
        if self.X.ndim == 1:
            self.X = np.expand_dims(self.X, -1)
        self.grid_x_og, self.grid_x, self.grid_y = self.castDataToGrid(self.X, self.Y)
        self.grid_y = (self.grid_y - np.min(self.Y_raw, axis=0)) / (np.max(self.Y_raw, axis=0) - np.min(self.Y_raw, axis=0))
        

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
        
    def normalizeData(self, rectangle, grid_n, X):


        #min_y = np.min(self.Y, axis = 0)
        min_x = np.min(X, axis = 0)

        # self.Y = self.Y - min_y # start at 0
        X = X - min_x

        # max_y = np.max(self.Y, axis = 0)
            
        if self.grid_n is not None:
            self.resolution = 1/np.max(self.grid_n)
            x_scale = self.resolution * self.grid_n
            max_x =  np.max(X, axis=0)
            X = X * x_scale / max_x
        elif rectangle: # retain proportions between data -- should be used when units are identical along all axes
            max_x = np.max(X)
            # self.Y = self.Y / max_y
            X = X / max_x
        else: # else scale to square
            max_x = np.max(X, axis = 0)
            # self.Y = self.Y / max_y
            X = X / max_x

        return X
            

            
    
    def castDataToGridSmooth(self, X, Y):
        

        if self.grid_n is not None: # if they pre-specified the number of grid points along each dimension
            self.resolution = 1/np.max(self.grid_n)
            xmax = self.resolution * self.grid_n

        else: 
            if self.resolution is None:
                self.resolution = 1/int(self.X_raw.max(axis=0).min()) # int so we get a natural number of grid cells

            xmax = np.max(X, axis=0)



        # set up grid
        grid_x = np.meshgrid(*[np.arange(0, xmax[i], self.resolution) for i in range(X.shape[1])], indexing="ij")
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

    def boundaryGridToData(self, J_grid, u, average=False):
        k = np.array(np.where(J_grid == 1))
        Y_jumpfrom = []
        Y_jumpto = []
        Y_jumpsize = []
        Y_boundary = []
        X_jumpfrom = []
        X_jumpto = []

        for i in range(k.shape[1]):
            point = k[:, i]
            # Make sure it's safe
            if np.any(point < 0) or np.any(point >= J_grid.shape):
                continue

            # Get "from" value
            from_val = float(u[tuple(point)])
            origin_points = self.grid_x_og[tuple(point)]
            if len(origin_points) == 0:
                origin_points = self.grid_x[tuple(point)] + self.resolution / 2
            origin_points = np.stack(origin_points).squeeze()
            if (origin_points.ndim > 1) or ((origin_points.ndim == 1) and (J_grid.ndim == 1)):
                jumpfrom = np.mean(origin_points, axis=0)
            else:
                jumpfrom = origin_points

            # Search in all dimensions, up to +/-5 pixels
            best_jump = 0.0
            best_to_val = from_val
            best_to_loc = jumpfrom
            max_steps = 5
            for d in range(J_grid.ndim):
                # Check forward direction
                forward_vals = [from_val]
                forward_pts = [point.copy()]
                for step in range(1, max_steps+1):
                    nxt = forward_pts[-1].copy()
                    if nxt[d] + 1 >= J_grid.shape[d]:
                        break
                    nxt[d] += 1
                    if J_grid[tuple(nxt)] == 1:
                        forward_pts.append(nxt)
                        forward_vals.append(float(u[tuple(nxt)]))
                    else:
                        # Reached non-jump
                        forward_pts.append(nxt)
                        forward_vals.append(float(u[tuple(nxt)]))
                        break

                # Check backward direction
                backward_vals = [from_val]
                backward_pts = [point.copy()]
                for step in range(1, max_steps+1):
                    prv = backward_pts[-1].copy()
                    if prv[d] - 1 < 0:
                        break
                    prv[d] -= 1
                    if J_grid[tuple(prv)] == 1:
                        backward_pts.append(prv)
                        backward_vals.append(float(u[tuple(prv)]))
                    else:
                        backward_pts.append(prv)
                        backward_vals.append(float(u[tuple(prv)]))
                        break

                # Combine
                possible_vals = forward_vals + backward_vals
                jump_candidate = np.max(possible_vals) - np.min(possible_vals)
                if abs(jump_candidate) > abs(best_jump):
                    best_jump = jump_candidate
                    # figure out which gave the max
                    max_val = np.max(possible_vals)
                    min_val = np.min(possible_vals)
                    if (max_val - min_val) == best_jump:
                        # see which is which
                        if (max_val == np.max(forward_vals) or max_val == from_val):
                            best_to_val = max_val
                            best_idx = np.argmax(possible_vals)
                            # pick that grid location
                            if best_idx < len(forward_vals):
                                to_pt = forward_pts[best_idx]
                            else:
                                to_pt = backward_pts[best_idx - len(forward_vals)]
                        else:
                            best_to_val = min_val
                            best_idx = np.argmin(possible_vals)
                            if best_idx < len(forward_vals):
                                to_pt = forward_pts[best_idx]
                            else:
                                to_pt = backward_pts[best_idx - len(forward_vals)]
                        # coordinate
                        dest_points = self.grid_x_og[tuple(to_pt)]
                        if len(dest_points) == 0:
                            dest_points = self.grid_x[tuple(to_pt)] + self.resolution / 2
                        dest_points = np.stack(dest_points).squeeze()
                        if (dest_points.ndim > 1) or ((dest_points.ndim == 1) and (J_grid.ndim == 1)):
                            best_to_loc = np.mean(dest_points, axis=0)
                        else:
                            best_to_loc = dest_points

            Y_boundary.append((jumpfrom + best_to_loc) / 2)
            Y_jumpfrom.append(from_val)
            Y_jumpto.append(best_to_val)
            Y_jumpsize.append(best_to_val - from_val)
            X_jumpfrom.append(jumpfrom)
            X_jumpto.append(best_to_loc)

        if Y_boundary:
            Y_boundary = np.stack(Y_boundary)
            Y_jumpfrom = np.stack(Y_jumpfrom)
            Y_jumpto = np.stack(Y_jumpto)
            Y_jumpsize = np.stack(Y_jumpsize)
            if Y_boundary.ndim == 1:
                Y_boundary = np.expand_dims(Y_boundary, -1)
            rays = [Y_boundary[:, d] for d in range(Y_boundary.shape[1])] + [Y_jumpfrom, Y_jumpto, Y_jumpsize]
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
            u = u * (np.max(self.Y_raw, axis = 0) - np.min(self.Y_raw, axis = 0))  + np.min(self.Y_raw, axis = 0)

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

        model = FDR(Y_1, X_1, level = self.level, lmbda = self.lmbda, nu = self.nu, iter = self.iter, tol = self.tol, resolution=self.resolution,
            pick_nu = self.pick_nu, scaled = self.scaled, scripted = self.scripted, rectangle = self.rectangle, average=self.average, CI=False, 
            grid_n = self.grid_n)

        self.u_cs = model.run()['u']

        I2 = [i for i in I if i not in I1]
        X_2 = self.X_raw[I2]
        Y_2 = self.Y_raw[I2]

        model_temp = FDR(Y_2, X_2, level = self.level, lmbda = self.lmbda, nu = self.nu, iter = self.iter, tol = self.tol, resolution=self.resolution,
            pick_nu = self.pick_nu, scaled = self.scaled, scripted = self.scripted, rectangle = self.rectangle, average=self.average, CI=False,
            grid_n = self.grid_n)


        grid_x_reshaped = model.grid_x.reshape(-1, model.grid_x.shape[-1])[:,::-1]


        # Initializing a list to hold the indices of the closest points
        closest_indices = []

        # normalize X_2
        X_2 = self.normalizeData(self.rectangle, self.grid_n, X_2)


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
        y_diff = model.forward_differences(y_closest, D = len(y_closest.shape))
        #y_closest_norm = np.linalg.norm(y_diff, axis=0, ord=2)

        # # Converting the list to a numpy array
        closest_indices = np.array(closest_indices)

        u_pred = (self.u_cs.copy()) #  - np.min(model.Y_raw,axis=0)) / np.max(model.Y_raw,axis=0)
        self.u_diff = model.forward_differences(u_pred, D = len(u_pred.shape))
        #u_norm =  np.linalg.norm(u_diff, axis = 0, ord = 2) # 2-norm

        # calculate interval length for function
        if X_2.ndim == 1:
            idx = closest_indices[:,0]
            y_diff = y_diff.squeeze()
            self.u_diff = self.u_diff.squeeze()
            indexing_tuple = idx
        else:
            idx = tuple(closest_indices.T)
            indexing_tuple = (slice(None),) + idx
        self.R_u = np.abs(Y_2 - self.u_cs[idx])


        # calculate interval length for forward differences
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
    
    def conformalUncertainty(self, a_bins=1000):
        start = 0+1/a_bins
        for a in np.arange(start,1,1/a_bins):
            _, _, J_lower = self.conformalSplitBounds(alpha=a)
            if a == start:
                J_uc = J_lower.copy() * (1-a)
            else:
                J_replace = (J_uc == 0) & (J_lower > 0)
                J_uc[J_replace] = (1-a)
        
        return J_uc
    
    @staticmethod
    def bootstrap_trial_factory(num_gpus, num_cpus):
        @ray.remote(num_gpus=num_gpus, num_cpus=num_cpus)
        def bootstrap_trial(model, b, I, s):
            Y_raw = model.Y_raw.copy()
            X_raw = model.X_raw.copy()
            res = np.empty((len(b),) + model.grid_y.squeeze().shape)
            I_star = I.copy()
            for j in range(len(b)-1, -1, -1):
                I_star = random.sample(I_star, b[j])
                X_star = X_raw[I_star]
                Y_star = Y_raw[I_star]
                print(f"Running trial {s}")
                model_temp = FDR(Y_star, X_star, level = model.level, lmbda = model.lmbda, nu = model.nu, iter = model.iter, tol = model.tol, resolution=model.resolution,
                    pick_nu = model.pick_nu, scaled = model.scaled, scripted = model.scripted, rectangle = model.rectangle, average=model.average, CI=False, 
                    grid_n = model.grid_n)
                results = model_temp.run()
                print(f"Done with trial {s}")
                res[j,...] = results['u'] 
            return res
        return bootstrap_trial
    
    def subSampling(self, nboot=300):
        boots = list(range(nboot))
        n = self.Y_raw.shape[0]
        N = self.grid_x.size
        b = sorted(np.random.randint(low=2*N, high=2*N+0.1*n, size=4))
        I = list(range(self.Y_raw.shape[0]))
        bootstrap_trial_dynamic = self.bootstrap_trial_factory(num_gpus=self.num_gpus, num_cpus=self.num_cpus)
        results = ray.get([bootstrap_trial_dynamic.remote(self, b, I, s) for s in boots])
        
        return (results, b)
    
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
        
    @staticmethod    
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
    

        
        
   



        

        
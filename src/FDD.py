
import numpy as np
import torch
from typing import Tuple, List, Dict
from utils import * 
import cv2
from matplotlib import pyplot as plt


class FDD():
    def __init__(self, Y : np.array, X : np.array, level : int=16, 
                 lmbda : float=1, nu : float=0.01, iter : int=1000, tol : float=5e-5, rectangle : bool=False) -> None:

        self.device = self.setDevice()
        torch.set_grad_enabled(False)
        
        self.Y_raw = Y.copy() # retain original data
        self.X_raw = X.copy()
        self.Y = Y.copy() # arrays I'll pass to PyTorch
        self.X = X.copy()

        self.level = level
        self.lmbda = lmbda
        self.nu = nu
        self.iter = iter
        self.tol = tol
        self.rectangle = rectangle
        
        self.normalizeData() # scale data to unit hypercube
        self.castDataToGrid()
        
        self.model = torch.jit.load("scripted_primal_dual.pt", map_location = self.device)
        
        # TODO: exclude duplicate points (there shouldnt be any cause the variables are assumed to be continuous but anyway)
        

        
    def setDevice(self):
        if torch.cuda.is_available(): # cuda gpus
            device = torch.device("cuda")
            #torch.cuda.set_device(int(gpu_id))
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        elif torch.backends.mps.is_available(): # mac gpus
            device = torch.device("mps")
        elif torch.backends.mkl.is_available(): # intel cpus
            device = torch.device("mkl")
        torch.set_grad_enabled(True)
        return device
    
    def normalizeData(self):
        
        min_y = np.min(self.Y, axis = 0)
        min_x = np.min(self.X, axis = 0)
        
        self.Y = self.Y - min_y # start at 0
        self.X = self.X - min_x
        
        max_y = np.max(self.Y, axis = 0)
        if self.rectangle: # retain proportions between data -- should be used when units are identical along all axes
            max_x = np.max(self.X)
            self.Y = self.Y / max_y
            self.X = self.X / max_x
        else: # else scale to square
            max_x = np.max(self.X, axis = 0)
            self.Y = self.Y / max_y
            self.X = self.X / max_x
        
    def castDataToGrid(self):
        
        n = self.Y.shape[0]
        
        # calculate 0.5% quantile of distances between points
        
        # if data is large, use a random sample of 1000 points to calculate quantile
        if n > 1000:
            idx = np.random.permutation(n)
            idx = idx[:1000]
            X_sample = self.X[idx,:]
            distances = np.sqrt(np.sum((X_sample[:,None,:] - X_sample[None,:,:])**2, axis = 2))

        else:   
            distances = np.sqrt(np.sum((self.X[:,None,:] - self.X[None,:,:])**2, axis = 2))
        np.fill_diagonal(distances, 1) # remove self-comparisons
        distances = np.min(distances, axis = 0) # get closest point for each point
        qile = np.quantile(distances, 0.005) # get 0.5% quantile
        
        # pythagoras
        resolution = qile # take as side-length the 0.5% quantile of distances between points
        
        xmax = np.max(self.X, axis = 0)
        
        # set up grid
        grid_x = np.meshgrid(*[np.arange(0, xmax[i] + resolution, resolution) for i in range(X.shape[1])])
        grid_x = np.stack(grid_x, axis = -1)
        if self.Y.ndim > 1: # account for vector-valued outcomes
            grid_y = np.zeros(list(grid_x.shape[:-1]) + [self.Y.shape[1]])
        else:
            grid_y = np.zeros(list(grid_x.shape[:-1]))


        # find closest data point for each point on grid and assign value
        # Iterate over the grid cells
        it = np.nditer(grid_x[...,0], flags = ['multi_index'])
        for x in it:
            distances = np.linalg.norm(self.X - grid_x[it.multi_index], axis=1, ord = 2)
            # Find the closest seed
            closest_seed = np.argmin(distances)
            # Assign the value of the corresponding data point to the grid cell
            grid_y[it.multi_index] = self.Y[closest_seed] #.min()
        
        if self.Y.ndim == 1:
            grid_y = grid_y.reshape(grid_y.shape + (1,))
            
        self.grid_x = grid_x
        self.grid_y = grid_y
        
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
    
    def boundary(self, u):
        u_diff = self.forward_differences(u, D = len(u.shape))
        resolution = np.max(self.X, axis = 0)[0] / self.grid_x.shape[0]
        u_diff = u_diff / resolution
        u_norm = np.linalg.norm(u_diff, axis = 0, ord = 2) # 2-norm
        return (u_norm >= np.sqrt(self.nu)).astype(int)
    
    #def treatmentEffects(self, u, J):
        
        
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
        
    def run(self):
        
        f = torch.tensor(self.grid_y, device = self.device, dtype = torch.float32)
        repeats = torch.tensor(self.iter, device = self.device, dtype = torch.int32)
        level = torch.tensor(self.level, device = self.device, dtype = torch.int32)
        lmbda = torch.tensor(self.lmbda, device = self.device, dtype = torch.float32)
        nu = torch.tensor(self.nu, device = self.device, dtype = torch.float32)
        tol = torch.tensor(self.tol, device = self.device, dtype = torch.float32)
        
        results = self.model(f, repeats, level, lmbda, nu, tol)
        
        v, nrj, eps, it = results
        v = v.cpu().detach().numpy()
        nrj = nrj.cpu().detach().numpy()
        eps = eps.cpu().detach().numpy()
        
        u = self.isosurface(v)
        
        J = self.boundary(u)
        
        return (u, J, nrj, eps, it)
        

        
        
        
if __name__ == "__main__":
    def setDevice():
        if torch.cuda.is_available(): # cuda gpus
            device = torch.device("cuda")
            #torch.cuda.set_device(int(gpu_id))
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        elif torch.backends.mps.is_available(): # mac gpus
            device = torch.device("mps")
        elif torch.backends.mkl.is_available(): # intel cpus
            device = torch.device("mkl")
        torch.set_grad_enabled(True)
        return device

    # detect GPU device and set it as default
    dev = setDevice()
    g = DeviceMode(torch.device(dev))
    g.__enter__()

    image = "marylin.png"
    mIn = cv2.imread(image, (0))
    mIn = mIn.astype(np.float32)
    mIn /= 255
    
    Y = mIn.flatten()
    #Y = np.stack([Y, Y], axis = 1)
    # get labels of grid points associated with Y values in mIn
    X = np.stack([np.tile(np.arange(0, mIn.shape[0], 1), mIn.shape[1]), np.repeat(np.arange(0, mIn.shape[0], 1), mIn.shape[1])], axis = 1)
    
    # reshuffle Y and X in the same way so that it resembles normal data
    idx = np.random.permutation(Y.shape[0])
    Y = Y[idx]
    X = X[idx]
            
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

    model = FDD(Y, X, level = 16, lmbda = 0.1, nu = 5, iter = 1000, tol = 5e-5)
    u, J, nrj, eps, it = model.run()
    cv2.imwrite("result.png",u*255)

    plt.imshow(J)

    # histogram of gradient norm

    test = np.linalg.norm(forward_differences(u, D = len(u.shape)), axis = 0)
    out = plt.hist(test)

    X = np.tile(out[1][1:], out[0].shape[0])
    Y = out[0].reshape(-1,1).squeeze(1)
    Z = np.stack([X, Y], axis = 1)

    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=2, random_state=0).fit(Z)
    nu = X[kmeans.labels_ == 1].max()

    # get new boundary
    J_new = boundary(u, nu = nu**2) # the squared is just cause we're taking the square root

    

    # plt.hist(test.reshape(-1,1)[kmeans.labels_ == 2], bins = 100)

        
    # from pomegranate import  *

    # model = GeneralMixtureModel.from_samples(NormalDistribution, n_components=2, X=test.reshape(-1,1))
    # labels = model.predict(test.reshape(-1,1))

    # plt.hist(test.reshape(-1,1)[labels == 0], bins = 100)

    # plt.hist(test.reshape(-1,1),  bins = 500)
    # plt.ylim(0,5)




        

        
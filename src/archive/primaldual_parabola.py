import numpy as np
import torch
import argparse
import cv2
import sys
from tqdm import tqdm
from utils import * 
import time
from torch.profiler import profile, record_function, ProfilerActivity
import os
import itertools
import gc 
import copy
import weakref
from skimage import io

class PrimalDual:
        
    def __init__(self, i, o, parm, repeats, gray, level, lmbda, nu) -> None:
        """
        Constructor
        
        Parameters
        
        i: input image
        o: output image
        parm: parameters filename
        repeats: number of iterations
        gray: grayscale
        level: number of levels
        lmbda: lmbda
        nu: nu
        
        Returns
        
        None
        
        """
        self.parseArguments(i, o, parm, repeats, gray, level, lmbda, nu)
        
        # detect GPU device and set it as default
        self.dev = self.setDevice()
        g = DeviceMode(torch.device(self.dev))
        g.__enter__()

        # Load the input image using opencv (load as grayscale if "gray==true", otherwise as is (may be color or grayscale))
        self.mIn = cv2.imread(self.image, (0 if self.gray else 1))
        # check
        if self.mIn is None:
            print("ERROR: Could not load image " + self.image)
            sys.exit(1)
            
        # convert to float representation (opencv loads image values as single bytes by default)
        self.mIn = self.mIn.astype(np.float32)
        # convert range of each channel to [0,1] (opencv default is [0,255])
        self.mIn /= 255
        
        # initialize parameters
        self.initialize_parameters()

        # initialize output image (numpy array)
        self.mOut = np.zeros((self.h,self.w,self.nc), dtype=np.float32)  # mOut will have the same number of channels as the input image, nc layers
        
        # allocate memory on device
        self.allocateTensors()
        
        # initialize tensors    
        self.initialize() # not sure if u is being assigned correctly
        
        self.k_indices_for_mu() # initialize some indices for more efficient indexing later in parabola function
        
        
        
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
        
    
    def parseArguments(self, i, o, parm, repeats, gray, level, lmbda, nu):

        # input image
        self.image = i
        # output image
        self.output = o
        # parameter values
        self.parm = parm
        # number of computation repetitions to get a better run time measurement
        self.repeats = repeats
        # grayscale or not
        self.gray = gray
        # discretization level
        #self.l = level 
        self.l = level
        # weight on data term
        self.lmbda = lmbda
        # weight on haussdorff term
        self.nu = nu
        
    def initialize_parameters(self):
        # time-steps
        self.nrj = 0
        self.tauu = 1.0 / 6.0
        self.sigmap = 1.0 / (3.0 + self.l)
        self.sigmas = 1.0

        # get image dimensions
        iter = 1
        self.w = self.mIn.shape[1]         # width
        self.h = self.mIn.shape[0]         # height
        self.nc = self.mIn.shape[2] if self.mIn.ndim == 3 else 1  # number of channels
         
        # image dimension
        self.dim = self.h*self.w*self.nc
        #self.nbyted = self.dim # *np.dtype(np.float32).itemsize # dual?
        
        # u, un, ubar, p1, p2, p3 dimension
        self.size = self.h*self.w*self.l*self.nc
        #self.nbytes = self.size # *np.dtype(np.float32).itemsize
        
        # s1, s2, mu1, mu2, mun1, mun2, mubar1, mubar2 dimension
        self.proj = int(self.l*(self.l-1)/2 + self.l) # see eq. 4.24 in thesis -- number of non-local constraint sets
        #self.nbytep = self.proj*self.dim # *np.dtype(np.float32).itemsize # primal?
    
    def allocateTensors(self):
        # allocate raw input image array
        # h_u = new float[size];
        # h_un = new float[size];
        self.h_img = np.zeros((self.h,self.w,self.nc,), dtype = np.float32)
        # self.h_u = torch.zeros((self.h,self.w,self.nc,self.l,), dtype = torch.float32)
        # self.h_un = torch.zeros((self.h,self.w,self.nc,self.l,), dtype = torch.float32)

        self.u = torch.zeros((self.h,self.w,self.nc,self.l,), dtype = torch.float32)
        self.ubar = torch.zeros((self.h,self.w,self.nc,self.l,), dtype = torch.float32)
        self.p1 = torch.zeros((self.h,self.w,self.nc,self.l,), dtype = torch.float32)
        self.p2 = torch.zeros((self.h,self.w,self.nc,self.l,), dtype = torch.float32)
        self.p3 = torch.zeros((self.h,self.w,self.nc,self.l,), dtype = torch.float32)
        self.s1 = torch.zeros((self.h,self.w,self.nc,self.proj,), dtype = torch.float32)
        self.s2 = torch.zeros((self.h,self.w,self.nc,self.proj,), dtype = torch.float32)
        self.mu1 = torch.zeros((self.h,self.w,self.nc,self.proj,), dtype = torch.float32)
        self.mu2 = torch.zeros((self.h,self.w,self.nc,self.proj,), dtype = torch.float32)
        self.mubar1 = torch.zeros((self.h,self.w,self.nc,self.proj,), dtype = torch.float32)
        self.mubar2 = torch.zeros((self.h,self.w,self.nc,self.proj,), dtype = torch.float32)
        self.f = torch.zeros((self.h,self.w,self.nc,), dtype = torch.float32)

        
        self.h_img = self.convert_mat_to_layered(self.h_img, self.mIn.flatten())
        self.f = torch.as_tensor(self.h_img).view(self.h,self.w,self.nc,).detach().clone()
        
    def k_indices_for_mu(self):
        """Pre-compute indices for indexing mu1 and mu2"""
        # initialize dict with range(self.l) as keys
        self.k_indices = {k: [] for k in range(self.l)}
        for z in range(self.l):
            K = 0
            K_indices = []
            for k1 in range(0,self.l): # self.l - z
                for k2 in range(k1,self.l):
                    if ((z <= k2) and (z >= k1)): 
                        K_indices.append(K)
                    K += 1
            self.k_indices[z] = K_indices
            

        
        
    def initialize(self):
        self.u = self.f[:,:,:,None].repeat(1,1,1,self.l).detach()
        self.ubar = self.f[:,:,:,None].repeat(1,1,1,self.l).detach()
        
    
    # def initialize(self):
    #     for x in range(self.w):
    #         for y in range(self.h):
    #             for c in range(self.nc):
    #                 img = self.f[x * y * c]
    #                 for k in range(self.proj):
    #                     I = x + y * self.w + k * self.h * self.w + c * self.h * self.w * self.l
    #                     J = x + y * self.w + k * self.h * self.w + c * self.h * self.w * self.proj
    #                     if k < self.l:
    #                         self.u[I] = img
    #                         self.ubar[I] = img
    #                         self.p1[I] = 0.0
    #                         self.p2[I] = 0.0
    #                         self.p3[I] = 0.0
    #                     self.s1[J] = 0.0
    #                     self.s2[J] = 0.0
    #                     self.mu1[J] = 0.0
    #                     self.mu2[J] = 0.0
    #                     self.mubar1[J] = 0.0
    #                     self.mubar2[J] = 0.0
                        

        
    @staticmethod
    def convert_interleaved_to_layered(aOut, aIn, w, h, nc):
        if nc==1:
            aOut=aIn
            return aOut
        nOmega = w*h
        for y in range(h):
            for x in range(w):
                for c in range(nc):
                    aOut[x + w*y + nOmega*c] = aIn[(nc-1-c) + nc*(x + w*y)]
        return aOut

    def convert_mat_to_layered(self, aOut, mIn):
        return self.convert_interleaved_to_layered(aOut, mIn, self.mIn.shape[1], self.mIn.shape[0], self.nc)


    def run(self):
        
        
        # start timer
        self.start = time.time()
        
        #p1, p2, p3 = self.p1.detach().clone(), self.p2.detach().clone(), self.p3.detach().clone()
        

        pbar = tqdm(total=self.repeats)
        for iter in range(self.repeats):
            pbar.update(1)
            #start = time.time()
            self.parabola() # project onto parabola (set K)s
            #print("parabola: ", time.time() - start)
            #start = time.time()
            self.l2projection() # project onto l2 ball 
            #print("l2projection: ", time.time() - start)
            #start = time.time()
            self.mu() # constrain lagrange multipliers
            #print("mu: ", time.time() - start)
            if iter%10 == 0:
                h_un = self.u.detach().clone()
            #start = time.time()
            self.clipping() # project onto set C
            #print("clipping: ", time.time() - start)
            if iter%10 == 0:
                h_u = self.u.detach().clone()
                nrj = self.energy(h_u, h_un, self.size).detach().item() # calculate energy
                #del self.h_u, self.h_un
                if nrj/(self.w*self.h*self.l) <= 5*1E-5: # if tolerance criterion is met,
                    break
                    # print energy and error to pbar
                pbar.set_description("Energy: %f, Error: %f" % (nrj, nrj/(self.w*self.h*self.l)))
                del h_un, h_u, nrj
            if iter == self.repeats-1:
                print("debug")
            #print("Energy: %f, Error: %f" % (nrj, nrj/(self.w*self.h*self.l)))
            
            
    
                
        self.isosurface() # back out estimated image from superlevel sets using 0.5-isosurface and assign to self.h_img
        
        # end timer
        end = time.time()
        
        
        # write parameters
        if self.parm is None:
            self.parameterToConsole(taum = 1, iter = iter, t = end)
        else:
            self.parameterToFile(taum = 1, iter = iter, t = end)
            
            

        # save input and result
        cv2.imwrite(self.output,self.h_img*255)
        
        # delete all the tensors assigned above
        del self.u, self.ubar, self.p1, self.p2, self.p3, self.s1, self.s2, self.mu1, self.mu2, self.mubar1, self.mubar2, self.f
        
        if self.dev == "cuda":
            torch.cuda.empty_cache()




    def parabola(self):
        
        u1 = torch.cat((torch.diff(self.ubar, dim=0), torch.zeros_like(self.ubar[:1,:,:,:], dtype = torch.float32)), dim = 0)
        u2 = torch.cat((torch.diff(self.ubar, dim=1), torch.zeros_like(self.ubar[:,:1,:,:], dtype = torch.float32)), dim = 1)
        u3 = torch.cat((torch.diff(self.ubar, dim=3), torch.zeros_like(self.ubar[:,:,:,:1], dtype = torch.float32)), dim = 3)
        
        mu1sum = torch.stack(tuple([self.mu1[:,:,:,self.k_indices[z]].sum(dim=-1) for z in range(self.l)]), dim=-1)
        mu2sum = torch.stack(tuple([self.mu2[:,:,:,self.k_indices[z]].sum(dim=-1) for z in range(self.l)]), dim=-1)
        

        # Calculate u1, u2, and u3 using the formulas in the original function
        
        # this is what's causing memory trouble
        u1 = self.p1 + self.sigmap * (u1+ mu1sum)
        u2 = self.p2 + self.sigmap * (u2 + mu2sum)
        u3 = self.p3 + self.sigmap * u3
        
        img = self.f.unsqueeze(-1).repeat(1,1,1, self.l)

        # Calculate B using bound and broadcast it along the last dimension
        k = torch.arange(1, self.l+1, dtype=torch.int64).repeat(self.h, self.w, self.nc, 1)
        
        B = self.bound(u1, u2, self.lmbda, k, self.l, img)
        
        # Use mask to select elements where u3 < B
        mask = (u3 < B)

        x1 = u1.detach().clone() #[mask]
        x2 = u2.detach().clone() #[mask]
        x3 = u3.detach().clone() #[mask]
        
        y = x3 + self.lmbda * torch.pow(k / self.l - img, 2)
        norm = torch.sqrt(x1 * x1 + x2 * x2)
        #v = torch.zeros_like(norm)
        a = 2.0 * 0.25 * norm
        b = 2.0 / 3.0 * (1.0 - 2.0 * 0.25 * y)
        d = torch.zeros_like(a)
        mask_b = (b < 0)
        d = torch.where(mask_b, (a - torch.pow(torch.sqrt(-b), 3.0)) * (a + torch.pow(torch.sqrt(-b), 3.0)), 
                    torch.pow(a,2.0) + torch.pow(b, 3.0))
        c = torch.pow((a + torch.sqrt(d)), 1.0/3.0)
        mask1 = (d >= 0) & (c == 0)
        # mask2 = (d >= 0) & (c == 0)
        mask3 = (d < 0)
        
        v = torch.where(mask1, 0.0, 
                        torch.where(mask3,  2.0 * torch.sqrt(-b) * torch.cos(torch.tensor(1.0 / 3.0) * torch.acos(a /  (torch.pow(torch.sqrt(-b), 3)))),
                                     c - b / c))
        # v[mask1] = c[mask1] - b[mask1] / c[mask1] # change masked assignment
        # # v[mask2] = 0
        # v[mask3] = 2 * torch.sqrt(-b[mask3]) * torch.cos(torch.tensor(1 / 3)) * torch.acos(a[mask3] /  torch.pow(torch.sqrt(-b[mask3]), 3))
 

        self.p1 = torch.where(mask, torch.where(norm == 0, 0.0, (v / (2.0*0.25) ) * x1 / norm),
                         u1)
        self.p2 = torch.where(mask, torch.where(norm == 0, 0.0, (v / (2.0*0.25) ) * x2 / norm),
                         u2)
        self.p3 = torch.where(mask, self.bound(self.p1, self.p2, self.lmbda, k, self.l, img),
                         u3)
 
        
# for x in range(1):
#     for y in range(1):
#         for c in range(self.nc):
#             for z in range(self.l):
#                 print("---")
#                 print(x+y*self.w+c*self.h*self.w)
#                 print(x+y*self.w+z*self.h*self.w+c*self.h*self.w*self.l)

        
    def bound(self, x1, x2, lmbda, k, l, f):
        return 0.25 * (x1*x1 + x2*x2) - lmbda * torch.pow(k / l - f, 2)
    
    def on_parabola(self, x1, x2, x3, f, lmbda, k, j, l):
        # u1, u2, u3: 1D array
        # x1, x2, x3: 1D array
        # f: 1D array
        # k: int
        # j: int
        # l: int
        y = x3 + lmbda * torch.pow(k / l - f, 2)
        norm = torch.sqrt(x1 * x1 + x2 * x2)
        v = torch.zeros_like(norm)
        a = 2.0 * 0.25 * norm
        b = 2.0 / 3.0 * (1.0 - 2.0 * 0.25 * y)
        d = torch.zeros_like(a)
        mask = (b < 0)
        torch.where(mask, (a - torch.pow(torch.sqrt(-b), 3)) * (a + torch.pow(torch.sqrt(-b), 3)), 
                    torch.pow(a,2) + torch.pow(b, 3), out = d)
        c = (a + torch.sqrt(d)).pow(1 / 3)
        mask1 = (d >= 0) & (c != 0)
        mask2 = (d >= 0) & (c == 0)
        mask3 = (d < 0)
        v[mask1] = c[mask1] - b[mask1] / c[mask1]
        # v[mask2] = 0
        v[mask3] = 2 * torch.sqrt(-b[mask3]) * torch.cos((1.0 / 3.0) * torch.acos(a[mask3] /  torch.pow(torch.sqrt(-b[mask3]), 3)))
        #v[mask1] = c-b/c # torch.where(mask1, c - b / c, v, out = v)
        #v[mask3] = 2 * torch.sqrt(-b[mask3]) * torch.cos((1 / 3) * torch.acos(a[] / torch.pow(torch.sqrt(-b), 3))) #
        self.p1[j] = torch.where(norm == 0, 0, (v / (2.0+0.25) ) * x1 / norm)
        self.p2[j] = torch.where(norm == 0, 0, (v / (2.0+0.25) ) * x2 / norm)
        self.p3[j] = self.bound(self.p1[j], self.p2[j], lmbda, k, l, f)
        
        del y, norm, v, a, b, d, c, mask, mask1, mask2, mask3
        



        
    # def on_parabola(self, x1, x2, x3, f, lmbda, k, j, l):
    #     # u1, u2, u3: 1D array
    #     # x1, x2, x3: 1D array
    #     # f: 1D array
    #     # k: int
    #     # j: int
    #     # l: int
    #     y = x3 + lmbda * (k / l - f).pow(2)
    #     norm = (x1 * x1 + x2 * x2).sqrt()
    #     v = 0
    #     a = 2 * 0.25 * norm
    #     b = 2 / 3 * (1 - 2 * 0.25 * y)
    #     d = (a - (-b).sqrt().pow(3)) * (a + (-b).sqrt().pow(3)) if b < 0 else a * a + b.pow(3)
    #     c = (a + d.sqrt()).pow(1 / 3)
    #     if d >= 0:
    #         v = c - b / c if c != 0 else 0
    #     else:
    #         v = 2 * (-b).sqrt() * torch.cos((1 / 3) * torch.acos(a / (-b).sqrt().pow(3)))
    #     self.p1[j] = x1 - v
    #     self.p2[j] = x2
    #     self.p3[j] = y - lmbda * (k / l - f).pow(2)

        
        
                
    # def parabola(self):
    #     for x in range(0,self.w):
    #         for y in range(0,self.h):
    #             for z in range(0,self.l):
    #                 for c in range(0,self.nc):
    #                     I = x+y*self.w+c*self.h*self.w # index for image
    #                     J = x+y*self.w+z*self.h*self.w+c*self.h*self.w*self.l # index for

    #                     img = self.f[I]
    #                     mu1sum = 0
    #                     mu2sum = 0

    #                     val = self.ubar[J]
    #                     u1 = (self.ubar[x+(y+1)*self.w+z*self.h*self.w+c*self.h*self.w*self.l]-val) if (y+1<self.h) else 0
    #                     u2 = (self.ubar[(x+1)+y*self.w+z*self.h*self.w+c*self.h*self.w*self.l]-val) if  (x+1<self.w)   else 0
    #                     u3 = (self.ubar[x+y*self.w+(z+1)*self.h*self.w+c*self.h*self.w*self.l]-val) if (z+1<self.l) else 0

    #                     K = 0
    #                     # loop over all possible values of k1 and k2
    #                     for k1 in range(0,self.l):
    #                         for k2 in range(k1,self.l):
    #                             if (z <= k2 and z >= k1):
    #                                 mu1sum+=self.mu1[x+y*self.w+K*self.h*self.w+c*self.h*self.w*self.proj]
    #                                 mu2sum+=self.mu2[x+y*self.w+K*self.h*self.w+c*self.h*self.w*self.proj]
    #                             K+=1

    #                     u1 = self.p1[J]+self.sigmap*(u1+mu1sum)
    #                     u2 = self.p2[J]+self.sigmap*(u2+mu2sum)
    #                     u3 = self.p3[J]+self.sigmap*u3

    #                     B = self.bound(u1,u2,self.lmbda,z+1,self.l,img)
    #                     if (u3 < B):
    #                         self.on_parabola(u1,u2,u3,img,self.lmbda,z+1,J,self.l)
    #                     else:
    #                         self.p1[J] = u1
    #                         self.p2[J] = u2
    #                         self.p3[J] = u3
                            

                            
    def energy(self, u, un, size):
        nrj = torch.sum(abs(u - un))
        return nrj

                            

        
    def l2projection(self):
        m1 = self.s1 - self.sigmas * self.mubar1
        m2 = self.s2 - self.sigmas * self.mubar2
        norm = torch.sqrt(torch.tensor(torch.pow(m1,2) + torch.pow(m2,2), dtype = torch.float32))
        mask = (norm > self.nu)
        self.s1 = torch.where(mask, m1 * self.nu / norm, m1)
        self.s2 = torch.where(mask, m2 * self.nu / norm, m2)
                
    # def l2projection(self):
    #     for i in range(self.s1.shape[0]):
    #         m1 = self.s1[i] - self.sigmas * self.mubar1[i]
    #         m2 = self.s2[i] - self.sigmas * self.mubar2[i]
    #         norm = torch.sqrt(m1**2 + m2**2)
    #         self.s1[i] = m1 * self.nu / norm if norm > self.nu else m1
    #         self.s2[i] = m2 * self.nu / norm if norm > self.nu else m2
            
    # def mu(self):
    #     tau = 1 / (2 + (float)(self.proj/4))
    #     for x in range(self.w):
    #         for y in range(self.h):
    #             for c in range(self.nc):
    #                 K = 0
    #                 for k1 in range(self.l):
    #                     for k2 in range(k1,self.l):
    #                         I = x+y*self.w+K*self.h*self.w+c*self.h*self.w*self.proj
    #                         c1 = self.mu1[I]; c2 = self.mu2[I]
    #                         t1 = 0
    #                         t2 = 0
    #                         for k in range(k1,k2+1):
    #                             J = x+y*self.w+k*self.h*self.w+c*self.h*self.w*self.l
    #                             t1+=self.p1[J]
    #                             t2+=self.p2[J]
    #                         self.mu1[I] = c1+tau*(self.s1[I]-t1)
    #                         self.mu2[I] = c2+tau*(self.s2[I]-t2)
    #                         self.mubar1[I] = 2 * self.mu1[I] - c1
    #                         self.mubar2[I] = 2 * self.mu2[I] - c2
    #                         K+=1
                            
    def mu(self):    # first vectorized version
        
        # # TODO: these two versions produce different results but maybe bcs of numerical approx?
        # tau = 1 / (2 + (self.proj/4))
        # K = 0
        # for k1 in range(self.l):
        #     for k2 in range(k1,self.l):
        #         c1 = self.mu1[:,:,:,K]
        #         c2 = self.mu2[:,:,:,K]
        #         t1 = self.p1[:,:,:,k1:k2+1].sum(dim=3)
        #         t2 = self.p2[:,:,:,k1:k2+1].sum(dim=3)
        #         self.mu1[:,:,:,K] = c1+tau*(self.s1[:,:,:,K]-t1)
        #         self.mu2[:,:,:,K] = c2+tau*(self.s2[:,:,:,K]-t2)
        #         self.mubar1[:,:,:,K] = 2 * self.mu1[:,:,:,K] - c1
        #         self.mubar2[:,:,:,K] = 2 * self.mu2[:,:,:,K] - c2
        #         K+=1
        #print("---- \n mu1 ------ \n", self.mubar1.flatten().sum())
                
        tau = 1.0 / (2.0 + (float)(self.proj/4.0))
                
        k1_k2_combinations = [(k1, k2) for k1 in range(self.l) for k2 in range(k1,self.l)]

        # Stack the values of t1 and t2 for each combination of k1 and k2
        t1 = torch.stack(tuple([self.p1[:,:,:,k1:(k2+1)].sum(dim=3) for k1, k2 in k1_k2_combinations]), dim=-1)
        t2 = torch.stack(tuple([self.p2[:,:,:,k1:(k2+1)].sum(dim=3) for k1, k2 in k1_k2_combinations]), dim=-1)

        c1 = self.mu1.detach().clone()
        c2 = self.mu2.detach().clone()
        self.mu1 = c1+tau*(self.s1-t1)
        self.mu2 = c2+tau*(self.s2-t2)
        self.mubar1 = 2.0 * self.mu1 - c1
        self.mubar2 = 2.0 * self.mu2 - c2
        #print("---- \n mu1 ------ \n", self.mubar1.flatten().sum())
        
        
        

        
                
                

            


                            
    # def clipping(self):
    #     # projection onto C
    #     for x in range(0, self.w):
    #         for y in range(0, self.h):
    #             for z in range(0, self.l):
    #                 for c in range(0, self.nc):
    #                     I = x+y*self.w+self.z*self.h*self.w+c*self.h*self.w*self.l
    #                     cur = self.u[I] # current element of u
    #                     d1 = (self.p1[I] if y+1<self.h else 0) - (self.p1[x+(y-1)*self.w+z*self.h*self.w+c*self.h*self.w*self.l] if y>0 else 0)
    #                     d2 = (self.p2[I] if x+1<self.w else 0) - (self.p2[(x-1)+y*self.w+z*self.h*self.w+c*self.h*self.w*self.l] if x>0 else 0)
    #                     d3 = (self.p3[I] if z+1<self.l else 0) - (self.p3[x+y*self.w+(z-1)*self.h*self.w+c*self.h*self.w*self.l] if z>0 else 0)
    #                     D = cur+self.tau*(d1+d2+d3)
    #                     self.u[I] = 1 if z==0 else 0 if z==self.l-1 else np.min(1, np.max(0, D))
    #                     self.ubar[I] = 2 * self.u[I] - cur
    
    def clipping(self):    # first vectorized version
        # projection onto C
        
        temp = self.u.detach().clone()
        d1 = torch.cat((self.p1[:-1,:,:,:], torch.zeros(1,self.w,self.nc,self.l)), dim=0) - torch.cat((torch.zeros(1,self.w,self.nc,self.l), self.p1[:-1,:,:,:]), dim=0)
        d2 = torch.cat((self.p2[:,:-1,:,:], torch.zeros(self.h,1,self.nc,self.l)), dim=1) - torch.cat((torch.zeros(self.h,1,self.nc,self.l), self.p2[:,:-1,:,:]), dim=1)
        d3 = torch.cat((self.p3[:,:,:,:-1], torch.zeros(self.h,self.w,self.nc,1)), dim=3) - torch.cat((torch.zeros(self.h,self.w,self.nc,1), self.p3[:,:,:,:-1]), dim=3)
        
        
        D = temp+self.tauu*(d1+d2+d3)
        self.u = torch.clamp(D, min=0, max=1)
        # self.u[abs(self.u) < 1e-5] = 0 # avoid overflow
        # self.u[abs(self.u-1) < 1e-5] = 1
        self.u[:,:,:,0] = torch.ones(self.h,self.w,self.nc)
        self.u[:,:,:,self.l-1] = torch.zeros(self.h,self.w,self.nc)
        self.ubar = 2.0 * self.u - temp
        #print(self.u[0,0,0,0])

        
        

    def isosurface(self):
        # for x in range(self.w):
        #     for y in range(self.h):
        #         for c in range(self.nc):
        #             for k in range(self.l - 1):
        #                 uk0 = self.u[y,x,c,k]
        #                 uk1 = self.u[y,x,c,k+1]
        #                 if uk0 > 0.5 and uk1 <= 0.5:
        #                     val = self.interpolate(k + 1, uk0, uk1, self.l)
        #                     break
        #                 else:
        #                     val = uk1
        #             self.f[y,x,c] = val
        # # Extract the u tensor for convenience
        # u = self.u

        u = self.u.detach().cpu().numpy()
        mask = (u[:,:,:,:-1] > 0.5) & (u[:,:,:,1:] <= 0.5)
        # Find the indices of the first True value along the last dimension, and set all the following ones to False
        mask[:, :, :, 1:] = (mask[:, :, :, 1:]) & (mask.cumsum(-1)[:,:,:,:-1] < 1)

        uk0 = u[:,:,:,:-1][mask]
        uk1 = u[:,:,:,1:][mask]
        
        # get the indices of the last dimension where mask is True
        k = np.where(mask == True)[-1] + 1
        
        self.h_img = self.interpolate(k, uk0, uk1, self.l).reshape(self.h, self.w, self.nc)





    # def isosurface(self):
    #     uk0 = self.u[:,:,:,0:self.l-1]
    #     uk1 = self.u[:,:,:,1:self.l]
    #     self.f = torch.where(uk0 > 0.5, torch.where(uk1 <= 0.5, self.interpolate(uk0, uk1, self.l), uk1), uk1)
        
    
            
    @staticmethod
    def interpolate(k, uk0, uk1, l):
        return (k + (0.5 - uk0) / (uk1 - uk0)) / l


    def parameterToFile(self, taum, iter, t):
        file = open(self.parm, "w")
        if file == None:
            print("ERROR: Could not open file!")
        else:
            file.write("image: %d x %d x %d\n" % (self.w, self.h, self.nc))
            file.write("repeats: %d\n" % self.repeats)
            file.write("gray: %d\n" % self.gray)
            file.write("level: %d\n" % self.l)
            file.write("tauu: %f\n" % self.tauu)
            file.write("taum: %f\n" % taum)
            file.write("sigmas: %f\n" % self.sigmas)
            file.write("lmbda: %f\n" % self.lmbda)
            file.write("nu: %f\n" % self.nu)
            #file.write("GPU Memory: %zd - %zd = %f GB\n" % (total, available, (total-available)/pow(10,9)))
            file.write("time: %f s\n" % t)
            file.write("iterations: %d\n" % iter)
        file.close()

    def parameterToConsole(self, taum, iter, t):
        print("image: %d x %d x %d\n" % (self.w, self.h, self.nc))
        print("repeats: %d\n" % self.repeats)
        print("gray: %d\n" % self.gray)
        print("level: %d\n" % self.l)
        print("tauu: %f\n" % self.tauu)
        print("taum: %f\n" % taum)
        print("sigmas: %f\n" % self.sigmas)
        print("lmbda: %f\n" % self.lmbda)
        print("nu: %f\n" % self.nu)
        #print("GPU Memory: %zd - %zd = %f GB\n" % (self.total, self.available, (self.total-self.available)/pow(10,9)))
        print("time: %f s\n" % t)
        print("iterations: %d\n" % iter)



 
print("setting up")
test = PrimalDual(i = "marylin.png", o = "result.png", parm = "parameters.png", repeats = 1000, gray = True, level = 16, lmbda = 1, nu = 0.1)
test.run()
print("running")
# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
#      with record_function("model_inference"):
#         test.run()
test.run()

import torch
from torch import Tensor

from typing import Tuple, List, Dict
from utils import * 


class PrimalDual(torch.nn.Module):
    def __init__(self) -> None:

        super(PrimalDual, self).__init__()
        
        self.device = self.setDevice()
        torch.set_grad_enabled(False)

        

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
            
    def forward(self, f, repeats, l, lmbda, nu):
    # Original __init__ code moved here (with 'self.' removed)
    
        # repeats = int(repeats_a)
        # l = int(level_a)
        # lmbda = float(lmbda_a)
        # nu = float(nu_a)
        

        # initialize parameters
        nrj = torch.tensor(0, device = self.device)
        tauu = torch.tensor(1.0 / 6.0, device = self.device)
        sigmap = torch.tensor(1.0 / (3.0 + l), device = self.device)
        sigmas = torch.tensor(1.0, device = self.device)

        # get image dimensions
        w = f.size(dim=1)  # width
        h = f.size(dim=0)  # height
        nc = f.size(dim=2) if len(f.size()) == 3 else 1  # number oHAMf channels
  
        
        # s1, s2, mu1, mu2, mun1, mun2, mubar1, mubar2 dimension
        proj = int(l * (l - 1) / 2 + l)  # see eq. 4.24 in thesis -- number of non-local constraint sets
        
        # allocate memory on device
        u = torch.zeros(h, w, nc, l, dtype=torch.float16, device = self.device)
        ubar = torch.zeros(h, w, nc, l, dtype=torch.float16, device = self.device)
        p1 = torch.zeros(h, w, nc, l, dtype=torch.float16, device = self.device)
        p2 = torch.zeros(h, w, nc, l, dtype=torch.float16, device = self.device)
        p3 = torch.zeros(h, w, nc, l, dtype=torch.float16, device = self.device)
        s1 = torch.zeros(h, w, nc, proj, dtype=torch.float16, device = self.device)
        s2 = torch.zeros(h, w, nc, proj, dtype=torch.float16, device = self.device)
        mu1 = torch.zeros(h, w, nc, proj, dtype=torch.float16, device = self.device)
        mu2 = torch.zeros(h, w, nc, proj, dtype=torch.float16, device = self.device)
        mubar1 = torch.zeros(h, w, nc, proj, dtype=torch.float16, device = self.device)
        mubar2 = torch.zeros(h, w, nc, proj, dtype=torch.float16, device = self.device)

        
        # remove brackets in all lines above as in the first one
        
        
        # initialize tensors    
        u = f[:,:,:,None].repeat(1,1,1,l).detach()
        ubar = f[:,:,:,None].repeat(1,1,1,l).detach()
        
        k_indices = torch.jit.annotate(Dict[int, List[int]], {})
        for z in range(int(l)):
            K = 0
            K_indices : List[int] = []
            for k1 in range(0,int(l)): # l - z
                for k2 in range(int(k1),int(l)):
                    if ((z <= k2) and (z >= k1)): 
                        K_indices.append(K)
                    K += 1
            k_indices[z] = K_indices
        h_un = torch.zeros_like(u, device = self.device)  # Initialize h_un with a default value (None in this case)
        h_u = torch.zeros_like(u, device = self.device)  # Initialize h_u with a default value (None in this case)
        nrj = torch.tensor([0], device = self.device)  # Initialize nrj with a default value (None in this case)
        for iter in range(int(repeats)):

            p1, p2, p3 = self.parabola(p1, p2, p3, ubar, mu1, mu2, lmbda, l, f, k_indices, h, w, nc, sigmap) # project onto parabola (set K)s

            s1, s2 = self.l2projection(s1, s2, mubar1, mubar2, sigmas, nu) # project onto l2 ball 
            #print("l2projection: ", time.time() - start)
            #start = time.time()
            mu1, mu2, mubar1, mubar2 = self.mu(p1, p2, s1, s2, mu1, mu2, proj, l) # constrain lagrange multipliers
            #print("mu: ", time.time() - start)
            if iter%10 == 0:
                h_un = u.detach().clone()
            u, ubar = self.clipping(p1, p2, p3, u, tauu, h, w, nc, l) # project onto set C
            if iter%10 == 0:
                h_u = u.detach().clone()
                nrj = self.energy(h_u, h_un) # .detach().item() # calculate energy
                if torch.le(nrj/(w*h*l), torch.tensor(5*1E-5)): # if tolerance criterion is met,
                    break
 
            # if torch.equal(iter, repeats-1):
            #     print("debug")
            
        return u
        




    #@torch.jit.script
    def parabola(self, p1, p2, p3, ubar, mu1, mu2, lmbda, l, f, k_indices : Dict[int, List[int]], h : int, w : int, nc : int, sigmap):
        
        u1 = torch.cat((torch.diff(ubar, dim=0), torch.zeros_like(ubar[:1,:,:,:], dtype = torch.float16, device = self.device)), dim = 0)
        u2 = torch.cat((torch.diff(ubar, dim=1), torch.zeros_like(ubar[:,:1,:,:], dtype = torch.float16, device = self.device)), dim = 1)
        u3 = torch.cat((torch.diff(ubar, dim=3), torch.zeros_like(ubar[:,:,:,:1], dtype = torch.float16, device = self.device)), dim = 3)
        

        mu1sum_list = []
        mu2sum_list = []

        for z in range(int(l)):
            mu1sum_temp = mu1[:, :, :, k_indices[z]].sum(dim=-1)
            mu2sum_temp = mu2[:, :, :, k_indices[z]].sum(dim=-1)
            mu1sum_list.append(mu1sum_temp)
            mu2sum_list.append(mu2sum_temp)

        mu1sum = torch.stack(mu1sum_list, dim=-1)
        mu2sum = torch.stack(mu2sum_list, dim=-1)

        # Calculate u1, u2, and u3 using the formulas in the original function
    
        u1 = p1 + sigmap * (u1+ mu1sum)
        u2 = p2 + sigmap * (u2 + mu2sum)
        u3 = p3 + sigmap * u3
        
        img = f.unsqueeze(-1).repeat(1,1,1, l)

        # Calculate B using bound and broadcast it along the last dimension
        k = torch.arange(1, l+1, dtype=torch.int64, device = self.device).repeat(h, w, nc, 1)
        
        B = self.bound(u1, u2, lmbda, k, l, img)
        
        # Use mask to select elements where u3 < B
        mask = (u3 < B)

        x1 = u1.detach().clone() #[mask]
        x2 = u2.detach().clone() #[mask]
        x3 = u3.detach().clone() #[mask]
        
        y = x3 + lmbda * torch.pow(k / l - img, 2)
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
        mask3 = (d < 0)
        
        v = torch.where(mask1, 0.0, 
                        torch.where(mask3,  2.0 * torch.sqrt(-b) * torch.cos(1.0 / 3.0 * torch.acos(a /  (torch.pow(torch.sqrt(-b), 3)))),
                                     c - b / c))

        p1 = torch.where(mask, torch.where(norm == 0, 0.0, (v / (2.0*0.25) ) * x1 / norm),
                         u1)
        p2 = torch.where(mask, torch.where(norm == 0, 0.0, (v / (2.0*0.25) ) * x2 / norm),
                         u2)
        p3 = torch.where(mask, self.bound(p1, p2, lmbda, k, l, img),
                         u3)
        
        return p1, p2, p3
 

        
    #@torch.jit.script
    def bound(self, x1, x2, lmbda, k, l, f):
        return 0.25 * (x1*x1 + x2*x2) - lmbda * torch.pow(k / l - f, 2)
    
                     
    #@torch.jit.script
    def energy(self, u, un):
        nrj = torch.sum(torch.abs(u - un))
        return nrj

                        
        
    #@torch.jit.script
    def l2projection(self, s1, s2, mubar1, mubar2, sigmas, nu):
        m1 = s1 - sigmas * mubar1
        m2 = s2 - sigmas * mubar2
        norm = torch.sqrt(torch.pow(m1,2) + torch.pow(m2,2))
        mask = (norm > nu)
        s1 = torch.where(mask, m1 * nu / norm, m1)
        s2 = torch.where(mask, m2 * nu / norm, m2)
        
        return s1, s2

    #@torch.jit.script
    def mu(self, p1, p2, s1, s2, mu1, mu2, proj : int, l : int):
        tau = 1.0 / (2.0 + (proj/4.0))

        k1_k2_combinations = []
        for k1 in range(int(l)):
            for k2 in range(int(k1), int(l)):
                k1_k2_combinations.append(torch.tensor([k1, k2]).unsqueeze(0))

        k1_k2_combinations = torch.cat(k1_k2_combinations, dim=0)

        # t1 = torch.stack(tuple([p1[:,:,:,k[0].item():(k[1].item()+1)].sum(dim=3) for k in k1_k2_combinations]), dim=-1)
        # t2 = torch.stack(tuple([p2[:,:,:,k[0].item():(k[1].item()+1)].sum(dim=3) for k in k1_k2_combinations]), dim=-1)
        
        t1_list = []
        t2_list = []
        for k in k1_k2_combinations:
            t1_list.append(p1[:, :, :, k[0].item():(k[1].item() + 1)].sum(dim=3))
            t2_list.append(p2[:, :, :, k[0].item():(k[1].item() + 1)].sum(dim=3))

        t1 = torch.stack(t1_list, dim=-1)
        t2 = torch.stack(t2_list, dim=-1)


        c1 = mu1.detach().clone()
        c2 = mu2.detach().clone()
        mu1 = c1+tau*(s1-t1)
        mu2 = c2+tau*(s2-t2)
        mubar1 = 2.0 * mu1 - c1
        mubar2 = 2.0 * mu2 - c2
        
        return mu1, mu2, mubar1, mubar2

    #@torch.jit.script
    def clipping(self, p1, p2, p3, u, tauu, h : int, w : int, nc : int, l):
        temp = u.detach().clone()
        d1 = torch.cat((p1[:-1,:,:,:], torch.zeros(1,w,nc,l, device = self.device)), dim=0) - torch.cat((torch.zeros(1,w,nc,l, device = self.device), p1[:-1,:,:,:]), dim=0)
        d2 = torch.cat((p2[:,:-1,:,:], torch.zeros(h,1,nc,l, device = self.device)), dim=1) - torch.cat((torch.zeros(h,1,nc,l, device = self.device), p2[:,:-1,:,:]), dim=1)
        d3 = torch.cat((p3[:,:,:,:-1], torch.zeros(h,w,nc,1, device = self.device)), dim=3) - torch.cat((torch.zeros(h,w,nc,1, device = self.device), p3[:,:,:,:-1]), dim=3)
        D = temp+tauu*(d1+d2+d3)
        u = torch.clamp(D, min=0, max=1)
        u[:,:,:,0] = torch.ones(h,w,nc, device = self.device)
        u[:,:,:,l-1] = torch.zeros(h,w,nc, device = self.device)
        ubar = 2.0 * u - temp
        
        return u, ubar


        
    
    
if __name__ == "__main__":
    f = torch.randn(10, 10, 1)
    repeats = torch.tensor(10)
    level = torch.tensor(16)
    lmbda = torch.tensor(1)
    nu = torch.tensor(0.1)
    
    PD = PrimalDual()
    PD.forward(f, repeats, level, lmbda, nu)
    
    
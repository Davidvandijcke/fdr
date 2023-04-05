
from torch import Tensor
import torch
from typing import Tuple, List, Dict
from utils import * 


class PrimalDual(torch.nn.Module):
    def __init__(self) -> None:

        super(PrimalDual, self).__init__()
        
        torch.set_grad_enabled(False)

        

    # def setDevice(self):
    #     if torch.cuda.is_available(): # cuda gpus
    #         device = torch.device("cuda")
    #         #torch.cuda.set_device(int(gpu_id))
    #         torch.set_default_tensor_type('torch.cuda.FloatTensor')
    #     elif torch.backends.mps.is_available(): # mac gpus
    #         device = torch.device("mps")
    #     elif torch.backends.mkl.is_available(): # intel cpus
    #         device = torch.device("mkl")
    #     torch.set_grad_enabled(True)
    #     return device
            
    def forward(self, f, repeats, l, lmbda, nu, tol):
    # Original __init__ code moved here (with 'self.' removed)
    
        # repeats = int(repeats_a)
        # l = int(level_a)
        # lmbda = float(lmbda_a)
        # nu = float(nu_a)
        
        dev = f.device
        res = torch.tensor(1 / f.shape[0], device = dev)
        
        # initialize parameters
        nrj = torch.tensor(0, device=dev) 
        tw = torch.tensor(12, dtype=torch.float32, device = dev)
        tauu =  torch.tensor( res * 1.0 / 6.0, device=dev) # *res
        sigmap = torch.tensor(res * (1.0 / (3.0 + l)) , device=dev) # *res
        sigmas = torch.tensor(1.0, device=dev) 

        # get image dimensions
        dim = len(f.size())
        
        dims = [f.size(dim = x) for x in range(dim)]
        
        # s1, s2, mu1, mu2, mun1, mun2, mubar1, mubar2 dimension
        proj = int(l * (l - 1) / 2 + l)  # see eq. 4.24 in thesis -- number of non-local constraint sets
        tau =  (1.0 / (2.0 + (proj/4.0)))  

        
        # allocate memory on device
        u = torch.zeros(dims + [int(l)], dtype=torch.float32, device=dev)
        ubar = torch.zeros(dims + [int(l)], dtype=torch.float32, device=dev)
        px = torch.zeros([dim-1] + dims + [int(l)], dtype=torch.float32, device=dev)
        pt = torch.zeros(dims + [int(l)], dtype=torch.float32, device=dev)
        # p1 = torch.zeros(h, w, nc, l, dtype=torch.float32)
        # p2 = torch.zeros(h, w, nc, l, dtype=torch.float32)
        # p3 = torch.zeros(h, w, nc, l, dtype=torch.float32)
        sx = torch.zeros([dim-1] + dims + [proj], dtype=torch.float32, device=dev)
        mux = torch.zeros([dim-1] + dims + [proj], dtype=torch.float32, device=dev)
        mubarx = torch.zeros([dim-1] + dims + [proj], dtype=torch.float32, device=dev)
        # s1 = torch.zeros(h, w, nc, proj, dtype=torch.float32)
        # s2 = torch.zeros(h, w, nc, proj, dtype=torch.float32)
        # mu1 = torch.zeros(h, w, nc, proj, dtype=torch.float32)
        # mu2 = torch.zeros(h, w, nc, proj, dtype=torch.float32)
        # mubar1 = torch.zeros(h, w, nc, proj, dtype=torch.float32)
        # mubar2 = torch.zeros(h, w, nc, proj, dtype=torch.float32)

        
        # remove brackets in all lines above as in the first one
        
        
        # initialize tensors    
        u = torch.stack([f] * l, dim=-1).detach()
        ubar = u.clone().detach()
        
        # preallocate indices for parabola projection set
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
        
        # preallocate indices for helper variable
        k1_k2_combinations =  torch.jit.annotate(List[Tensor], []) # TODO: take this out of loops
        for k1 in range(int(l)):
            for k2 in range(int(k1), int(l)):
                k1_k2_combinations.append(torch.tensor([k1, k2]).unsqueeze(0))

        k1_k2_combinations = torch.cat(k1_k2_combinations, dim=0)
            
        h_un = torch.zeros_like(u)  # Initialize h_un with a default value (None in this case)
        h_u = torch.zeros_like(u)  # Initialize h_u with a default value (None in this case)
        nrj = torch.tensor([0], device = dev)  # Initialize nrj with a default value (None in this case)
        it_total = 0
        
        # START loop
        for it in range(int(repeats)):

            px, pt = self.parabola(px, pt, ubar, mux, lmbda, l, f, k_indices, dims, sigmap) # project onto parabola (set K)s

            sx = self.l2projection(sx, mubarx, sigmas, nu) # project onto l2 ball 
            #print("l2projection: ", time.time() - start)
            #start = time.time()
            mux, mubarx = self.mu(px, sx, mux, proj, l, k1_k2_combinations, tau) # constrain lagrange multipliers
            #print("mu: ", time.time() - start)
            if it%10 == 0:
                h_un = u.detach().clone()
            u, ubar = self.clipping(px, pt, u, tauu, dims, l) # project onto set C
            if it%10 == 0:
                h_u = u.detach().clone()
                nrj = self.energy(h_u, h_un) # .detach().item() # calculate energy
                if torch.le(nrj/(torch.prod(torch.tensor(dims[:-1] + [int(l)]))), tol): # if tolerance criterion is met,
                    it_total = it
                    break
 
            # if torch.equal(iter, repeats-1):
            #     print("debug")
        
        return (u, nrj, nrj/(torch.prod(torch.tensor(dims[:-1] + [int(l)]))), it_total)
        
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
            zeros = torch.zeros(zeros_shape, device=ubar.device)
            diff = torch.cat((torch.diff(ubar, dim=dim), zeros), dim=dim)  / (1 / ubar.shape[0])
            diffs.append(diff)

        # Stack the results along a new dimension (first dimension)
        u_star = torch.stack(diffs, dim=0)

        return u_star


    #@torch.jit.script
    def parabola(self, px, pt, ubar, mux, lmbda, l, f, k_indices : Dict[int, List[int]], dims : List[int], sigmap):

        # take forward differences
        ux = self.forward_differences(ubar, len(dims)-1)
        # u1 = torch.cat((torch.diff(ubar[...], dim=0), torch.zeros_like(ubar[:1,...], dtype = torch.float16)), dim = 0)
        # (u1 == ux[0,...]).all()
        ut = (torch.cat((torch.diff(ubar, dim=-1), \
                         torch.zeros_like(ubar[...,:1], dtype = torch.float32)), dim = -1))  / (1 / ubar.shape[-1])
        
        musum_list = []

        for z in range(int(l)):

            #musum_temp = mux[..., k_indices[z]].sum(dim=-1)
            musum_temp = torch.index_select(mux, -1, torch.tensor(k_indices[z],  device = px.device)).sum(dim=-1)

            musum_list.append(musum_temp)

        musum = torch.stack(musum_list, dim=-1)

        # Calculate u1, u2, and u3 using the formulas in the original function
    
        ux = px + sigmap * (ux + musum)
        ut = pt + sigmap * ut
        
        img = torch.stack([f] * l, dim=-1)

        # Calculate B using bound and broadcast it along the last dimension
        k = torch.arange(1, int(l)+1, dtype=torch.int64, device = px.device).repeat(dims + [1])
        
        B = self.bound(ux, lmbda, k, l, img) # chcked
        
        # Use mask to select elements where u3 < B
        mask = (ut < B)

        xx = ux.detach().clone() #[mask]
        xt = ut.detach().clone() #[mask]
        
        y = xt + lmbda * torch.pow(k / l - img, 2)
        norm = torch.sqrt(torch.sum(xx * xx, dim=0))

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

        px = torch.where(torch.stack([mask] * ux.size(dim = 0), dim=0), 
                         torch.where(torch.stack([norm] * ux.size(dim = 0), dim=0) == 0, 0.0, 
                                     (v / (2.0*0.25) ) * xx / norm),
                         ux)
        pt = torch.where(mask, self.bound(px, lmbda, k, l, img),
                         ut)
        
        return px, pt
 

        
    #@torch.jit.script
    def bound(self, x, lmbda, k, l, f):
        return 0.25 * torch.sum(x * x, dim=0) - lmbda * torch.pow(k / l - f, 2)

    
                     
    #@torch.jit.script
    def energy(self, u, un):
        nrj = torch.sum(torch.abs(u - un))
        return nrj

                        
        
    #@torch.jit.script
    def l2projection(self, sx, mubarx, sigmas, nu):
        mx = sx - sigmas * mubarx
        norm = torch.sqrt(torch.sum(mx * mx, dim = 0))
        
        mask = (norm > (nu * sx.shape[1]))
        sx = torch.where(torch.stack([mask] * mx.size(dim = 0), dim=0), mx * nu / norm, mx)
        
        return sx

    #@torch.jit.script
    def mu(self, px, sx, mux, proj : int, l : int, k1_k2_combinations, tau : float):

        # t1 = torch.stack(tuple([p1[:,:,:,k[0].item():(k[1].item()+1)].sum(dim=3) for k in k1_k2_combinations]), dim=-1)
        # t2 = torch.stack(tuple([p2[:,:,:,k[0].item():(k[1].item()+1)].sum(dim=3) for k in k1_k2_combinations]), dim=-1)
        
        t_list = []
        for k in k1_k2_combinations:
            t_list.append(px[..., k[0].item():(k[1].item() + 1)].sum(dim=-1))

        t = torch.stack(t_list, dim=-1)

        cx = mux.detach().clone()
        mux = cx+tau*(sx-t)
        mubarx = 2.0 * mux - cx
        
        return mux, mubarx

    #@torch.jit.script
    def clipping(self, px, pt, u, tauu, dims : List[int], l):
        
        temp = u.detach().clone()
        
        # take backward differences
        dx = self.backward_differences(px, px.size(dim = 0))
        dt = (torch.cat((pt[...,:-1], torch.zeros(dims + [1], device = u.device)), dim=-1) - \
            torch.cat((torch.zeros(dims + [1], device = u.device), pt[...,:-1]), dim=-1))  / (1 / pt.shape[-1])
        
        D = temp+tauu*(torch.sum(dx, dim=0)+dt)
        u = torch.clamp(D, min=0, max=1)
        u[...,0] = torch.ones(dims, device = u.device)
        u[...,int(l)-1] = torch.zeros(dims, device = u.device)
        ubar = 2.0 * u - temp
        return u, ubar
    
    
    def backward_differences(self, p, dims : int):
        output = []

        for i in range(dims):
            # zeros_shape = [1 if j == i else p[i].shape[j] for j in range(p[0].dim())]
            # zeros = torch.zeros(*zeros_shape)
            zeros_shape = list(p[i].shape)
            zeros_shape[i] = 1
            zeros = torch.zeros(zeros_shape, device = p.device)

            before_cat = torch.cat((p[i].narrow(i, 0, p[i].shape[i] - 1), zeros), dim=i)
            after_cat = torch.cat((zeros, p[i].narrow(i, 0, p[i].shape[i] - 1)), dim=i)
            diff = (before_cat - after_cat)  / (1 / p.shape[1])
            output.append(diff)

        result = torch.stack(output, dim=0)
        return result


 

        
        
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

    f = torch.randn(10, 10, 1, device = dev)
    repeats = torch.tensor(10, device = dev)
    level = torch.tensor(16)
    lmbda = torch.tensor(1)
    nu = torch.tensor(0.1)
    tol = torch.tensor(1e-5)
    
    # import numpy as np
    # import cv2
    # image = "resources/images/marylin.png"
    # mIn = cv2.imread(image, (0))
    # mIn = mIn.astype(np.float32)
    # mIn /= 255
    # f = torch.tensor(mIn, device = dev)
    
    # model = PrimalDual()
    # u, nrj, eps, it = model.forward(f, repeats, level, lmbda, nu, tol)
    
    pd = PrimalDual()
    pd = pd.to(dev)
    
    scripted_primal_dual = torch.jit.script(pd, example_inputs = [f, repeats, level, lmbda, nu, tol])
    torch.jit.save(scripted_primal_dual, 'scripted_primal_dual.pt')
    
    # test = scripted_primal_dual(f, repeats, level, lmbda, nu, tol)
    
    
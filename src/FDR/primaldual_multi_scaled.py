
import math
from torch import Tensor
import torch
from typing import Tuple, List, Dict
#from .utils import * 


def step_bound(side: int, dimensions: int, levels: int) -> float:
    """Upper bound for the norm of the preconditioned coupled PDHG operator."""
    t = 1.0 / (2 * math.sqrt(float(dimensions + 1)) * side)
    count = levels * (levels + 1) // 2
    # B maps a vector of lifted values to all contiguous interval sums.
    norm_b = math.sqrt(float(levels + 1)) / (2 * math.sin(math.pi / (2 * (levels + 1))))
    a = t * 2 * math.sqrt(float(dimensions * side * side + levels * levels))
    b, e = norm_b * math.sqrt(t / count), 1.0 / math.sqrt(float(count))
    trace = a * a + b * b + e * e
    return math.sqrt((trace + math.sqrt(max(0.0, trace * trace - 4 * a * a * e * e))) / 2)


def interval_sums(field: Tensor, starts: Tensor, stops: Tensor) -> Tensor:
    prefix = torch.cat([torch.zeros_like(field[..., :1]), field.cumsum(-1)], dim=-1)
    return prefix.index_select(-1, stops) - prefix.index_select(-1, starts)


def interval_ball(field: Tensor, radius: float) -> Tensor:
    norm = torch.sqrt(torch.sum(field * field, dim=0))
    # In particular a zero radius maps the origin to itself.
    safe = torch.where(norm > 0, norm, torch.ones_like(norm))
    return field * (radius / safe).clamp(max=1.0)


def project_parabola(x: Tensor, t: Tensor, fidelity: Tensor) -> Tuple[Tensor, Tensor]:
    """Projection onto t >= ||x||^2/4 - fidelity, independently at each site."""
    length = torch.sqrt((x * x).sum(dim=0))
    outside = t < .25 * (x * x).sum(dim=0) - fidelity
    a = .5 * length
    b = (2.0 / 3.0) * (1.0 - .5 * (t + fidelity))
    negative_b = torch.sqrt((-b).clamp(min=0)).pow(3)
    discriminant = torch.where(b < 0, (a-negative_b)*(a+negative_b), a*a+b.pow(3))
    c = (a + torch.sqrt(discriminant.clamp(min=0))).pow(1.0 / 3.0)
    safe_c = torch.where(c > 0, c, torch.ones_like(c))
    q = b / safe_c
    # c - b/c cancels at small positive roots. Rationalize the cubic there.
    denominator = safe_c*safe_c + b + q*q
    positive_root = torch.where(b >= 0, 2*a/denominator, c-q)
    negative_b = torch.where(negative_b > 0, negative_b, torch.ones_like(negative_b))
    angle = torch.acos((a/negative_b).clamp(-1.0, 1.0)) / 3
    root = torch.where(discriminant < 0, 2*torch.sqrt((-b).clamp(min=0))*torch.cos(angle), positive_root)
    safe_length = torch.where(length > 0, length, torch.ones_like(length))
    projected_x = torch.where(outside, 2*root*x/safe_length, x)
    projected_t = torch.where(outside, .25*(projected_x*projected_x).sum(dim=0)-fidelity, t)
    return projected_x, projected_t


class PrimalDual(torch.nn.Module):
    """Scaled, unit-density FDR solver. The scripted and eager paths share each update."""

    step_safety = 1.0
    fused_intervals = False

    def __init__(self, step_safety: float = 1.0, fused_intervals: bool = False) -> None:
        super().__init__()
        if not 0 < step_safety <= 1:
            raise ValueError("step_safety must lie in (0, 1]")
        self.step_safety = float(step_safety)
        self.fused_intervals = fused_intervals

    @torch.jit.unused
    def fused_update(self, px, slack, mu, mu_bar, starts, stops, radius, sigma, tau):
        from ._cuda_intervals import update
        return update(px, slack, mu, mu_bar, starts, stops, radius, sigma, tau)

    def forward(self, f: Tensor, repeats: int, l: int, lmbda: float, nu: float, tol: float):
        if f.dim() < 2 or f.numel() == 0 or l < 2 or repeats < 1:
            raise ValueError("Provide a spatial grid, at least two levels and a positive iteration count")
        if nu < 0 or lmbda < 0 or tol < 0:
            raise ValueError("lambda, nu and tolerance must be nonnegative")
        # The numeric parameters of the original six-argument eager API can
        # also be passed as scalar tensors (the public FDR entry point does so).
        f = f.detach()
        if self.fused_intervals and (torch.jit.is_scripting() or not f.is_cuda or f.dtype != torch.float32):
            raise ValueError("fused_intervals requires the eager float32 CUDA solver")
        levels, cap = int(l), int(repeats)
        n, dim = f.size(0), f.dim()-1
        safety = min(self.step_safety, .99/step_bound(n, dim, levels))
        spatial_step = safety / (2*math.sqrt(float(dim+1))*n)
        count = levels*(levels+1)//2
        radius = levels*float(nu)
        dims = list(f.shape)
        u = f.detach().unsqueeze(-1).expand(dims+[levels]).clone()
        u_bar = u
        px = torch.zeros([dim]+dims+[levels], device=f.device, dtype=f.dtype)
        pt = torch.zeros_like(u)
        slack = torch.zeros([dim]+dims+[count], device=f.device, dtype=f.dtype)
        mu = torch.zeros_like(slack)
        mu_bar = mu
        # Construct integer intervals on CPU for older MPS backends.
        pairs = torch.triu_indices(levels, levels, device="cpu").to(f.device)
        starts, stops = pairs[0], pairs[1]+1
        nodes = torch.arange(levels, device=f.device)
        incidence = ((starts.unsqueeze(1) <= nodes) & (nodes < stops.unsqueeze(1))).to(f.dtype)
        fidelity = float(lmbda) * ((nodes.to(f.dtype)+1)/levels-f.unsqueeze(-1)).square()
        change = torch.zeros((), device=f.device, dtype=f.dtype)
        iterations = cap
        for it in range(cap):
            px, pt = self.dual_update(px, pt, u_bar, mu_bar, incidence, fidelity, spatial_step)
            if not torch.jit.is_scripting() and self.fused_intervals:
                slack, mu, mu_bar = self.fused_update(px,slack,mu,mu_bar,starts,stops,radius,safety,safety/count)
            else:
                slack = interval_ball(slack - safety*mu_bar, radius)
                delta = (safety/count) * (slack - interval_sums(px, starts, stops))
                mu_bar = mu + 2.0*delta
                mu = mu + delta
            previous = u
            u, u_bar = self.primal_update(px, pt, u, spatial_step)
            if it % 10 == 0 or it+1 == cap:
                change = torch.abs(u-previous).sum()
                if change/u.numel() <= float(tol) and self.stationary(
                        u, px, pt, slack, mu, starts, stops, incidence, fidelity, radius, float(tol)):
                    iterations = it
                    break
        return u, change, change/u.numel(), iterations

    def dual_update(self, px: Tensor, pt: Tensor, u: Tensor, mu: Tensor,
                    incidence: Tensor, fidelity: Tensor, step: float) -> Tuple[Tensor, Tensor]:
        dx = self.forward_differences(u, px.size(0))
        dt = torch.cat((u[..., 1:]-u[..., :-1], torch.zeros_like(u[..., :1])), dim=-1)*u.size(-1)
        # Flatten the batch for backends that only support a 2D matrix product.
        adjoint = (mu.reshape(-1, incidence.size(0)) @ incidence).reshape_as(px)
        return project_parabola(px+step*(dx+adjoint), pt+step*dt, fidelity)

    def primal_update(self, px: Tensor, pt: Tensor, u: Tensor, step: float) -> Tuple[Tensor, Tensor]:
        dx = self.backward_differences(px, px.size(0)).sum(dim=0)
        zero = torch.zeros_like(pt[..., :1])
        dt = (torch.cat([pt[..., :-1], zero], -1) - torch.cat([zero, pt[..., :-1]], -1))*pt.size(-1)
        updated = (u+step*(dx+dt)).clamp(0.0, 1.0)
        updated[..., 0] = 1.0
        updated[..., -1] = 0.0
        return updated, 2.0*updated-u

    def stationary(self, u: Tensor, px: Tensor, pt: Tensor, slack: Tensor, mu: Tensor,
                   starts: Tensor, stops: Tensor, incidence: Tensor, fidelity: Tensor,
                   radius: float, tol: float) -> bool:
        threshold = math.sqrt(tol)
        difference = interval_sums(px, starts, stops)-slack
        if torch.sqrt((difference*difference).sum(dim=0)).max() > (1+radius)*threshold:
            return False
        if (u[..., 1:]-u[..., :-1]).max() > threshold:
            return False
        xp, tp = self.dual_update(px, pt, u, mu, incidence, fidelity, 1.0)
        sp = interval_ball(slack-mu, radius)
        up, _ = self.primal_update(px, pt, u, 1.0)
        for new, old in [(xp, px), (tp, pt), (sp, slack), (up, u)]:
            if (new-old).abs().max() > (1+old.abs().max())*threshold:
                return False
        return True
        
    def updateStepSizes(self, tau_u, tau, sigma_p, sigma_s, gamma_u, gamma_mu, theta_u, theta_mu):
        theta_u = 1 / torch.sqrt(2*gamma_u*tau_u)
        theta_mu = 1 / torch.sqrt(2*gamma_mu*tau)
        
        # above 3 lines but remove self
        tau_u = tau_u * theta_u
        tau = tau * theta_mu
        sigma_p = sigma_p / theta_u
        sigma_s = sigma_s / theta_mu
        
        return tau_u, tau, sigma_p, sigma_s
    
    def adaptiveStepSizes(u, mux, px, pt, sx, u_old, mux_old, px_old, pt_old, sx_old,
                          sigmap, sigmas, tauu, tau, alpha, eta):
        c = 0.9
        
        u_norm = torch.linalg.vector_norm(u-u_old, ord=2)**2
        mux_norm = torch.linalg.vector_norm(mux-mux_old, ord=2)**2
        px_norm = torch.linalg.vector_norm(px-px_old, ord=2)**2
        pt_norm = torch.linalg.vector_norm(pt-pt_old, ord=2)**2
        sx_norm = torch.linalg.vector_norm(sx-sx_old, ord=2)**2
        # backtrack = (c / (2*tauu) * u_norm + c / (2*tau) * mux_norm + 
        #              c / (2*sigmap) * px_norm + c / (2*sigmas) * sx_norm) - 
        #             2*()
    
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
            diff = torch.cat((torch.diff(ubar, dim=dim), zeros), dim=dim)  / (1 / ubar.shape[0]) # / (1 / ubar.shape[0])
            diffs.append(diff)

        # Stack the results along a new dimension (first dimension)
        u_star = torch.stack(diffs, dim=0)

        return u_star


    #@torch.jit.script
    def parabola(self, px, pt, ubar, mux, lmbda, l, f, k_indices : Dict[int, List[int]], dims : List[int], sigmap):

        musum = torch.stack([mux[..., k_indices[z]].sum(-1) for z in range(int(l))], dim=-1)
        dx = self.forward_differences(ubar, len(dims)-1)
        dt = torch.cat([ubar[...,1:]-ubar[...,:-1], torch.zeros_like(ubar[...,:1])], -1)*int(l)
        nodes = torch.arange(1, int(l)+1, dtype=f.dtype, device=f.device)/int(l)
        return project_parabola(px+sigmap*(dx+musum), pt+sigmap*dt, lmbda*(nodes-f.unsqueeze(-1)).square())
 

        
    #@torch.jit.script
    def bound(self, x, lmbda, k, l, f):
        return 0.25 * torch.sum(x * x, dim=0) - lmbda * torch.pow(k / l - f, 2)

    
                     
    #@torch.jit.script
    def energy(self, u, un):
        nrj = torch.sum(torch.abs(u - un))
        return nrj

                        
        
    #@torch.jit.script
    def l2projection(self, sx, mubarx, sigmas, nu):
        # Last axis contains S(S+1)/2 intervals.
        levels = int((math.sqrt(8.0*sx.size(-1)+1.0)-1.0)/2.0)
        return interval_ball(sx - sigmas*mubarx, levels*float(nu))

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
            torch.cat((torch.zeros(dims + [1], device = u.device), pt[...,:-1]), dim=-1))  / (1 / pt.shape[-1]) # / (1 / pt.shape[-1])
        
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
            diff = (before_cat - after_cat)  / (1 / p.shape[1])  # / (1 / p.shape[1])
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
#     g = DeviceMode(torch.device(dev))
#     g.__enter__()

    f = torch.randn(10, 10, 1, device = dev)
    repeats = torch.tensor(10, device = dev)
    level = torch.tensor(16, device = dev)
    lmbda = torch.tensor(1, device = dev)
    nu = torch.tensor(0.1, device = dev)
    tol = torch.tensor(1e-5, device = dev)
    
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
    
    

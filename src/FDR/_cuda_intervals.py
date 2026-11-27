"""Fused float32 CUDA update for the scaled solver's interval variables."""
import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


if triton is not None:
    @triton.jit
    def _update(prefix, lo, hi, slack, mu, mu_bar, new_slack, new_mu, new_mu_bar,
                n_sites, radius: tl.constexpr, sigma: tl.constexpr, tau: tl.constexpr,
                dims: tl.constexpr, levels: tl.constexpr, intervals: tl.constexpr, block: tl.constexpr):
        offset = tl.program_id(0).to(tl.int64)*block + tl.arange(0,block)
        mask = offset < n_sites*intervals
        site, interval = offset//intervals, offset%intervals
        first = tl.load(lo+interval, mask=mask, other=0)
        last = tl.load(hi+interval, mask=mask, other=0)
        squared = tl.full((block,), 0, tl.float32)
        for axis in tl.static_range(dims):
            index = (site+axis*n_sites)*intervals+interval
            value = tl.load(slack+index,mask=mask,other=0)-sigma*tl.load(mu_bar+index,mask=mask,other=0)
            squared += value*value
        factor = tl.where(squared>radius*radius, radius/tl.sqrt(tl.maximum(squared,1e-38)), 1.0)
        for axis in tl.static_range(dims):
            index = (site+axis*n_sites)*intervals+interval
            start = (site+axis*n_sites)*(levels+1)
            value = tl.load(slack+index,mask=mask,other=0)-sigma*tl.load(mu_bar+index,mask=mask,other=0)
            value *= factor
            total = tl.load(prefix+start+last,mask=mask,other=0)-tl.load(prefix+start+first,mask=mask,other=0)
            previous = tl.load(mu+index,mask=mask,other=0)
            delta = tau*(value-total)
            tl.store(new_slack+index,value,mask=mask)
            tl.store(new_mu+index,previous+delta,mask=mask)
            tl.store(new_mu_bar+index,previous+2*delta,mask=mask)


def update(px, slack, mu, mu_bar, starts, stops, radius, sigma, tau):
    if triton is None:
        raise RuntimeError("fused_intervals=True requires Triton")
    prefix = torch.cat([torch.zeros_like(px[..., :1]),px.cumsum(-1)],dim=-1).contiguous()
    dims, levels, intervals = px.size(0), px.size(-1), slack.size(-1)
    sites = px.numel()//(dims*levels)
    outputs = tuple(torch.empty_like(slack) for _ in range(3))
    with torch.cuda.device(px.device):
        _update[(triton.cdiv(sites*intervals,256),)](
            prefix, starts, stops, slack, mu, mu_bar, *outputs, sites,
            radius, sigma, tau, dims, levels, intervals, 256, enable_fp_fusion=False)
    return outputs

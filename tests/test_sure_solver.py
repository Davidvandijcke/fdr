import numpy as np
import pytest
import torch
from FDR import FDR
from FDR.SURE import SURE, SURE_objective_tune, make_probes, tune_func
from FDR.primaldual_multi_scaled_tune import make_solver
from FDR.utils import isosurface


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("scripted", [False, True])
def test_sure_uses_same_solver_on_both_sides(legacy, scripted):
    f = torch.tensor([[.1],[.2],[.6],[.3],[.9]])
    grid, eps, sigma = f.numpy(), .01, .006
    repeats, levels, tol = torch.tensor(60), torch.tensor(6), torch.tensor(0.)
    pair = torch.tensor([3.,.04])
    probes = make_probes(f, 2, 7)
    options = dict(legacy_jump_projection=legacy, scripted=scripted, step_safety=.7)
    computed = SURE_objective_tune(pair, tol, eps, f, repeats, levels, grid, sigma, probes, 2, options)
    model = make_solver(f.device, **options)
    def surface(data):
        v = model(data,repeats,levels,pair[0],pair[1],tol)[0]
        return isosurface(v.numpy(),6,grid)
    base = surface(f)
    divergences = []
    for probe in probes.unbind(dim=-1):
        delta = surface((f+eps*probe).clamp(0,1))-base
        divergences.append(np.vdot(probe.numpy().ravel(),delta)/(eps*base.size))
    expected = np.mean((grid.ravel()-base)**2)-sigma+2*sigma*np.mean(divergences)
    assert computed == pytest.approx(expected, abs=1.e-8)
    torch.testing.assert_close(make_probes(f, 2, 7), probes)


def test_local_search_and_ray_worker_on_cpu(monkeypatch):
    import FDR.main as main
    monkeypatch.setattr(main, "setDevice", lambda: torch.device("cpu"))
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    x = np.linspace(0,1,49)
    model = FDR(x*x+.2*(x>.5),x,level=5,iter=12,resolution=.125,CI=False,pick_nu="MS")
    result = SURE(model,maxiter=1,R=2,probe_seed=31,lmbda_min=2,lmbda_max=4,nu_min=.01,nu_max=.03)
    assert 2 <= result.x[0] <= 4 and .01 <= result.x[1] <= .03 and np.isfinite(result.fun)
    args = ({"lmbda":3.,"nu":.02},model.tol,.01,model.grid_y,model.iter,
            model.level,model.grid_y,.003,2)
    first = tune_func(*args,probe_seed=31)["score"]
    assert first == tune_func(*args,probe_seed=31)["score"]

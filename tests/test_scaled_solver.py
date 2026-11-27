import io
import itertools

import numpy as np
import pytest
import torch

from FDR import FDR, PrimalDual
from FDR.primaldual_multi_scaled import interval_ball, interval_sums, project_parabola, step_bound


def test_integrated_radius_and_small_spatial_norm():
    # Four lifted levels and seven spatial sites. One vector is interior,
    # one lies on the boundary, and one must be shortened to S * nu.
    field = torch.zeros(2,7,1,10)
    field[0,0,0,:3] = torch.tensor([.2,.4,.9])
    expected = field.clone()
    expected[0,0,0,2] = .4
    torch.testing.assert_close(interval_ball(field, 4*.1), expected)
    projected = PrimalDual().l2projection(field,torch.zeros_like(field),1.,.1)
    torch.testing.assert_close(projected,expected)
    torch.testing.assert_close(interval_ball(field*0, 0.), torch.zeros_like(field))
    x = torch.tensor([[1.e-5, 0.]])
    t = -torch.ones(2)
    p, pt = project_parabola(x, t, torch.zeros_like(t))
    torch.testing.assert_close(pt, p.square().sum(0)/4)
    assert (p-x+.5*p*(pt-t)).abs().max() < 2.e-11


def test_prefix_operator_and_step_bound():
    x = torch.tensor([.2, -.1, .4, -.6])
    starts, ends = torch.triu_indices(4, 4)
    explicit = torch.stack([x[i:j+1].sum() for i,j in zip(starts, ends)])
    torch.testing.assert_close(interval_sums(x, starts, ends+1), explicit)
    assert step_bound(25, 1, 32) > 1


@pytest.mark.parametrize("shape", [(9,1), (5,7,1), (4,3,5,1), (5,7,2)])
def test_eager_and_serialized_solver_agree(shape):
    torch.manual_seed(12)
    grid = torch.rand(shape)
    solver = PrimalDual(step_safety=.7)
    expected = solver(grid, 60, 7, 8., .03, 0.)
    archive = io.BytesIO()
    torch.jit.save(torch.jit.script(solver), archive)
    archive.seek(0)
    actual = torch.jit.load(archive)(grid, 60, 7, 8., .03, 0.)
    torch.testing.assert_close(actual[:3], expected[:3])
    assert actual[3] == expected[3] == 60
    assert torch.isfinite(actual[0]).all()
    assert (actual[0][...,0] == 1).all() and (actual[0][...,-1] == 0).all()


@pytest.mark.parametrize("scripted", [False, True])
def test_public_fit_returns_surface_and_jump_fields(scripted, monkeypatch):
    import FDR.main as main
    monkeypatch.setattr(main, "setDevice", lambda: torch.device("cpu"))
    x = np.linspace(0, 1, 101)
    y = .3*(x > .5) + x
    fit = FDR(y, x, scripted=scripted, CI=False, resolution=.125,
              pick_nu="MS", level=8, iter=40).run()
    assert {"u", "J", "jumps", "nrj", "eps", "it"} <= fit.keys()
    assert fit["u"].shape == fit["J"].shape == (8,)
    assert np.isfinite(fit["u"]).all()


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_against_small_independent_conic_problem(dimension):
    cp = pytest.importorskip("cvxpy")
    from scipy.sparse import csr_matrix
    side, levels = 3, 5
    shape = (side,)*dimension+(levels,)
    indices = np.arange(np.prod(shape)).reshape(shape)
    k = indices.size
    operators = []
    for axis, length in enumerate(shape):
        left = np.take(indices, np.arange(length-1), axis=axis).ravel()
        right = np.take(indices, np.arange(1,length), axis=axis).ravel()
        operators.append(csr_matrix((np.r_[-np.ones(len(left)), np.ones(len(left))]*length,
                                     (np.r_[left,left], np.r_[left,right])), shape=(k,k)))
    data = np.random.default_rng(42+dimension).uniform(.18,.82,shape[:-1]).astype("float32")
    lam, nu = 8., .055
    cost = lam*(np.arange(1,levels+1)/levels-data[...,None])**2
    px, pt = cp.Variable((dimension,k)), cp.Variable(k)
    constraints = [cp.sum(cp.square(px),axis=0)/4-cost.ravel() <= pt]
    for site in indices.reshape(-1,levels):
        for start, end in itertools.combinations(range(levels+1), 2):
            constraints.append(cp.norm(cp.sum(px[:,site[start:end]], axis=1)) <= levels*nu)
    coefficients = operators[-1].T@pt
    for axis in range(dimension):
        coefficients = coefficients+operators[axis].T@px[axis,:]
    free = indices[...,1:-1].ravel()
    objective = cp.sum(coefficients[indices[...,0].ravel()])+cp.sum(cp.minimum(coefficients[free],0))
    optimum = cp.Problem(cp.Maximize(objective/k),constraints).solve(solver="CLARABEL")
    u, _, _, it = PrimalDual(.5)(torch.from_numpy(data[...,None]),20000,levels,lam,nu,1.e-9)
    assert 70 < it < 20000
    u = u.numpy().reshape(shape).astype(float)
    assert np.diff(u,axis=-1).max() < 5.e-5
    # Construct a feasible monotone lift. Independently maximize its linear
    # objective over all feasible dual fields to evaluate its primal loss.
    feasible_u = np.minimum.accumulate(u,axis=-1)
    value = cp.Problem(cp.Maximize((coefficients@feasible_u.ravel())/k),constraints).solve(solver="CLARABEL")
    assert optimum-2.e-6 <= value <= optimum+2.e-5

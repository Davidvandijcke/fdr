import pytest
import torch

from FDR import PrimalDual
from FDR.primaldual_multi_scaled import interval_ball, interval_sums


def test_fused_solver_requires_cuda():
    with pytest.raises(ValueError, match="CUDA"):
        PrimalDual(fused_intervals=True)(torch.rand(6,1),10,8,2.,.01,0.)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware required")
@pytest.mark.parametrize("shape", [(13,1),(5,7,2),(4,3,5,1)])
def test_fused_update_and_full_fit(shape):
    pytest.importorskip("triton")
    from FDR._cuda_intervals import update
    torch.manual_seed(23)
    levels, dims = 9, len(shape)-1
    starts, ends = torch.triu_indices(levels,levels,device="cuda")
    stops = ends+1
    px = torch.randn((dims,*shape,levels),device="cuda")
    state = (dims,*shape,len(starts))
    slack, mu, bar = (torch.randn(state,device="cuda") for _ in range(3))
    for radius in (0.,.37):
        expected = interval_ball(slack-.7*bar,radius)
        delta = .013*(expected-interval_sums(px,starts,stops))
        result = update(px,slack,mu,bar,starts,stops,radius,.7,.013)
        torch.testing.assert_close(result,(expected,mu+delta,mu+2*delta),atol=1e-6,rtol=1e-6)
    f = torch.rand(shape,device="cuda")
    baseline = PrimalDual(.7)(f,256,levels,8.,.035,0.)
    fused = PrimalDual(.7,fused_intervals=True)(f,256,levels,8.,.035,0.)
    torch.testing.assert_close(fused[:3],baseline[:3],rtol=1e-4,atol=5e-6)
    assert baseline[3]==fused[3]

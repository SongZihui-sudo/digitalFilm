import torch
from kernel.my_model_port import trilinear_port


B: int = 16
C: int = 3
W: int = 224
H: int = 224
lut_dim: int = 33
    
def test_trilinear(x: torch.Tensor, lut: torch.Tensor, is_cuda: bool) -> torch.Tensor:
    if is_cuda:
        x = x.to("cuda:0")
        lut = lut.to("cuda:0")

    return trilinear_port.trilinearPort.apply(x, lut)  # type: ignore

if __name__ == "__main__":
    '''
        test_trilinear
    '''
    x: torch.Tensor = torch.randn([B, C, H, W])
    lut: torch.Tensor = torch.randn([B, lut_dim, lut_dim, lut_dim])
    
    output = test_trilinear(x, lut, False)
    assert len(output.shape) == 4, "[Wrong] trilinear Wrong output dimension!"
    print("[pass] trilinear")
    
    output = test_trilinear(x, lut, True)
    assert len(output.shape) == 4, "[Wrong] trilinear Wrong output dimension!"
    print("[pass] trilinear")
    
    print("[pass] Test ok!")
    
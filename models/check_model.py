import torch
from models.quad_vim_block import QuadMambaBlock

# Test with a typical sequence (e.g., 224x224 image, patch 16 -> 14x14 grid)
dim = 128
h, w = 14, 14
x = torch.randn(1, h*w, dim).to("cuda")

# Initialize your new block
model = QuadMambaBlock(dim).to("cuda")

# Forward pass
output = model(x, h, w)

print(f"Input:  {x.shape}")
print(f"Output: {output.shape}")
if x.shape == output.shape:
    print("✅ UNIT TEST PASSED: Quad-Scan dimensions are perfect.")

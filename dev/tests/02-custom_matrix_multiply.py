import torch

if not torch.mram.is_available():
    print("MRAM backend is not available in this build.")
    exit()

print("MRAM backend is available!")

device = torch.device("mram")

a = torch.randn(8, 8, device=device)
b = torch.randn(8, 16, device=device)

c = torch.ops.openreg.custom_matrix_multiply(a, b)

print("Result y:\n", c)
print(f"Device of y: {c.device}")

a_cpu = a.cpu()
b_cpu = b.cpu()

c_cpu = torch.matmul(a_cpu, b_cpu)

print("CPU result c_cpu:\n", c_cpu)

if torch.allclose(c.cpu(), c_cpu, atol=1e-1):
    print("MRAM result matches CPU result!")
else:
    print("MRAM result differs from CPU result!")
import torch

if not torch.mram.is_available():
    print("MRAM backend is not available in this build.")
    exit()

print("MRAM backend is available!")

device = torch.device("mram")

x = torch.tensor([[1., 2.], [3., 4.]], device=device)
y = x + 2
print("Result y:\n", y)
print(f"Device of y: {y.device}")

z = y.cpu()
print("Result z:\n", z)
print(f"Device of z: {z.device}")
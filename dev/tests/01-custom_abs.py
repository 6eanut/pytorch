import torch

if not torch.mram.is_available():
    print("MRAM backend is not available in this build.")
    exit()

print("MRAM backend is available!")

device = torch.device("mram")

x = torch.tensor([[-1., -2.], [-3., -4.]], device=device)
y = torch.ops.openreg.custom_abs(x)

print("Result y:\n", y)
print(f"Device of y: {y.device}")
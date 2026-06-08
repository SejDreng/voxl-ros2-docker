import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class XORModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2, 8)
        self.linear15 = nn.Linear(8, 2)
        self.linear2 = nn.Linear(2, 1)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear15(x))
        x = torch.sigmoid(self.linear2(x))
        return x


model = XORModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()

num_iterations = 10000

for step in range(num_iterations):
    # Random XOR input
    a = torch.randint(0, 2, (1,)).item()
    b = torch.randint(0, 2, (1,)).item()

    x = torch.tensor([[a, b]], dtype=torch.float32, device=device)
    y = torch.tensor([[a ^ b]], dtype=torch.float32, device=device)

    output = model(x)
    loss = criterion(output, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 1000 == 0:
        print(f"Step {step:5d} | Loss: {loss.item():.6f}")

# Save model
torch.save(model.state_dict(), "xor_model.pth")

# Test
print("\nTesting:")
test_inputs = torch.tensor(
    [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ],
    device=device,
)

with torch.no_grad():
    outputs = model(test_inputs)

for inp, out in zip(test_inputs.cpu(), outputs.cpu()):
    print(
        f"{inp.tolist()} -> {out.item():.4f} "
        f"(predicted {int(out.item() >= 0.5)})"
    )
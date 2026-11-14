
import torch
import torch.nn as nn

torch.manual_seed(42)

# -----------------------------------
# PART 1: GENERATE CARNOT TRAINING DATA
# -----------------------------------

N = 6000

P1 = 5 + 45 * torch.rand(N)
V1 = 1 + 4 * torch.rand(N)

TH = 400 + 400 * torch.rand(N)
TC = TH * (0.50 + 0.35 * torch.rand(N))

r = 1.1 + 0.9 * torch.rand(N)
gamma = 1.30 + 0.37 * torch.rand(N)

# -----------------------------------
# PART 2: CALCULATE THE FOUR PROCESSES
# -----------------------------------

k = (TH / TC) ** (1 / (gamma - 1))

# Process 1: Isothermal expansion
P2 = P1 / r
V2 = V1 * r

# Process 2: Adiabatic expansion
P3 = P2 / (k ** gamma)
V3 = V2 * k

# Process 3: Isothermal compression
P4 = P3 * r
V4 = V1 * k

# Process 4: Adiabatic compression
P5 = P1
V5 = V1

# -----------------------------------
# PART 3: INPUTS AND TARGET OUTPUTS
# -----------------------------------

X = torch.stack(
    [P1, V1, TH, TC, r, gamma], dim=1
)

Y = torch.stack(
    [P2, V2, P3, V3, P4, V4, P5, V5], dim=1
)

# Shuffle and split data
indices = torch.randperm(N)

train_idx = indices[:5000]
test_idx = indices[5000:]

X_train = X[train_idx]
Y_train = Y[train_idx]

X_test = X[test_idx]
Y_test = Y[test_idx]

# -----------------------------------
# PART 4: NORMALIZE THE DATA
# -----------------------------------

# Standardize inputs
X_mean = X_train.mean(dim=0)
X_std = X_train.std(dim=0).clamp_min(1e-8)

X_train = (X_train - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

# Use logarithms for positive pressure and volume outputs
Y_train = torch.log(Y_train)
Y_test = torch.log(Y_test)

Y_mean = Y_train.mean(dim=0)
Y_std = Y_train.std(dim=0).clamp_min(1e-8)

Y_train = (Y_train - Y_mean) / Y_std
Y_test = (Y_test - Y_mean) / Y_std

# -----------------------------------
# PART 5: BUILD THE NEURAL NETWORK
# -----------------------------------

class CarnotNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, 8)
        )

    def forward(self, x):
        return self.network(x)


model = CarnotNetwork()

# -----------------------------------
# PART 6: OPTIMIZER AND LOSS
# -----------------------------------

optimizer = torch.optim.Adam(
    model.parameters(), lr=0.001
)

loss_function = nn.MSELoss()


# -----------------------------------
# PART 7: TRAIN THE NETWORK
# -----------------------------------

import matplotlib.pyplot as plt

# Create the list BEFORE training
training_losses = []

for epoch in range(2001):

    optimizer.zero_grad()

    predictions = model(X_train)

    loss = loss_function(predictions, Y_train)

    loss.backward()

    optimizer.step()

    # Record loss from every epoch
    training_losses.append(loss.item())

    if epoch % 100 == 0:
        print(
            f"Epoch: {epoch}, Loss: {loss.item():.6f}"
        )

# -----------------------------------
# PART 7.1: PLOT LEARNING CURVE
# -----------------------------------

print("Recorded losses:", len(training_losses))

plt.plot(training_losses)

plt.xlabel("Epochs")
plt.ylabel("Training Loss")
plt.title("Carnot Network Learning Curve")

plt.show()


# -----------------------------------
# PART 8: TEST ON UNSEEN DATA
# -----------------------------------

model.eval()

with torch.no_grad():

    predicted_scaled = model(X_test)

    # Undo output normalization and logarithms
    predicted_log = predicted_scaled * Y_std + Y_mean
    predicted_values = torch.exp(predicted_log)

    actual_values = torch.exp(Y_test * Y_std + Y_mean)

    test_loss = loss_function(
        predicted_scaled, Y_test
    )

    print("\nTest loss:", test_loss.item())

# -----------------------------------
# PART 9: DISPLAY ONE TEST CYCLE
# -----------------------------------

sample = 0

names = [
    "Isothermal expansion",
    "Adiabatic expansion",
    "Isothermal compression",
    "Adiabatic compression",
]

print("\nPREDICTED CARNOT CYCLE")

for i, name in enumerate(names):

    p_index = 2 * i
    v_index = 2 * i + 1

    print("\n" + name)

    print(
        "Pressure:",
        predicted_values[sample, p_index].item()
    )

    print(
        "Volume:",
        predicted_values[sample, v_index].item()
    )

print("\nFINAL STATE")

print(
    "Predicted final pressure:",
    predicted_values[sample, 6].item()
)

print(
    "Actual initial pressure:",
    P1[test_idx][sample].item()
)

print(
    "Predicted final volume:",
    predicted_values[sample, 7].item()
)

print(
    "Actual initial volume:",
    V1[test_idx][sample].item()
)

print("\nACTUAL vs PREDICTED STATES")

for i, name in enumerate(names):

    p_index = 2 * i
    v_index = 2 * i + 1

    print("\n" + name)

    print("Actual Pressure:", torch.exp(
        Y_test[sample, p_index] * Y_std[p_index] + Y_mean[p_index]
    ).item())

    print("Predicted Pressure:",
          predicted_values[sample, p_index].item())

    print("Actual Volume:", torch.exp(
        Y_test[sample, v_index] * Y_std[v_index] + Y_mean[v_index]
    ).item())

    print("Predicted Volume:",
          predicted_values[sample, v_index].item())

with torch.no_grad():

    pressure_actual = actual_values[:, [0, 2, 4, 6]]
    pressure_predicted = predicted_values[:, [0, 2, 4, 6]]

    volume_actual = actual_values[:, [1, 3, 5, 7]]
    volume_predicted = predicted_values[:, [1, 3, 5, 7]]

    pressure_error = torch.abs(
        pressure_predicted - pressure_actual
    ) / pressure_actual * 100

    volume_error = torch.abs(
        volume_predicted - volume_actual
    ) / volume_actual * 100

    print("Average pressure error:",
          pressure_error.mean().item(), "%")

    print("Average volume error:",
          volume_error.mean().item(), "%")
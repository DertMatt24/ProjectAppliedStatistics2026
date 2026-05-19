import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleAttention(nn.Module):
    def __init__(self, window_size, hidden_dim=64):
        super().__init__()
        self.window_size = window_size
        self.hidden_dim = hidden_dim

        self.query = nn.Sequential(
            nn.Linear(window_size, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.key = nn.Sequential(
            nn.Linear(window_size, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.value = nn.Sequential(
            nn.Linear(window_size, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        scores = torch.bmm(q, k.transpose(1, 2)) / (self.hidden_dim ** 0.5)
        weights = F.softmax(scores, dim=-1)

        context = torch.bmm(weights, v)
        out = self.output(context)

        out = out.mean(dim=1)

        return out


class SimpleAttention(nn.Module):
    def __init__(self, window_size, hidden_dim=128):
        super().__init__()
        self.embed = nn.Linear(window_size, hidden_dim)
        self.context = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.attention_weights = None

    def forward(self, x):
        embedded = self.embed(x)
        scores = self.context(embedded)
        weights = F.softmax(scores, dim=1)
        self.attention_weights = weights.detach()
        weighted_sum = (embedded * weights).sum(dim=1)
        out = self.fc(weighted_sum)
        return out

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from loader.patients import PatientsCSVLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Filling the samples with the couples of patient and night id
samples = []
for patient_id in range(1, 40 + 1):
    for night_id in range(1, 2+1):
        samples.append((patient_id, night_id))

# Removing problematic samples (Totally not understandable)
samples.remove((8, 1))
samples.remove((14, 2))

df = PatientsCSVLoader.load_dataframe('C:/Users/picul/Videos/Applied/Dataset_Full/patients.csv')
# shape: (n_epochs, n_channels * 5)

# Extract attack counts for each sample
attack_counts = []
for (p_id, n_id) in samples:
    # Query the dataframe for this patient/night pair
    row = df[(df['user_id'] == p_id) & (df['night_id'] == n_id)]

    attacks = row.iloc[0]['NAp']

    # Handle empty/NaN values, the kaggle page says that missing entries corresond
    # to zero
    if pd.isna(attacks) or attacks == '' or attacks == 'NaN':
        attacks = 0
    else:
        attacks = int(attacks)  # Convert to int if it's a string

    attack_counts.append(attacks)

y = np.array(attack_counts)
X = np.load("C:/Users/picul/Videos/Applied/Dataset_Full/Embeddings/windowed_power_embeddings.npy")
shift = np.load("C:/Users/picul/Videos/Applied/Dataset_Full/Embeddings/power_ratios.npy")

# import umap

# reducer = umap.UMAP(n_components=5)
# X = reducer.fit_transform(np.array(X))

# np.save("C:/Users/picul/Videos/Applied/Dataset_Full/Embeddings/reduced_power_embeddings.npy", X)

from sklearn.preprocessing import StandardScaler


print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train).reshape(-1, 1)
y_test = torch.FloatTensor(y_test).reshape(-1, 1)

y_train_mean = y_train.mean()
y_train_std = y_train.std()

X_train_mean = X_train.mean()
X_train_std = X_train.std()

X_train = (X_train - X_train_mean) / X_train_std
X_test = (X_test - X_train_mean) / X_train_std

y_train = (y_train - y_train_mean) / y_train_std
y_test = (y_test - y_train_mean) / y_train_std

dataset = TensorDataset(X_train, y_train)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = nn.Sequential(
    nn.Linear(X.shape[1], 10),
    nn.LeakyReLU(),
    nn.Linear(10, 10),
    nn.LeakyReLU(),
    nn.Linear(10, 10),
    nn.LeakyReLU(),
    nn.Linear(10, 5),
    nn.LeakyReLU(),
    nn.Linear(5, 1),
)

window_size = 30
model = SimpleAttention(window_size, hidden_dim=25)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0006)

for epoch in range(100):
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        with torch.no_grad():
            train_loss = criterion(model(X_train), y_train)
            test_loss = criterion(model(X_test), y_test)
            print(f"Epoch {epoch + 1}: train {train_loss:.4f}, test {test_loss:.4f}")

with torch.no_grad():
    y_pred = model(X_test).numpy()
    saliency = model.attention_weights[0]

from sklearn.metrics import r2_score, mean_squared_error

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"R² = {r2:.4f}, MSE = {mse:.4f}")

print(y_pred.shape)
plt.plot(y_pred)
print(y_pred)
plt.plot(y_test)
plt.legend(["Predicted", "Actual data"])

plt.show()


plt.figure()
plt.plot(saliency.squeeze())
plt.xlabel("Window")
plt.ylabel("Attention Weight")
plt.show()
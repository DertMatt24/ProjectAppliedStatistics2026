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
for patient_id in range(1, 40):
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
X = np.load("C:/Users/picul/Videos/Applied/Dataset_Full/Embeddings/embeddings.npy")
# import umap

# reducer = umap.UMAP(n_components=5)
# X = reducer.fit_transform(np.array(X))

# np.save("C:/Users/picul/Videos/Applied/Dataset_Full/Embeddings/reduced_power_embeddings.npy", X)

from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

y_train = np.log1p(y_train)
y_test = np.log1p(y_test)

y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train).reshape(-1, 1)
y_test = torch.FloatTensor(y_test).reshape(-1, 1)


print(X_test)

dataset = TensorDataset(X_train, y_train)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

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

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001,  weight_decay=0.00001)

for epoch in range(300):
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            train_loss = criterion(model(X_train), y_train)
            test_loss = criterion(model(X_test), y_test)
            print(f"Epoch {epoch + 1}: train {train_loss:.4f}, test {test_loss:.4f}")

with torch.no_grad():
    y_pred = model(X_test).numpy()

from sklearn.metrics import r2_score, mean_squared_error

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"R² = {r2:.4f}, MSE = {mse:.4f}")

print(y_pred.shape)
plt.plot(y_pred)
plt.plot(y_test)
plt.legend(["Predicted", "Actual data"])

plt.show()
import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from loader.patients import PatientsCSVLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class SimpleAttention(nn.Module):
    def __init__(self, window_size, hidden_dim=128, n_patient_features=10):
        super().__init__()
        self.embed = nn.Linear(window_size, hidden_dim)
        self.context = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.patient_embed = nn.Sequential(
            nn.Linear(n_patient_features, hidden_dim),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.shared = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )

        self.classification_head = nn.Linear(hidden_dim, 1)
        self.regression_head = nn.Linear(hidden_dim, 1)

        self.attention_weights = None

    def forward(self, x, patient_features):
        embedded = self.embed(x)
        scores = self.context(embedded)
        weights = F.softmax(scores, dim=1)
        self.attention_weights = weights.detach()
        weighted_sum = (embedded * weights).sum(dim=1)

        patient_embed = self.patient_embed(patient_features)

        fused = torch.cat([weighted_sum, patient_embed], dim=1)
        shared_repr = self.shared(fused)

        classification = torch.sigmoid(self.classification_head(shared_repr))
        regression = self.regression_head(shared_repr)

        return classification, regression


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

# Extract physiological features
patient_features = []
y_regression = []
y_classification = []

for (p_id, n_id) in samples:
    row = df[(df['user_id'] == p_id) & (df['night_id'] == n_id)]

    attacks = row.iloc[0]['AHI']
    if pd.isna(attacks) or attacks == '' or attacks == 'NaN':
        attacks = 0
    else:
        attacks = float(attacks.replace(',', '.'))

    odi = row.iloc[0]['ODI']
    if pd.isna(odi) or odi == '' or odi == 'NaN':
        odi = 0

    age = row.iloc[0]['age']
    sex = 1 if row.iloc[0]['sex'] == 'M' else 0
    height = row.iloc[0]['height']
    weight = row.iloc[0]['weight']
    pulse = row.iloc[0]['pulse']
    bp = row.iloc[0]['BPsys/BPdia']

    patient_features.append([age, sex, height, weight, pulse, float(bp.split('/')[0]), odi])
    y_regression.append(attacks)
    y_classification.append(1 if attacks > 0 else 0)

patient_features = np.array(patient_features)
y_regression = np.array(y_regression)
y_classification = np.array(y_classification)

X = np.load("C:/Users/picul/Videos/Applied/Dataset_Full/Embeddings/windowed_power_embeddings.npy")
shift = np.load("C:/Users/picul/Videos/Applied/Dataset_Full/Embeddings/spectral_centroids.npy")

X_train, X_test, y_reg_train, y_reg_test, y_class_train, y_class_test, pf_train, pf_test, shift_train, shift_test = train_test_split(
    X, y_regression, y_classification, patient_features, shift, test_size=0.2, random_state=114
)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_reg_train = torch.FloatTensor(y_reg_train).reshape(-1, 1)
y_reg_test = torch.FloatTensor(y_reg_test).reshape(-1, 1)
y_class_train = torch.FloatTensor(y_class_train).reshape(-1, 1)
y_class_test = torch.FloatTensor(y_class_test).reshape(-1, 1)
pf_train = torch.FloatTensor(pf_train)
pf_test = torch.FloatTensor(pf_test)

X_train_mean = X_train.mean()
X_train_std = X_train.std()
X_train = (X_train - X_train_mean) / X_train_std
X_test = (X_test - X_train_mean) / X_train_std

y_reg_train_mean = y_reg_train.mean()
y_reg_train_std = y_reg_train.std()
y_reg_train = (y_reg_train - y_reg_train_mean) / y_reg_train_std
y_reg_test = (y_reg_test - y_reg_train_mean) / y_reg_train_std

pf_train_mean = pf_train.mean(dim=0)
pf_train_std = pf_train.std(dim=0)
pf_train = (pf_train - pf_train_mean) / pf_train_std
pf_test = (pf_test - pf_train_mean) / pf_train_std

dataset = TensorDataset(X_train, y_class_train, y_reg_train, pf_train)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

window_size = 30


model = SimpleAttention(window_size, n_patient_features=7, hidden_dim=16)


criterion_class = nn.BCELoss()
criterion_reg = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0002, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.25)

losses_train = []
losses_test = []
for epoch in range(500):
    epoch_loss = 0
    batch_count = 0

    for x_batch, y_class_batch, y_reg_batch, patient_features_batch in loader:
        optimizer.zero_grad()

        y_pred_class, y_pred_reg = model(x_batch, patient_features_batch)

        loss_class = criterion_class(y_pred_class, y_class_batch)

        sick_mask = y_class_batch == 1
        if sick_mask.sum() > 0:
            loss_reg = criterion_reg(y_pred_reg[sick_mask], y_reg_batch[sick_mask])
        else:
            loss_reg = 0

        loss = loss_class + 1.5 * loss_reg

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        batch_count += 1

    avg_train_loss = epoch_loss / batch_count
    losses_train.append(avg_train_loss)

    with torch.no_grad():
        y_pred_class_test, y_pred_reg_test = model(X_test, pf_test)

        test_loss_class = criterion_class(y_pred_class_test, y_class_test)

        sick_mask_test = y_class_test == 1
        if sick_mask_test.sum() > 0:
            test_loss_reg = criterion_reg(y_pred_reg_test[sick_mask_test], y_reg_test[sick_mask_test])
        else:
            test_loss_reg = 0

        avg_test_loss = test_loss_class + test_loss_reg
        losses_test.append(avg_test_loss.item())

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1}: train {avg_train_loss:.4f}, test {avg_test_loss:.4f}")

plt.plot(losses_train, label="Train")
plt.plot(losses_test, label="Test")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

with torch.no_grad():
    y_pred_class_test, y_pred_reg_test = model(X_test, pf_test)
    saliency = model.attention_weights.numpy()

y_class_test_np = y_class_test.numpy()
y_reg_test_np = y_reg_test.numpy()
y_pred_class_np = y_pred_class_test.numpy()
y_pred_reg_np = y_pred_reg_test.numpy()

y_pred_reg_denorm = y_pred_reg_np * y_reg_train_std.numpy() + y_reg_train_mean.numpy()
y_reg_test_denorm = y_reg_test_np * y_reg_train_std.numpy() + y_reg_train_mean.numpy()


for i in range(len(y_class_test_np)):
    is_sick = y_class_test_np[i, 0] > 0.5
    pred_sick = y_pred_class_np[i, 0] > 0.5


    shift_i = shift_test[i]
    print(f"Sample {i}: Actual={'Sick' if is_sick else 'Healthy'}, Predicted={'Sick' if pred_sick else 'Healthy'}")

    if is_sick:
        actual_attacksd = y_reg_test_denorm[i, 0]
        pred_attacksd = y_pred_reg_denorm[i, 0]
        actual_attacks = y_reg_test_np[i, 0]
        pred_attacks = y_pred_reg_np[i, 0]
        print(f"  Actual attacks: {actual_attacks:.4f}, Predicted attacks: {pred_attacks:.4f}")
        print(f"  Actual attacks: {actual_attacksd:.4f}, Predicted attacks: {pred_attacksd:.4f}")

        plt.figure(figsize=(10, 4))
        saliency_norm = (saliency[i].squeeze() - saliency[i].min()) / (saliency[i].max() - saliency[i].min())
        shift_norm = (shift_i - shift_i.min()) / (shift_i.max() - shift_i.min())

        plt.plot(saliency_norm)
        print(shift_norm.shape)
        plt.plot(shift_norm.T, linewidth=0.3)
        plt.title(f"Sample {i} Attention Weights (Actual: {actual_attacks:.2f}, Pred: {pred_attacks:.2f})")
        plt.xlabel("Window")
        plt.ylabel("Attention Weight")
        plt.show()
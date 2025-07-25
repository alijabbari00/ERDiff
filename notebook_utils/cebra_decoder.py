import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from torch import optim
from torch.utils.data import DataLoader, TensorDataset


class TwoLayerMLP(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=16, output_dim=2, dropout_rate=0.5):
        super(TwoLayerMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        return self.net(x)

    def fit(self, train_embedding, train_label, verbose=False):
        self.train()

        # Create a dataset and corresponding loader for batching
        train_dataset = TensorDataset(train_embedding.float(), train_label.float())
        batch_size = 1000  # Adjust batch size as necessary for your resources
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Initialize the criterion and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=0.1)

        num_epochs = 100
        for epoch in range(num_epochs):
            epoch_loss = 0.0

            # Iterate over each batch in the training loader
            for batch_data, batch_labels in train_loader:
                optimizer.zero_grad()
                outputs = self(batch_data.cuda())
                loss = criterion(outputs.cuda(), batch_labels.cuda())
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # Compute average loss per epoch
            avg_loss = epoch_loss / len(train_loader)

            # Optionally print out the loss for the current epoch
            if verbose and (epoch % 10 == 0 or epoch == num_epochs - 1):
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

        return self

    def score(self, embedding, label):
        self.eval()
        with torch.no_grad():
            predictions = self(embedding.float().cuda()).cpu().numpy()
            true_values = label.cpu().numpy()

        r2_scores = {}
        for i in range(len(predictions[0])):
            r2 = r2_score(true_values[:, i], predictions[:, i])
            r2_scores[f'Output_{i}'] = r2

        return sum(r2_scores.values()) / len(r2_scores)

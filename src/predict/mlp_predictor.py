import pickle
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, precision_score, recall_score
from sentence_transformers import SentenceTransformer


from src.predict.base_predictor import BasePredictor
from src.utils.scaler import BaseScaler


# Set random seeds for reproducibility
# SEED = int(os.environ["SEED"])
# SEED = 5226
# IS_CHAD = bool(os.environ["IS_CHAD"])
# CORPUS = os.environ["CORPUS"]
CORPUS = "books_read"

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.dropout = nn.Dropout(0.5)
        self.output = nn.Linear(1024, num_classes)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.tanh(self.fc3(x))
        x = self.dropout(x)
        x = self.output(x)
        return x


class TextDataset(Dataset):
    def __init__(self, vectors, labels):
        self.vectors = vectors
        self.labels = labels

    def __len__(self):
        return len(self.vectors)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.vectors[idx]),
            torch.LongTensor([self.labels[idx]])
        )



class MLPPredictor(BasePredictor):
    def __init__(self, random_state=0):
        super(MLPPredictor, self).__init__(random_state=random_state)

        torch.manual_seed(random_state)
        np.random.seed(random_state)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model, self.criterion, self.optimizer = None, None, None

    def train(self, X: pd.DataFrame, y: pd.Series, model_path: Path, scaler: Optional[BaseScaler] = None):
        if scaler is not None:
            X = scaler.fit_transform(X)

        # Prepare training labels
        train_labels = y.to_list()
        num_classes = len(set(train_labels))

        # Create datasets and dataloaders
        dataset = TextDataset(X.to_numpy(), train_labels)
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

        # Initialize model, loss, and optimizer
        self.model = MLP(X.shape[1], num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters())

        # Training loop with early stopping
        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0

        for epoch in range(100):
            train_loss, train_acc = self._train_model(self.model, train_loader, self.criterion, self.optimizer, self.device)
            val_loss, val_acc, f1, prec, recall = self._evaluate_model(self.model, val_loader, self.criterion, self.device)

            print(f'Epoch {epoch + 1}:')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            print(f'Val F1: {f1:.4f}, Val Precision: {prec:.4f}, Val Recall: {recall:.4f}')

            # Early stopping
            if val_loss < best_val_loss - 0.01:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'models/mlp_best.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break

        return self

    def _train_model(self, model, dataloader, criterion, optimizer, device):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.squeeze().to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(dataloader)
        train_acc = correct / total
        return train_loss, train_acc

    def inference(self, X: pd.DataFrame, model_path: Path, scaler: Optional[BaseScaler] = None):
        if scaler is not None:
            X = scaler.transform(X)
        self.model = MLP(X.shape[1], 5).to(self.device)
        self.model.load_state_dict(torch.load('models/mlp_best.pt'))
        features_tensor = torch.FloatTensor(X.to_numpy()).to(self.device)
        outputs = self.model(features_tensor)
        _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy().tolist()

    def _evaluate_model(self, model, dataloader, criterion, device):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.squeeze().to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        val_loss = running_loss / len(dataloader)
        val_acc = correct / total
        f1 = f1_score(all_labels, all_preds, average='weighted')
        prec = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')

        return val_loss, val_acc, f1, prec, recall


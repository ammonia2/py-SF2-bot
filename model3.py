# need to use pytorch model instead of keras Sequential

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam

class FeedForwardNet(nn.Module):
    def __init__(self, input_dim, hidden_layers, neurons, output_dim, dropout=0.3, l2_lambda=0.001):
        super(FeedForwardNet, self).__init__()
        self.l2_lambda = l2_lambda
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(input_dim, neurons))

        for _ in range(hidden_layers - 1):
            self.hidden_layers.append(nn.Linear(neurons, neurons))

        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(neurons, output_dim)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
            x = self.dropout(x)
        return torch.sigmoid(self.output(x))

class Model:
    def __init__(self, layers, neurons, inputDimension, outputDimension, trainingData, targetCols, learningRate):
        self.hiddenLayers = layers
        self.neurons = neurons
        self.inputDim = inputDimension
        self.outputDim = outputDimension
        self.learningRate = learningRate

        self.dataset = trainingData
        self.targetColumns = targetCols

        self.XTrain = None
        self.yTrain = None
        self.XTest = None
        self.yTest = None

        self.model = FeedForwardNet(
            input_dim=self.inputDim,
            hidden_layers=self.hiddenLayers,
            neurons=self.neurons,
            output_dim=self.outputDim
        )

    def train(self):
        X = self.dataset.drop(columns=self.targetColumns)
        y = self.dataset[self.targetColumns]

        self.XTrain, self.XTest, self.yTrain, self.yTest = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )

        X_train_tensor = torch.tensor(self.XTrain.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(self.yTrain.values, dtype=torch.float32)
        X_test_tensor = torch.tensor(self.XTest.values, dtype=torch.float32)
        y_test_tensor = torch.tensor(self.yTest.values, dtype=torch.float32)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        criterion = nn.BCELoss()
        optimizer = Adam(self.model.parameters(), lr=self.learningRate, weight_decay=0.001)

        self.model.train()
        for epoch in range(30):
            total_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/30, Loss: {total_loss:.4f}")

        torch.save(self.model.state_dict(), "SF2_model.pt")

    def predict(self):
        self.model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(self.XTest.values, dtype=torch.float32)
            predictions = self.model(X_test_tensor)
        return predictions.numpy()

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(self.XTest.values, dtype=torch.float32)
            y_test_tensor = torch.tensor(self.yTest.values, dtype=torch.float32)
            predictions = self.model(X_test_tensor)
            predicted_classes = (predictions > 0.5).float()
            accuracy = (predicted_classes == y_test_tensor).float().mean()
        return accuracy.item()

def concatenateData():
    data = pd.DataFrame()
    dataDirectory = 'training_data_p2/'

    dtype_dict = {
        "frame": int, "p1Id": int, "p1Health": int, "p1PosX": int, "p1PosY": int,
        "p1Jump": int, "p1Crouch": int, "p1InMove": int, "p1MoveId": int,
        "p1Up": int, "p1Down": int, "p1Left": int, "p1Right": int, "p1Select": int,
        "p1Start": int, "p1Y": int, "p1B": int, "p1X": int, "p1A": int, "p1L": int, "p1R": int,
        "p2Id": int, "p2Health": int, "p2PosX": int, "p2PosY": int,
        "p2Jump": int, "p2Crouch": int, "p2InMove": int, "p2MoveId": int,
        "p2Up": int, "p2Down": int, "p2Left": int, "p2Right": int, "p2Select": int,
        "p2Start": int, "p2Y": int, "p2B": int, "p2X": int, "p2A": int, "p2L": int, "p2R": int,
        "timer": int, "roundStarted": int, "roundOver": int, "fightResult": str
    }

    column_names = list(dtype_dict.keys())
    for filename in os.listdir(dataDirectory):
        filepath = os.path.join(dataDirectory, filename)
        df = pd.read_csv(filepath, low_memory=False, dtype=dtype_dict, skiprows=1, names=column_names)
        data = pd.concat([data, df], ignore_index=True)

    data = data[data['roundStarted'] != False]
    data = data.drop(['frame', 'roundStarted', 'fightResult', 'roundOver'], axis=1)

    data['xDist'] = data['p1PosX'] - data['p2PosX']
    data['yDist'] = data['p1PosY'] - data['p2PosY']
    data = data.drop(columns=['p1PosX', 'p1PosY', 'p2PosX', 'p2PosY'], axis=1)
    return data

def normaliseFeatures(data):
    scaler = StandardScaler()
    featuresToNormalise = ['p1Health', 'p2Health', 'timer', 'xDist', 'yDist']
    data[featuresToNormalise] = scaler.fit_transform(data[featuresToNormalise])
    joblib.dump(scaler, 'scaler.joblib')
    return data

def oneHotEncoding(data):
    data['p1Id'] = pd.to_numeric(data['p1Id'], errors='coerce')
    data['p2Id'] = pd.to_numeric(data['p2Id'], errors='coerce')

    for i in range(12):
        data[f'AI_is_{i}'] = (data['p1Id'] == i)
        data[f'CPU_is_{i}'] = (data['p2Id'] == i)

    data = data.drop(columns=['p1Id', 'p2Id', 'p1MoveId', 'p2MoveId', 'p1Select', 'p2Select', 'p1Start', 'p2Start'], axis=1)
    return data

if __name__ == '__main__':
    data = concatenateData()
    data = normaliseFeatures(data)
    data = oneHotEncoding(data)

    targetColumns = [
        'p1Up', 'p1Down', 'p1Left', 'p1Right', 'p1Y', 'p1B', 'p1X', 'p1A', 'p1L', 'p1R'
    ]
    temp_df = data.drop(columns=targetColumns, axis=1)
    featureNames = temp_df.columns.to_list()
    joblib.dump(featureNames, 'feature_names.joblib')

    model = Model(
        layers=3,
        neurons=128,
        inputDimension=len(temp_df.columns.to_list()),
        outputDimension=len(targetColumns),
        trainingData=data,
        targetCols=targetColumns,
        learningRate=0.001
    )

    model.train()

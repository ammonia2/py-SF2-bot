# when done training, have to try mirroring data to also make the model work when playing as Player 2 !!!

# 1. merge data from all the different files into one file
# 2. normalise features like timer, health, positions (consider taking difference of positionX & posY)
# 3. one hot encode player Ids so the model can differentiate who's it playing as well as the opponent character
# 4. implement a multi-layer perceptron model
# 5. input: flattened game state features
# 6. 2 to 3 hidden layers - experiment with neurons 128, 64, 32 etc
# 7. sigmoid activation function for each of player 1's actions
# 8. Loss function: sum: binary cross-entropy
# 9. back propagation after each forward pass and loss calculation
# 10. split data into training & (testing ?)
# 11. shuffle training data & train for certain epochs
# 12. save neural network and use it to play against CPU

import pandas as pd
import numpy as np
import os

import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class Model:
    def __init__(self, layers, neurons, inputDimension, outputDimension, trainingData, targetCols):
        # model
        self.hiddenLayers = layers
        self.neurons = neurons
        self.inputDim = inputDimension
        self.outputDim = outputDimension
        self.model = None

        # data
        self.dataset = trainingData
        self.trainingData = None
        self.predictionData = None

        self.XTrain = None
        self.yTrain = None
        self.XTest = None
        self.yTest = None
        
        self.targetColumns = targetCols
        self.initialiseModel() # iniitialise Model
        
    def initialiseModel(self):
        """initialise Multi Layer Perceptron model"""
        
        self.model = nn.Sequential()
        self.model.add_module('input', nn.Linear(self.inputDim, self.neurons))
        self.model.add_module('relu1', nn.ReLU())

        for i in range(self.hiddenLayers - 1):
            self.model.add_module(f'hidden{i+1}', nn.Linear(self.neurons, self.neurons))
            self.model.add_module(f'relu{i+2}', nn.ReLU())
        
        self.model.add_module('output', nn.Linear(self.neurons, self.outputDim))
        self.model.add_module('sigmoid', nn.Sigmoid())

    def train(self):
        """training after filtering the dataset"""
        
        self.initialiseModel()

        # transforming data before
        X = self.dataset.drop(columns=self.targetColumns)
        y = self.dataset[self.targetColumns]

        self.XTrain, self.XTest, self.yTrain, self.yTest = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )
        
        # Convert to PyTorch tensors - ensuring all data is numeric
        X_train_tensor = torch.FloatTensor(self.XTrain.astype(float).values)
        y_train_tensor = torch.FloatTensor(self.yTrain.astype(float).values)
        X_test_tensor = torch.FloatTensor(self.XTest.astype(float).values)
        y_test_tensor = torch.FloatTensor(self.yTest.astype(float).values)
        
        # Create DataLoader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        # Define optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        
        # Training loop
        epochs = 30
        history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
        
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                predicted = (outputs > 0.5).float()
                total += targets.size(0) * targets.size(1)
                correct += (predicted == targets).sum().item()
            
            # Calculate validation metrics
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_test_tensor)
                val_loss = criterion(val_outputs, y_test_tensor).item()
                val_predicted = (val_outputs > 0.5).float()
                val_total = y_test_tensor.size(0) * y_test_tensor.size(1)
                val_correct = (val_predicted == y_test_tensor).sum().item()
            
            # Record metrics
            epoch_loss = running_loss / len(train_loader)
            epoch_acc = correct / total
            val_acc = val_correct / val_total
            
            history['loss'].append(epoch_loss)
            history['accuracy'].append(epoch_acc)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)
            
            print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        return history

    def predict(self):
        self.model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(self.XTest.values)
            predictions = self.model(X_test_tensor).numpy()
        return predictions

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(self.XTest.values)
            y_test_tensor = torch.FloatTensor(self.yTest.values)
            
            outputs = self.model(X_test_tensor)
            criterion = nn.BCELoss()
            loss = criterion(outputs, y_test_tensor).item()
            
            predicted = (outputs > 0.5).float()
            total = y_test_tensor.size(0) * y_test_tensor.size(1)
            correct = (predicted == y_test_tensor).sum().item()
            accuracy = correct / total
        
        return [loss, accuracy]

def concatenateData():
    """Merge all character data in a single DataFrame & filter required cols"""
    data = pd.DataFrame()
    dataDirectory = 'training_data_p2/'

    # Defining expected dtypes for each column
    dtype_dict = {
        "frame": int,
        "p1Id": int,
        "p1Health": int,
        "p1PosX": int,
        "p1PosY": int,
        "p1Jump": int,
        "p1Crouch": int,
        "p1InMove": int,
        "p1MoveId": int,
        "p1Up": int,
        "p1Down": int,
        "p1Left": int,
        "p1Right": int,
        "p1Select": int,
        "p1Start": int,
        "p1Y": int,
        "p1B": int,
        "p1X": int,
        "p1A": int,
        "p1L": int,
        "p1R": int,
        "p2Id": int,
        "p2Health": int,
        "p2PosX": int,
        "p2PosY": int,
        "p2Jump": int,
        "p2Crouch": int,
        "p2InMove": int,
        "p2MoveId": int,
        "p2Up": int,
        "p2Down": int,
        "p2Left": int,
        "p2Right": int,
        "p2Select": int,
        "p2Start": int,
        "p2Y": int,
        "p2B": int,
        "p2X": int,
        "p2A": int,
        "p2L": int,
        "p2R": int,
        "timer": int,
        "roundStarted": int,
        "roundOver": int,
        "fightResult": str
    }

    column_names = [
        "frame", "p1Id", "p1Health", "p1PosX", "p1PosY", "p1Jump", "p1Crouch", "p1InMove", "p1MoveId",
        "p1Up", "p1Down", "p1Left", "p1Right", "p1Select", "p1Start", "p1Y", "p1B", "p1X", "p1A", "p1L", "p1R",
        "p2Id", "p2Health", "p2PosX", "p2PosY", "p2Jump", "p2Crouch", "p2InMove", "p2MoveId",
        "p2Up", "p2Down", "p2Left", "p2Right", "p2Select", "p2Start", "p2Y", "p2B", "p2X", "p2A", "p2L", "p2R",
        "timer", "roundStarted", "roundOver", "fightResult"
    ]

    for filename in os.listdir(dataDirectory):
        filepath = os.path.join(dataDirectory, filename)
        df = pd.read_csv(filepath, low_memory=False, dtype=dtype_dict, skiprows=1, names=column_names)
        data = pd.concat([data, df], ignore_index=True)

    data = data[data['roundStarted'] != False]

    # Filter out frames where all player 1 movement and attack keys are 0
    action_keys = ['p1Up', 'p1Down', 'p1Left', 'p1Right', 'p1Y', 'p1B', 'p1X', 'p1A', 'p1L', 'p1R']
    data = data[data[action_keys].sum(axis=1) > 0]


    data = data.dropna(axis=1, how='all')
    # dropping cuz not needed
    data = data.drop(['frame', 'roundStarted', 'fightResult', 'roundOver'], axis=1)

    data['xDist'] = data['p1PosX'] - data['p2PosX']
    data['yDist'] = data['p1PosY'] - data['p2PosY']
    data = data.drop(columns=['p1PosX', 'p1PosY', 'p2PosX', 'p2PosY'], axis=1)
    # print(data.dtypes)
    # print(data.sample(5))
    return data

def normaliseFeatures(data):
    """Normalise yDist, xDist, health & timer values"""
    # using StandardScalar here because it has advantages > standard max val normalisation
    scaler = StandardScaler()
    featuresToNormalise = ['p1Health', 'p2Health', 'timer', 'xDist', 'yDist']
    data[featuresToNormalise] = scaler.fit_transform(data[featuresToNormalise])

    # saving the scaler for later use
    joblib.dump(scaler, 'scaler.joblib')

    return data

def oneHotEncoding(data):
    """one hot encode p1Id and p2Id"""
    data['p1Id'] = pd.to_numeric(data['p1Id'], errors='coerce')
    data['p2Id'] = pd.to_numeric(data['p2Id'], errors='coerce')
    
    # Create one-hot encoded columns
    for i in range(12):
        data[f'AI_is_{i}'] = (data['p1Id'] == i)
        data[f'CPU_is_{i}'] = (data['p2Id'] == i)
    
    # now p1Id and p2Id no longer useful
    data = data.drop(columns=['p1Id', 'p2Id', 'p1MoveId', 'p2MoveId', 'p1Select', 'p2Select', 'p1Start', 'p2Start'], axis=1)

    return data

if __name__ == '__main__':
    data = concatenateData()
    data = normaliseFeatures(data)
    data = oneHotEncoding(data)
    
    targetColumns = [
        'p1Up', 'p1Down', 'p1Left', 'p1Right', 'p1Y', 'p1B', 'p1X', 
        'p1A', 'p1L', 'p1R'
    ]
    temp_df = data.drop(columns=targetColumns, axis=1)
    featureNames = temp_df.columns.to_list()
    joblib.dump(featureNames, 'feature_names.joblib')
    model = Model(
        layers=3,
        neurons=64,
        inputDimension=len(temp_df.columns.to_list()),  # total features after preprocessing
        outputDimension=len(targetColumns), # number of actions to predict
        trainingData=data,
        targetCols= targetColumns
    )
   
    history = model.train()
    torch.save(model.model.state_dict(), 'SF2_model.pth')
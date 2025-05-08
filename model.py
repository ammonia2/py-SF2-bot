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

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

class Model:
    def __init__(self, layers, neurons, inputDimension, outputDimension, trainingData):
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
        
        self.targetColumns = [
            'p1Up', 'p1Down', 'p1Left', 'p1Right',
            'p1Jump', 'p1Crouch', 'p1Y', 'p1B', 'p1X', 
            'p1A', 'p1L', 'p1R'
        ]
        self.initialiseModel() # iniitialise Model
        
    def initialiseModel(self):
        """initialise Multi Layer Perceptron model"""
        
        self.model = Sequential()
        self.model.add(Dense(self.neurons, activation='relu', input_dim=self.inputDim))

        for i in range(self.hiddenLayers - 1):
            self.model.add(Dense(self.neurons, activation='relu'))
        
        self.model.add(Dense(self.outputDim, activation = 'sigmoid'))

        self.model.compile(
            optimizer = Adam(learning_rate = 0.001),
            loss = 'binary_crossentropy',
            metrics = ['accuracy']
        )
        

    def train(self):
        """training after filtering the dataset"""
        
        self.initialiseModel()

        # transforming data before
        X = self.dataset.drop(columns=self.targetColumns)
        y = self.dataset[self.targetColumns]

        self.XTrain, self.XTest, self.yTrain, self.yTest = train_test_split(
            X, y, test_size=0.2,random_state=42,shuffle=True
        )
        
        history = self.model.fit(
            self.XTrain, self.yTrain,
            validation_data = (self.XTest, self.yTest),
            epochs=6,
            batch_size = 64,
            shuffle = True,
            verbose=1
        )
        
        return history

    def predict(self):
        return self.model.predict(self.XTest)

    def evaluate(self):
        return self.model.evaluate(self.XTest, self.yTest, verbose=0)

def concatenateData():
    """Merge all character data in a single DataFrame & filter required cols"""
    data = pd.DataFrame()
    dataDirectory = 'training_data/'

    for filename in os.listdir(dataDirectory):
        # print(filename)
        filepath = os.path.join(dataDirectory, filename)
        df = pd.read_csv(filepath, low_memory= False)
        data = pd.concat([data, df], ignore_index=True)

    data = data[data['roundStarted'] != False]

    # dropping cuz not needed
    data = data.drop(['frame', 'roundStarted', 'fightResult', 'roundOver'], axis=1)

    numericCols = ['p1PosX', 'p1PosY', 'p1Health', 'p1X', 'p1Y',
        'p2PosX', 'p2PosY', 'p2Health', 'p2X', 'p2Y',
        'timer']
    boolCols = ['p1Jump', 'p1Crouch', 'p1InMove', 'p1Up', 'p1Down', 'p1Left', 'p1Right', 
                'p1B', 'p1A', 'p1L', 'p1R', 'p2Jump', 'p2Crouch', 'p2InMove', 'p2Up', 
                'p2Down', 'p2Left', 'p2Right', 'p2B', 'p2A', 'p2L', 'p2R']
    
    for col in numericCols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    for col in boolCols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')


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
    # print(data.sample(5))
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
    # print(data.columns.to_list())
    data.sample(50).to_csv('pre_data.csv', index=False)

    return data

if __name__ == '__main__':
    data = concatenateData()
    data = normaliseFeatures(data)
    data = oneHotEncoding(data)
    
    targetColumns = [
        'p1Up', 'p1Down', 'p1Left', 'p1Right',
        'p1Jump', 'p1Crouch', 'p1Y', 'p1B', 'p1X', 
        'p1A', 'p1L', 'p1R'
    ]
    temp_df = data.drop(columns=targetColumns, axis=1)
    model = Model(
        layers=3,
        neurons=32,
        inputDimension=len(temp_df.columns.to_list()),  # total features after preprocessing
        outputDimension=12, # number of actions to predict
        trainingData=data
    )
   
    history = model.train()
    model.model.save('SF2_model.keras')
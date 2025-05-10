from command import Command
import numpy as np
from buttons import Buttons
import torch
import joblib
import os
import pandas as pd
import numpy as np

modelPath = 'SF2_model.pth'
scalerPath = 'scaler.joblib'
featureNamesPath = 'feature_names.joblib'

targetCols = [
    'p1Up', 'p1Down', 'p1Left', 'p1Right',
    'p1Y', 'p1B', 'p1X', 'p1A', 'p1L', 'p1R'
]

characterIds = list(range(12))

class Bot:
    def __init__(self):
        # Load the model architecture first
        input_dim = len(joblib.load(featureNamesPath))
        self.model = torch.nn.Sequential()
        self.model.add_module('input', torch.nn.Linear(input_dim, 64))
        self.model.add_module('relu1', torch.nn.ReLU())
        self.model.add_module('hidden1', torch.nn.Linear(64, 64))
        self.model.add_module('relu2', torch.nn.ReLU())
        self.model.add_module('hidden2', torch.nn.Linear(64, 64))
        self.model.add_module('relu3', torch.nn.ReLU())
        self.model.add_module('output', torch.nn.Linear(64, len(targetCols)))
        self.model.add_module('sigmoid', torch.nn.Sigmoid())
        # Load the trained weights
        self.model.load_state_dict(torch.load(modelPath))
        self.model.eval()  # Set to evaluation mode
        
        self.scaler = joblib.load(scalerPath)
        self.featureNames = joblib.load(featureNamesPath)
        print("Model and preprocessing tools loaded successfully.")

        self.buttonThreshold = 0.005 # Threshold to convert probabilities to binary button presses

    def preprocessGameState(self, rawGameState):
        """Preprocesses the raw game state dictionary into the model's input format."""
        if self.scaler is None or self.featureNames is None:
             print("Preprocessing tools not loaded. Cannot preprocess game state.")
             return None

        # DF to hold the game state as required by model
        stateDict = {
             "p1Id": rawGameState.player1.player_id,
             "p1Health": rawGameState.player1.health,
             "p1PosX": rawGameState.player1.x_coord,
             "p1PosY": rawGameState.player1.y_coord,
             "p1Jump": rawGameState.player1.is_jumping,
             "p1Crouch": rawGameState.player1.is_crouching,
             "p1InMove": rawGameState.player1.is_player_in_move,
             "p1MoveId": rawGameState.player1.move_id,
             # Player 1 buttons
             "p1Up": rawGameState.player1.player_buttons.up,
             "p1Down": rawGameState.player1.player_buttons.down,
             "p1Left": rawGameState.player1.player_buttons.left,
             "p1Right": rawGameState.player1.player_buttons.right,
             "p1Select": rawGameState.player1.player_buttons.select,
             "p1Start": rawGameState.player1.player_buttons.start,
             "p1Y": rawGameState.player1.player_buttons.Y,
             "p1B": rawGameState.player1.player_buttons.B,
             "p1X": rawGameState.player1.player_buttons.X,
             "p1A": rawGameState.player1.player_buttons.A,
             "p1L": rawGameState.player1.player_buttons.L,
             "p1R": rawGameState.player1.player_buttons.R,

             "p2Id": rawGameState.player2.player_id,
             "p2Health": rawGameState.player2.health,
             "p2PosX": rawGameState.player2.x_coord,
             "p2PosY": rawGameState.player2.y_coord,
             "p2Jump": rawGameState.player2.is_jumping,
             "p2Crouch": rawGameState.player2.is_crouching,
             "p2InMove": rawGameState.player2.is_player_in_move,
             "p2MoveId": rawGameState.player2.move_id,
             # Player 2 buttons (inputs)
             "p2Up": rawGameState.player2.player_buttons.up,
             "p2Down": rawGameState.player2.player_buttons.down,
             "p2Left": rawGameState.player2.player_buttons.left,
             "p2Right": rawGameState.player2.player_buttons.right,
             "p2Select": rawGameState.player2.player_buttons.select,
             "p2Start": rawGameState.player2.player_buttons.start,
             "p2Y": rawGameState.player2.player_buttons.Y,
             "p2B": rawGameState.player2.player_buttons.B,
             "p2X": rawGameState.player2.player_buttons.X,
             "p2A": rawGameState.player2.player_buttons.A,
             "p2L": rawGameState.player2.player_buttons.L,
             "p2R": rawGameState.player2.player_buttons.R,

             "timer": rawGameState.timer,
             "roundStarted": rawGameState.has_round_started,
             "roundOver": rawGameState.is_round_over,
             "fightResult": rawGameState.fight_result
         }

        # Convert to DataFrame
        stateDf = pd.DataFrame([stateDict])

        # Applying the filtering
        # stateDf = stateDf[stateDf['roundStarted'] != False]

        # Calculate relative positions (Matching concatenateData)
        stateDf['xDist'] = stateDf['p1PosX'] - stateDf['p2PosX']
        stateDf['yDist'] = stateDf['p1PosY'] - stateDf['p2PosY']
        stateDf = stateDf.drop(columns=['p1PosX', 'p1PosY', 'p2PosX', 'p2PosY'], axis=1)

        # Handle boolean-like columns conversion (Matching concatenateData logic)
        # Need to list ALL bool-like columns that were converted during training
        boolLikeCols = [
            'p1Jump', 'p1Crouch', 'p1InMove', 'p1Up', 'p1Down', 'p1Left', 'p1Right',
            'p1B', 'p1A', 'p1L', 'p1R', 'p1Select', 'p1Start', # P1 buttons are targets
            'p2Jump', 'p2Crouch', 'p2InMove', 'p2Up', 'p2Down', 'p2Left', 'p2Right',
            'p2B', 'p2A', 'p2L', 'p2R', 'p2Select', 'p2Start', # P2 buttons are features
            'roundStarted', 'roundOver' # These were dropped as features, but might be in initial data as bools
        ]
        mapDict = { True: 1, False: 0, 'True': 1, 'False': 0, 'true': 1, 'false': 0,
                    '1': 1, '0': 0, 1: 1, 0: 0, 1.0: 1, 0.0: 0 }
        for col in boolLikeCols:
            if col in stateDf.columns:
                 stateDf[col] = stateDf[col].map(mapDict).fillna(0).astype(int)

        # normalization using the loaded scaler
        featuresToNormalise = ['p1Health', 'p2Health', 'timer', 'xDist', 'yDist']
        colsToTransform = stateDf.columns.intersection(featuresToNormalise)
        if not colsToTransform.empty:
             stateDf[colsToTransform] = self.scaler.transform(stateDf[colsToTransform])

        # one-hot encoding for p1Id and p2Id
        for charId in characterIds:
             stateDf[f'AI_is_{charId}'] = (stateDf['p1Id'] == charId).astype(int)
             stateDf[f'CPU_is_{charId}'] = (stateDf['p2Id'] == charId).astype(int)

        colsToDrop = ['p1Id', 'p2Id', 'fightResult', 'p1MoveId', 'p2MoveId',
                      'roundStarted', 'roundOver', 'p1Select', 'p1Start', 
                      'p2Select', 'p2Start']

        # Only drop columns that actually exist in the current DataFrame row
        stateDf = stateDf.drop(columns=colsToDrop, axis=1)

        # Reindex the DataFrame row to match the exact order from self.featureNames
        # This is CRITICAL
        XInputDf = stateDf.reindex(columns=self.featureNames, fill_value=0)

        XInput = XInputDf.values # convert DF to numpy array
        XInput = XInput.astype(np.float32)

        return XInput


    def fight(self, currentGameState, player):
        """Predicts and returns the command for the specified player using the trained model."""

        myCommand = Command() # Create a new command object for this frame

        if player == "1":
            if self.model is None:
                 print("Model not loaded. Returning default command.")
                 myCommand.player_buttons = Buttons()
                 return myCommand

            # Preprocess the current game state
            XInput = self.preprocessGameState(currentGameState)

            if XInput is None: # Handle case where preprocessing failed
                 print("Preprocessing failed. Returning default command.")
                 myCommand.player_buttons = Buttons()
                 return myCommand

            # Get predictions from the model using PyTorch
            # print(f'Input: {XInput}')
            with torch.no_grad():  # Disable gradient calculation for inference
                input_tensor = torch.FloatTensor(XInput)
                outputProbs = self.model(input_tensor).numpy()
            
            # print(f'Output: {outputProbs}')
            outputProbs = outputProbs[0]

            # Converting probabilities to binary decisions using the threshold
            buttonDecisions = outputProbs >= self.buttonThreshold

            # Mapping the binary decisions back to the Command object's player_buttons
            myCommand.player_buttons.up = bool(buttonDecisions[targetCols.index('p1Up')])
            myCommand.player_buttons.down = bool(buttonDecisions[targetCols.index('p1Down')])
            myCommand.player_buttons.left = bool(buttonDecisions[targetCols.index('p1Left')])
            myCommand.player_buttons.right = bool(buttonDecisions[targetCols.index('p1Right')])
            myCommand.player_buttons.Y = bool(buttonDecisions[targetCols.index('p1Y')])
            myCommand.player_buttons.B = bool(buttonDecisions[targetCols.index('p1B')])
            myCommand.player_buttons.X = bool(buttonDecisions[targetCols.index('p1X')])
            myCommand.player_buttons.A = bool(buttonDecisions[targetCols.index('p1A')])
            myCommand.player_buttons.L = bool(buttonDecisions[targetCols.index('p1L')])
            myCommand.player_buttons.R = bool(buttonDecisions[targetCols.index('p1R')])


        elif player == "2":
            # --- Logic for controlling Player 2 ---
            print("Bot is currently configured for Player 1. Returning default command for Player 2.")
            myCommand.player2_buttons = Buttons() # Default 'do nothing' for P2

        return myCommand
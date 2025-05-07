import socket
import json
import sys
import keyboard  # External package: pip install keyboard
import pandas as pd
from game_state import GameState
from bot import Bot
from command import Command

# Global data collector for frame information
frameData = []

# Character names mapped to their IDs (0-11)
CHARACTER_NAMES = {
    0: "Ryu",
    1: "Ken",
    2: "ChunLi",
    3: "Guile",
    4: "Cammy",
    5: "Sagat",
    6: "Dhalsim",
    7: "Zangief",
    8: "Honda",
    9: "Blanka",
    10: "Vega",
    11: "Bison"
}

def logGameData(gameState, frameNum):
    """Log current game state to our data collector"""
    global frameData
    
    frameData.append({
        "frame": frameNum,

        # Player 1
        "p1Id": gameState.player1.player_id,
        "p1Health": gameState.player1.health,
        "p1PosX": gameState.player1.x_coord,
        "p1PosY": gameState.player1.y_coord,
        "p1Jump": gameState.player1.is_jumping,
        "p1Crouch": gameState.player1.is_crouching,
        "p1InMove": gameState.player1.is_player_in_move,
        "p1MoveId": gameState.player1.move_id,

        # Player 1 buttons
        "p1Up": gameState.player1.player_buttons.up,
        "p1Down": gameState.player1.player_buttons.down,
        "p1Left": gameState.player1.player_buttons.left,
        "p1Right": gameState.player1.player_buttons.right,
        "p1Select": gameState.player1.player_buttons.select,
        "p1Start": gameState.player1.player_buttons.start,
        "p1Y": gameState.player1.player_buttons.Y,
        "p1B": gameState.player1.player_buttons.B,
        "p1X": gameState.player1.player_buttons.X,
        "p1A": gameState.player1.player_buttons.A,
        "p1L": gameState.player1.player_buttons.L,
        "p1R": gameState.player1.player_buttons.R,

        # Player 2
        "p2Id": gameState.player2.player_id,
        "p2Health": gameState.player2.health,
        "p2PosX": gameState.player2.x_coord,
        "p2PosY": gameState.player2.y_coord,
        "p2Jump": gameState.player2.is_jumping,
        "p2Crouch": gameState.player2.is_crouching,
        "p2InMove": gameState.player2.is_player_in_move,
        "p2MoveId": gameState.player2.move_id,

        # Player 2 buttons
        "p2Up": gameState.player2.player_buttons.up,
        "p2Down": gameState.player2.player_buttons.down,
        "p2Left": gameState.player2.player_buttons.left,
        "p2Right": gameState.player2.player_buttons.right,
        "p2Select": gameState.player2.player_buttons.select,
        "p2Start": gameState.player2.player_buttons.start,
        "p2Y": gameState.player2.player_buttons.Y,
        "p2B": gameState.player2.player_buttons.B,
        "p2X": gameState.player2.player_buttons.X,
        "p2A": gameState.player2.player_buttons.A,
        "p2L": gameState.player2.player_buttons.L,
        "p2R": gameState.player2.player_buttons.R,

        # Round info
        "timer": gameState.timer,
        "roundStarted": gameState.has_round_started,
        "roundOver": gameState.is_round_over,
        "fightResult": gameState.fight_result
    })

def saveGameData(gameState=None):
    """Save collected frame data to CSV file using character name"""
    global frameData
    
    # Generate filename based on player 1's character ID
    if gameState and gameState.player1 and gameState.player1.player_id in CHARACTER_NAMES:
        characterName = CHARACTER_NAMES[gameState.player1.player_id]
        filename = f"{characterName}.csv"
    else:
        # Fallback to argument or default name
        filename = sys.argv[1] if len(sys.argv) > 1 else "gameData.csv"
    
    df = pd.DataFrame(frameData)
    outputPath = "training_data/" + filename
    df.to_csv(outputPath, index=False)
    print(f"Game data saved to {outputPath}")

def connectToGame(port):
    """Establish connection with the game client"""
    serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serverSocket.bind(("127.0.0.1", port))
    serverSocket.listen(5)
    (clientSocket, _) = serverSocket.accept()
    print("Connected to game!")
    return clientSocket

def sendCommand(clientSocket, command):
    """Send controller command to the game"""
    commandDict = command.object_to_dict()
    payload = json.dumps(commandDict).encode()
    clientSocket.sendall(payload)

def receiveGameState(clientSocket):
    """Receive and parse current game state"""
    payload = clientSocket.recv(4096)
    inputDict = json.loads(payload.decode())
    gameState = GameState(inputDict)
    return gameState

def getPlayerInput():
    """Get manual keyboard input for player controls"""
    command = Command()
    # Map keyboard keys to game controls
    command.player_buttons.up = keyboard.is_pressed('w')
    command.player_buttons.down = keyboard.is_pressed('s')
    command.player_buttons.left = keyboard.is_pressed('a')
    command.player_buttons.right = keyboard.is_pressed('d')
    command.player_buttons.A = keyboard.is_pressed('j')
    command.player_buttons.B = keyboard.is_pressed('k')
    command.player_buttons.X = keyboard.is_pressed('u')
    command.player_buttons.Y = keyboard.is_pressed('i')
    command.player_buttons.L = keyboard.is_pressed('n')
    command.player_buttons.R = keyboard.is_pressed('m')
    return command

currentState = None
def main():
    # Connect and initialize
    clientSocket = connectToGame(9999)
    
    frameCounter = 0
    
    # Main game loop
    while True:
        # Get current game state
        currentState = receiveGameState(clientSocket)
        
        # Log data if round is active
        if currentState.is_round_over == False:    
            logGameData(currentState, frameCounter)
        
        # Process player input and send to game
        playerCommand = getPlayerInput()
        sendCommand(clientSocket, playerCommand)
        
        # Increment frame counter
        frameCounter += 1

if __name__ == '__main__':
    try:
        main()
    finally:
        # Save data when exiting - pass the last game state if available
        saveGameData(currentState)
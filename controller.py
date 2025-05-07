import socket
import json
import sys
import keyboard  # You can install this via pip: pip install keyboard
from game_state import GameState
from bot import Bot
from command import Command

def connect(port):
    # For making a connection with the game
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(("127.0.0.1", port))
    server_socket.listen(5)
    (client_socket, _) = server_socket.accept()
    print("Connected to game!")
    return client_socket

def send(client_socket, command):
    # This function will send your updated command to Bizhawk so that game reacts according to your command
    command_dict = command.object_to_dict()
    pay_load = json.dumps(command_dict).encode()
    client_socket.sendall(pay_load)

def receive(client_socket):
    # Receive the game state and return the game state
    pay_load = client_socket.recv(4096)
    input_dict = json.loads(pay_load.decode())
    game_state = GameState(input_dict)
    return game_state

def get_manual_command():
    # Handle manual input for Player 1
    command = Command()

    # Set Player 1 controls based on keypresses
    # Example: Using 'w', 'a', 's', 'd' for up, left, down, right movement
    command.player_buttons.up = keyboard.is_pressed('w')
    command.player_buttons.down = keyboard.is_pressed('s')
    command.player_buttons.left = keyboard.is_pressed('a')
    command.player_buttons.right = keyboard.is_pressed('d')
    command.player_buttons.A = keyboard.is_pressed('j')  # Use 'j' for A button
    command.player_buttons.B = keyboard.is_pressed('k')  # Use 'k' for B button
    command.player_buttons.X = keyboard.is_pressed('l')  # Use 'l' for X button
    command.player_buttons.Y = keyboard.is_pressed('i')  # Use 'i' for Y button

    return command

def main():
    if sys.argv[1] not in ['1', '2']:
        raise ValueError("Invalid argument. Please provide '1' or '2' as the argument to control Player 1 or 2.")

    if sys.argv[1] == '1':
        client_socket = connect(9999)
    elif sys.argv[1] == '2':
        client_socket = connect(10000)

    current_game_state = None
    bot = Bot()

    while True:
        current_game_state = receive(client_socket)

        if sys.argv[1] == '1':
            # For Player 1, get the manual command input
            manual_command = get_manual_command()
            send(client_socket, manual_command)  # Send Player 1's manual command
            
        # roundover false -> save data

        # For Player 2, continue using the bot
        if sys.argv[1] == '2':
            bot_command = bot.fight(current_game_state, sys.argv[1])
            send(client_socket, bot_command)

if __name__ == '__main__':
    main()
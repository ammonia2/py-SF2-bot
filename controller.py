import socket
import json
import csv
import datetime
import os
import sys
import keyboard  # You can install this via pip: pip install keyboard
from game_state import GameState
from bot import Bot
from command import Command

def connect(port):
    # For making a connection with the game
    print(f"Attempting to connect on port {port}...")
    try:
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind(("127.0.0.1", port))
        print(f"Socket bound to port {port}, waiting for game to connect...")
        server_socket.listen(5)
        print("Listening for connections...")
        (client_socket, address) = server_socket.accept()
        print(f"Connected to game at {address}!")
        return client_socket
    except Exception as e:
        print(f"Connection error: {e}")
        raise

def send(client_socket, command):
    # This function will send your updated command to Bizhawk so that game reacts according to your command
    command_dict = command.object_to_dict()
    pay_load = json.dumps(command_dict).encode()
    client_socket.sendall(pay_load)

def receive(client_socket):
    # Receive the game state and return both the game state object and original dict
    try:
        pay_load = client_socket.recv(4096)
        input_dict = json.loads(pay_load.decode())
        game_state = GameState(input_dict)
        return game_state, input_dict
    except Exception as e:
        print(f"Error receiving data: {e}")
        return None, None

def get_manual_command():
    # Handle manual input for player
    command = Command()
    
    command.player_buttons.up = keyboard.is_pressed('w')
    command.player_buttons.down = keyboard.is_pressed('s')
    command.player_buttons.left = keyboard.is_pressed('a')
    command.player_buttons.right = keyboard.is_pressed('d')
    command.player_buttons.A = keyboard.is_pressed('j')  # A button
    command.player_buttons.B = keyboard.is_pressed('k')  # B button
    command.player_buttons.X = keyboard.is_pressed('l')  # X button
    command.player_buttons.Y = keyboard.is_pressed('i')  # Y button

    return command

def get_character_name(game_state, player_idx):
    """Get the character name based on character ID"""
    player = getattr(game_state, player_idx)
    character_id = player.character
    
    # Map of character IDs to names - update this with the actual mapping from your game
    character_names = {
        0: "Ryu",
        1: "Ken",
        2: "ChunLi",
        3: "Guile",
        4: "Zangief",
        # Add other character mappings as needed
    }
    
    return character_names.get(character_id, f"Character{character_id}")

def save_data_to_csv(data_rows, filename):
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile(filename)
    
    with open(filename, 'a', newline='') as csvfile:
        if data_rows and len(data_rows) > 0:
            fieldnames = list(data_rows[0].keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
                
            writer.writerows(data_rows)

def main():
    if len(sys.argv) < 3:
        print("Usage: python main.py [player_number] [control_mode]")
        print("player_number: 1 or 2 to control Player 1 or 2")
        print("control_mode: 'manual' or 'bot' or 'record_manual' or 'record_bot' to specify the control method")
        sys.exit(1)
        
    player_num = sys.argv[1]
    control_mode = sys.argv[2]
    
    if player_num not in ['1', '2']:
        raise ValueError("Invalid player number. Please provide '1' or '2' to control Player 1 or 2.")

    if control_mode not in ['manual', 'bot', 'record_manual', 'record_bot']:
        raise ValueError("Invalid control mode. Please specify 'manual', 'bot', 'record_manual', or 'record_bot'.")

    # Determine if we're recording
    is_recording = control_mode.startswith('record_')
    # Determine if we're using manual or bot control
    is_manual = control_mode == 'manual' or control_mode == 'record_manual'
    
    # Connect to the appropriate port based on player number
    port = 9999 if player_num == '1' else 10000
    client_socket = connect(port)
    
    # Set player indices based on player number
    player_idx = "player1" if player_num == '1' else "player2"
    opponent_idx = "player2" if player_num == '1' else "player1"
    
    bot = Bot()
    frame_counter = 0
    
    # Initialize variables for data recording
    data_buffer = []
    buffer_size = 100  # Save every 100 frames to reduce disk I/O
    
    if is_recording:
        # Get character name
        game_state, _ = receive(client_socket)
        if game_state:
            character_name = get_character_name(game_state, player_idx)
        else:
            character_name = "Unknown"
        
        # Determine control type for filename
        control_type = "manual" if is_manual else "bot"
        
        # Generate timestamp for unique filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create CSV filename
        csv_filename = f"training_data/{character_name}.csv"
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
        print(f"Recording data to {csv_filename}")
    
    control_description = "manual" if is_manual else "bot"
    if is_recording:
        control_description += " with recording"
    
    print(f"Starting {control_description} control...")
    print("Press Ctrl+C to exit")
    
    try:
        while True:
            try:
                # Receive game state
                game_state, raw_input_dict = receive(client_socket)
                
                # Handle connection issues or end of round
                if game_state is None:
                    print("Lost connection or received invalid data. Attempting to reconnect...")
                    break
                
                frame_counter += 1
                
                # Skip processing if round is over, but continue to receive data
                if game_state.is_round_over:
                    print("Round over! Waiting for next round...")
                    # Just send a neutral command to keep the connection alive
                    neutral_command = Command()
                    send(client_socket, neutral_command)
                    continue
                
                # Generate command based on control mode
                if is_manual:
                    command = get_manual_command()
                else:  # bot control
                    command = bot.fight(game_state, player_num)
                
                # Send the command
                send(client_socket, command)
                
                # Record data if in record mode
                if is_recording:
                    # Extract player and opponent data
                    player = getattr(game_state, player_idx)
                    opponent = getattr(game_state, opponent_idx)
                    
                    # Extract player data properly
                    player_fields = {}
                    for k, v in player.__dict__.items():
                        if k == "buttons":
                            # Extract button values directly instead of the object reference
                            for button_name, button_value in vars(v).items():
                                player_fields[f"player_button_{button_name}"] = 1 if button_value else 0
                        else:
                            player_fields[f"player_{k}"] = v
                    
                    # Extract opponent data properly
                    opponent_fields = {}
                    for k, v in opponent.__dict__.items():
                        if k == "buttons":
                            # Extract button values directly instead of the object reference
                            for button_name, button_value in vars(v).items():
                                opponent_fields[f"opponent_button_{button_name}"] = 1 if button_value else 0
                        else:
                            opponent_fields[f"opponent_{k}"] = v
                    
                    game_fields = {k: v for k, v in game_state.__dict__.items() 
                                  if k not in [player_idx, opponent_idx]}
                    
                    # Input button fields from command
                    button_fields = {
                        'input_up': int(command.player_buttons.up),
                        'input_down': int(command.player_buttons.down),
                        'input_left': int(command.player_buttons.left),
                        'input_right': int(command.player_buttons.right),
                        'input_A': int(command.player_buttons.A),
                        'input_B': int(command.player_buttons.B),
                        'input_X': int(command.player_buttons.X),
                        'input_Y': int(command.player_buttons.Y)
                    }
                    
                    # Combine all data
                    row_data = {}
                    row_data.update(player_fields)
                    row_data.update(opponent_fields)
                    row_data.update(game_fields)
                    row_data.update(button_fields)
                    row_data['timestamp'] = datetime.datetime.now().timestamp()
                    row_data['frame_number'] = frame_counter
                    
                    # Add to buffer
                    data_buffer.append(row_data)
                    
                    # Write to CSV periodically
                    if len(data_buffer) >= buffer_size:
                        save_data_to_csv(data_buffer, csv_filename)
                        data_buffer = []
                        print(f"Recorded {frame_counter} frames so far")
                
                # Print status every 100 frames
                if frame_counter % 100 == 0:
                    print(f"Frame {frame_counter} - Player health: {getattr(game_state, player_idx).health}, " 
                          f"Opponent health: {getattr(game_state, opponent_idx).health}")
                    
            except socket.error as e:
                print(f"Socket error: {e}")
                break
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                continue
            except Exception as e:
                print(f"Unexpected error: {e}")
                continue
                
    except KeyboardInterrupt:
        print("\nExiting program...")
    finally:
        # Save any remaining data in buffer
        if is_recording and data_buffer and csv_filename:
            save_data_to_csv(data_buffer, csv_filename)
            print(f"Final data saved to {csv_filename}")
        
        try:
            client_socket.close()
            print("Connection closed.")
        except:
            pass

if __name__ == '__main__':
    main()
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
def dropInactiveRows(dataframe, targetColumns):
    """
    Drops rows where all specified target columns are 0/False
    Args:
        dataframe: pandas DataFrame containing the data
        targetColumns: list of column names to check
    Returns:
        DataFrame with inactive rows removed
    """
    # Create a boolean mask that checks if ANY of the target columns are True
    active_mask = dataframe[targetColumns].any(axis=1)
    
    # Keep only rows where at least one target column is True
    filtered_df = dataframe[active_mask]
    
    # Print statistics about removed rows
    total_rows = len(dataframe)
    remaining_rows = len(filtered_df)
    removed_rows = total_rows - remaining_rows
    print(f"Total rows before filtering: {total_rows}")
    print(f"Rows removed (all actions False): {removed_rows}")
    print(f"Remaining rows: {remaining_rows}")
    
    return filtered_df

def concatenateData():
    """Merge all character data in a single DataFrame & filter required cols"""
    data = pd.DataFrame()
    dataDirectory = 'training_data_p2/'

    # Defining expected dtypes for each column
    dtype_dict = {
        "frame": int, "p1Id": int, "p1Health": int, "p1PosX": int, "p1PosY": int, "p1Jump": int,
        "p1Crouch": int, "p1InMove": int, "p1MoveId": int, "p1Up": int, "p1Down": int, "p1Left": int,
        "p1Right": int, "p1Select": int, "p1Start": int, "p1Y": int, "p1B": int, "p1X": int, "p1A": int,
        "p1L": int, "p1R": int, "p2Id": int, "p2Health": int, "p2PosX": int, "p2PosY": int, "p2Jump": int,
        "p2Crouch": int, "p2InMove": int, "p2MoveId": int, "p2Up": int, "p2Down": int, "p2Left": int,
        "p2Right": int, "p2Select": int, "p2Start": int, "p2Y": int, "p2B": int, "p2X": int, "p2A": int,
        "p2L": int, "p2R": int, "timer": int, "roundStarted": int, "roundOver": int, "fightResult": str
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
    columnsToDropInitial=['p1Up', 'p1Down', 'p1Left', 'p1Right', 'p1Y', 'p1B', 'p1X', 'p1A', 'p1L', 'p1R']
    data=dropInactiveRows(data, columnsToDropInitial)
    
    return data

def plot_key_frequencies():
    # Get the data
    data = concatenateData()
    data.size
    # Define the keys we want to analyze
    p1_keys = ['p1Up', 'p1Down', 'p1Left', 'p1Right', 'p1Y', 'p1B', 'p1X', 'p1A', 'p1L', 'p1R']
    p2_keys = ['p2Up', 'p2Down', 'p2Left', 'p2Right', 'p2Y', 'p2B', 'p2X', 'p2A', 'p2L', 'p2R']
    
    # Calculate frequencies
    p1_frequencies = {key: (data[key] != 0).sum() for key in p1_keys}
    p2_frequencies = {key: (data[key] != 0).sum() for key in p2_keys}
    
    # Create figure and axis
    plt.figure(figsize=(15, 6))
    
    # Plot frequencies
    x = range(len(p1_keys))
    width = 0.35
    
    # Simplify key names for display
    p1_labels = [k[2:] for k in p1_keys]  # Remove 'p1' prefix
    
    plt.bar([i - width/2 for i in x], p1_frequencies.values(), width, label='Player 1', color='blue', alpha=0.7)
    plt.bar([i + width/2 for i in x], p2_frequencies.values(), width, label='Player 2', color='red', alpha=0.7)
    
    # Customize the plot
    plt.title('Key Press Frequencies for Player 1 vs Player 2', fontsize=14)
    plt.xlabel('Keys', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.xticks(x, p1_labels, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add value labels on top of each bar
    for i, v in enumerate(p1_frequencies.values()):
        plt.text(i - width/2, v, str(v), ha='center', va='bottom')
    for i, v in enumerate(p2_frequencies.values()):
        plt.text(i + width/2, v, str(v), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('key_frequencies.png')
    plt.close()

if __name__ == "__main__":
    plot_key_frequencies()
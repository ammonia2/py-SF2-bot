# Street Fighter II – PyTorch Bot  
*A lightweight MLP agent that learns to press the right buttons from recorded gameplay frames.*

---

## ✨ Project Goal
1. **Combine** raw CSV logs from multiple fights.  
2. **Clean & enrich** the data (distance features, health normalization, one-hot character IDs).  
3. **Train** a multilayer perceptron (MLP) that maps a flattened game-state snapshot → the 10 possible buttons Player 1 can press.  
4. **Export** the scaler, feature list, and `.pth` weights so the bot can run inside the BizHawk / PythonAPI loop.  
5. *(Upcoming)* **Mirror** data so the model can fight as *Player 2* without retraining.

---

## 🗂️ Repository Layout

```
.
├── training_data_p2/       # CSV logs of recorded gameplay
├── model.py                # Main training script & model class
├── SF2_model.pth           # Trained model weights
├── scaler.joblib           # StandardScaler for feature normalization
├── feature_names.joblib    # Preserved feature list for inference
└── README.md               # Project documentation
```

## 🎮 How It Works

The project creates an AI agent for Super Street Fighter II Turbo through a behavior cloning approach:

1. **Data Collection**: 
   - Raw CSV logs are recorded from gameplay with frame-by-frame state information
   - Each row contains player positions, health, button presses, character IDs, etc.

2. **Data Preprocessing**:
   - Merges data from multiple fights
   - Normalizes health, timer, positions using StandardScaler
   - Calculates distance features between players (xDist, yDist)
   - One-hot encodes character IDs (12 characters × 2 players)
   - Filters frames where player is actually pressing buttons

3. **Model Architecture**:
   - Multi-layer perceptron (MLP)
   - Input: Flattened game state (positions, health, character IDs, etc.)
   - 3 hidden layers with 64 neurons each and ReLU activation
   - Output: 10 sigmoid units (one for each possible button press)

4. **Training Process**:
   - Binary cross-entropy loss for multi-label classification
   - Adam optimizer with learning rate 0.001
   - 30 epochs with batch size 64
   - 80/20 train/test split

5. **Output**:
   - Trained model weights (.pth file)
   - Feature scaler for normalizing new inputs
   - List of feature names for inference

## 🚀 Usage

### Prerequisites
- Python 3.7+
- PyTorch
- pandas
- numpy
- scikit-learn
- joblib

### Training

To train the model:

```bash
python model.py
```

This will:
- Load and preprocess all CSV files in the `training_data_p2/` directory
- Train the MLP model
- Output model weights, feature names, and scaler to the current directory

### Inference

The inference code should be integrated into a BizHawk emulator loop. 
Sample pseudocode for using the model in an emulator:

```python
import torch
import joblib
import numpy as np

# Load model and preprocessing artifacts
model = torch.load('SF2_model.pth')
scaler = joblib.load('scaler.joblib')
feature_names = joblib.load('feature_names.joblib')

def get_action(game_state):
    # Preprocess the current game state
    processed_state = preprocess_state(game_state, scaler, feature_names)
    
    # Convert to tensor
    state_tensor = torch.FloatTensor(processed_state)
    
    # Get model predictions
    with torch.no_grad():
        actions = model(state_tensor)
    
    # Convert to binary actions (press/don't press)
    binary_actions = (actions > 0.5).float().numpy()
    
    return binary_actions
```

## 📊 Performance

The model achieves approximately:
- Training accuracy: ~90-92%
- Validation accuracy: ~85-87%

This is expected behavior for a behavior cloning approach, as the model learns to mimic the recorded gameplay rather than optimizing win rate directly.

## 🔮 Future Improvements

1. **Data Mirroring**: Implement mirroring of training data so the model can play as Player 2 without retraining
2. **Hyperparameter Tuning**: Experiment with different network architectures and learning rates
3. **Frame Context**: Include previous N frames as input to better capture action sequences
4. **Reinforcement Learning**: Implement RL fine-tuning to optimize for win rate after initial behavior cloning
5. **Character-Specific Models**: Train specialized models for each character matchup

## 📝 License


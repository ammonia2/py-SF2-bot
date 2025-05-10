# Street Fighter II - AI Bot

This repository contains an AI bot designed to play **Street Fighter II** using deep learning and automation via the **EmuHawk Emulator**.

---

## 📁 Repository Structure

```
.
├── bot.py             # Main bot logic
├── buttons.py         # Game button definitions
├── command.py         # Command execution helper
├── controller.py      # Simulates controller inputs
├── game_state.py      # Parses emulator game state
├── player.py          # Handles player state data
├── README.md          # Setup and usage instructions
```

---

## ✅ Prerequisites

### 1. Install .NET Framework

EmuHawk requires .NET Framework to run.  
Install it from:  
🔗 https://dotnet.microsoft.com/en-us/download

---

## 📦 Python Library Installation

Ensure Python 3.8+ is installed. Then run:

```bash
pip install tensorflow torch scikit-learn numpy pandas joblib keyboard
```

---

## 🕹️ EmuHawk Setup

1. Download and extract EmuHawk:  
   🔗 https://tasvideos.org/BizHawk

2. Launch EmuHawk.

3. Open the **Street Fighter II** ROM.

4. Start the game and select a character for the bot to control.

---

## 🚀 Running the Bot

1. In EmuHawk, go to **Toolbox > Gyroscope Bot**, and connect the API.

2. In a terminal, start the controller handler:

```bash
python controller.py
```

3. Then start the AI bot:

```bash
python bot.py
```

---

## ⚠️ Notes

- Make sure EmuHawk is in focus during gameplay.
- Keep `controller.py` running in the background.
- Lua scripting or EmuHawk API may need to be enabled for input/output communication.

---

## 🧩 Troubleshooting

- If input isn’t registering, ensure EmuHawk is active and the bot is connected via the Toolbox.
- Verify all libraries are installed correctly via `pip list`.

---

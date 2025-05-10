# Street Fighter II – PyTorch Bot  
_A lightweight MLP agent that learns to press the right buttons from recorded gameplay frames._

---

## ✨ Project Goal
1. **Combine** raw CSV logs from multiple fights.  
2. **Clean & enrich** the data (distance features, health normalization, one-hot character IDs).  
3. **Train** a multilayer perceptron (MLP) that maps a flattened game-state snapshot → the 10 possible buttons Player 1 can press.  
4. **Export** the scaler, feature list, and `.pth` weights so the bot can run inside the BizHawk / PythonAPI loop.  
5. (Upcoming) **Mirror** data so the model can fight as _Player 2_ without retraining.

---

## 🗂️ Repository Layout


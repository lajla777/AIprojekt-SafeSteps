import torch
import numpy as np
from model import ObstacleCNN

LABEL_NAMES = ["no_obstacle", 
               "obstacle_left", 
               "obstacle_right",
               "obstacle_center", 
               "obstacle_right_left"]

def predict(angle_distance_window):

    model = ObstacleCNN()
    model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
    model.eval()

    x = torch.tensor(angle_distance_window, dtype=torch.float32).unsqueeze(0)  # (1, T, 2)
    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1).squeeze()
        idx    = probs.argmax().item()

    print(f"Napoved: {LABEL_NAMES[idx]}  (zaupanje: {probs[idx]:.2%})")
    for i, (name, p) in enumerate(zip(LABEL_NAMES, probs)):
        print(f"  {name}: {p:.2%}")

    return LABEL_NAMES[idx]
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium.wrappers as wrappers
import os

# Hyperparameters
vision_model_hidden_dim = 256
memory_model_hidden_dim = 256
action_dim = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Models
class VisionModel(nn.Module):
    def __init__(self):
        super(VisionModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(64 * 8 * 8, vision_model_hidden_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        return F.relu(self.fc(x))

class MemoryModel(nn.Module):
    def __init__(self):
        super(MemoryModel, self).__init__()
        self.lstm = nn.LSTM(vision_model_hidden_dim, memory_model_hidden_dim)

    def forward(self, x, hidden_state):
        x, hidden_state = self.lstm(x.unsqueeze(0), hidden_state)
        return x.squeeze(0), hidden_state

class Controller(nn.Module):
    def __init__(self):
        super(Controller, self).__init__()
        self.fc = nn.Linear(memory_model_hidden_dim, action_dim)

    def forward(self, x):
        return torch.tanh(self.fc(x))

# Preprocessing
def preprocess_state(state):
    state = torch.from_numpy(state).float() / 255.0
    state = state.permute(2, 0, 1)
    return state.unsqueeze(0).to(device)

# Load environment with video recording
env = gym.make('CarRacing-v3', render_mode='rgb_array')
env = wrappers.RecordVideo(
    env,
    video_folder='./videos',
    name_prefix='world_model_eval',
    episode_trigger=lambda x: True
)

# Instantiate models
vision_model = VisionModel().to(device)
memory_model = MemoryModel().to(device)
controller = Controller().to(device)

# Load best checkpoint
checkpoint_path = "checkpoints/best_model.pth"
assert os.path.exists(checkpoint_path), "Best model checkpoint not found!"
checkpoint = torch.load(checkpoint_path, map_location=device)

vision_model.load_state_dict(checkpoint['vision_model'])
memory_model.load_state_dict(checkpoint['memory_model'])
controller.load_state_dict(checkpoint['controller'])

vision_model.eval()
memory_model.eval()
controller.eval()

# Evaluation loop
state, _ = env.reset()
hidden_state = (torch.zeros(1, 1, memory_model_hidden_dim).to(device),
                torch.zeros(1, 1, memory_model_hidden_dim).to(device))
total_reward = 0
done = False

while not done:
    state_tensor = preprocess_state(state)
    with torch.no_grad():
        vision_output = vision_model(state_tensor)
        memory_output, hidden_state = memory_model(vision_output, hidden_state)
        action = controller(memory_output)

    state, reward, done, truncated, _ = env.step(action.squeeze(0).cpu().numpy())
    total_reward += reward
    if truncated:
        break

env.close()
print(f"Total Reward Achieved: {total_reward:.2f}")
print("Evaluation completed. Video saved in ./videos folder.")
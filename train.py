### Full World Model - Production Ready Version (Fixed)

import gymnasium as gym
import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Hyperparameters
vision_model_hidden_dim = 256
memory_model_hidden_dim = 256
controller_hidden_dim = 256
action_dim = 3
learning_rate = 1e-4
num_episodes = 500
max_steps = 1000
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

# Setup
env = gym.make('CarRacing-v3')
vision_model = VisionModel().to(device)
memory_model = MemoryModel().to(device)
controller = Controller().to(device)

vision_optimizer = optim.Adam(vision_model.parameters(), lr=learning_rate)
memory_optimizer = optim.Adam(memory_model.parameters(), lr=learning_rate)
controller_optimizer = optim.Adam(controller.parameters(), lr=learning_rate)

mse_loss = nn.MSELoss()

# Training Loop
for episode in range(num_episodes):
    state, _ = env.reset()
    hidden_state = (torch.zeros(1, 1, memory_model_hidden_dim).to(device),
                    torch.zeros(1, 1, memory_model_hidden_dim).to(device))
    episode_reward = 0
    total_loss = 0

    for t in range(max_steps):
        state_tensor = preprocess_state(state)
        vision_output = vision_model(state_tensor)

        # Detach hidden state to avoid in-place modification issues
        hidden_state = (hidden_state[0].detach(), hidden_state[1].detach())
        memory_output, hidden_state = memory_model(vision_output, hidden_state)

        action = controller(memory_output)
        next_state, reward, done, truncated, _ = env.step(action.squeeze(0).cpu().detach().numpy())
        next_state_tensor = preprocess_state(next_state)

        # Predict next latent state
        with torch.no_grad():
            target_latent = vision_model(next_state_tensor)

        # Compute prediction loss (vision + memory)
        prediction_loss = mse_loss(memory_output, target_latent.detach())

        # Compute reward loss (encourage higher rewards)
        reward_loss = -reward

        # Total loss
        loss = prediction_loss + 0.01 * reward_loss

        vision_optimizer.zero_grad()
        memory_optimizer.zero_grad()
        controller_optimizer.zero_grad()

        loss.backward()

        vision_optimizer.step()
        memory_optimizer.step()
        controller_optimizer.step()

        state = next_state
        episode_reward += reward
        total_loss += loss.item()

        if done or truncated:
            break

    print(f"Episode {episode+1}/{num_episodes}, Reward: {episode_reward:.2f}, Loss: {total_loss:.4f}")

    # Save model checkpoints every 50 episodes
    if (episode + 1) % 50 == 0:
        torch.save({
            'vision_model': vision_model.state_dict(),
            'memory_model': memory_model.state_dict(),
            'controller': controller.state_dict()
        }, f"world_model_checkpoint_{episode+1}.pth")

env.close()

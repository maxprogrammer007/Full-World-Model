# world_model_carracing.py

import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# -----------------------------------------------------------------------------
#  Hyperparameters
# -----------------------------------------------------------------------------
IMG_SIZE        = 96
LATENT_DIM      = 32
VISION_HIDDEN   = 256
RNN_HIDDEN      = 256
ACTION_DIM      = 3
VAE_LR          = 1e-4
RNN_LR          = 1e-4
POLICY_LR       = 5e-4
VAE_EPOCHS      = 10
RNN_EPOCHS      = 10
VAE_BATCH       = 64
RNN_BATCH       = 16
SEQ_LEN         = 50
NUM_SEQ         = 200
POLICY_EPISODES = 500
POLICY_MAX_STEPS= 1000
GAMMA           = 0.99

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
#  Utils
# -----------------------------------------------------------------------------
def preprocess(frame):
    """Normalize and convert H×W×C → C×H×W tensor."""
    t = torch.from_numpy(frame).float() / 255.0
    return t.permute(2, 0, 1).unsqueeze(0).to(device)

def discounted_returns(rewards, γ=GAMMA):
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + γ * R
        returns.insert(0, R)
    returns = torch.tensor(returns, dtype=torch.float32, device=device)
    # normalize
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns

# -----------------------------------------------------------------------------
#  Phase 1: Convolutional VAE
# -----------------------------------------------------------------------------
class VAE(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        # encoder
        self.conv1    = nn.Conv2d(3, 32, 8, 4)   # → 32×23×23
        self.conv2    = nn.Conv2d(32,64, 4, 2)   # → 64×10×10
        self.conv3    = nn.Conv2d(64,64, 3, 1)   # → 64× 8× 8
        self.fc_enc   = nn.Linear(64*8*8, VISION_HIDDEN)
        self.fc_mu    = nn.Linear(VISION_HIDDEN, latent_dim)
        self.fc_logvar= nn.Linear(VISION_HIDDEN, latent_dim)
        # decoder
        self.fc_dec   = nn.Linear(latent_dim, 64*8*8)
        self.deconv1  = nn.ConvTranspose2d(64,64,3,1)       # → 64×10×10
        self.deconv2  = nn.ConvTranspose2d(64,32,4,2,0,1)   # → 32×23×23
        self.deconv3  = nn.ConvTranspose2d(32,3, 8,4,0,0)   # →  3×96×96

    def encode(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        h = F.relu(self.fc_enc(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = F.relu(self.fc_dec(z))
        x = x.view(-1, 64, 8, 8)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        return torch.sigmoid(self.deconv3(x))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def collect_random_frames(env, n_frames=50000):
    frames = []
    state, _ = env.reset()
    while len(frames) < n_frames:
        action = env.action_space.sample()
        next_state, _, done, truncated, _ = env.step(action)
        frames.append(state)
        if done or truncated:
            state, _ = env.reset()
        else:
            state = next_state
    return frames

def train_vae(vae, frames):
    optimizer = optim.Adam(vae.parameters(), lr=VAE_LR)
    for epoch in range(1, VAE_EPOCHS+1):
        random.shuffle(frames)
        epoch_loss = 0
        for i in range(0, len(frames), VAE_BATCH):
            batch = frames[i:i+VAE_BATCH]
            x = np.stack(batch) / 255.0
            x = torch.from_numpy(x).permute(0,3,1,2).float().to(device)
            recon, mu, logvar = vae(x)
            recon_loss = F.mse_loss(recon, x)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + 1e-3 * kl
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)
        print(f"[VAE] Epoch {epoch}/{VAE_EPOCHS}  avg_loss: {epoch_loss/len(frames):.6f}")

# -----------------------------------------------------------------------------
#  Phase 2: LSTM world-model (latent + action → next latent)
# -----------------------------------------------------------------------------
class MDNRNN(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, action_dim=ACTION_DIM, hidden_dim=RNN_HIDDEN):
        super().__init__()
        self.lstm   = nn.LSTM(latent_dim + action_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, latent_dim)

    def forward(self, latents, actions):
        # latents: [B,T,latent], actions: [B,T,action]
        inp, _ = self.lstm(torch.cat([latents, actions], dim=-1))
        return self.fc_out(inp)  # [B,T,latent]

def collect_random_sequences(env, vae):
    seqs = []
    while len(seqs) < NUM_SEQ:
        state, _ = env.reset()
        latents, actions = [], []
        for t in range(SEQ_LEN+1):
            z = vae.encode(preprocess(state))[0].detach().cpu().numpy()
            latents.append(z)
            if t < SEQ_LEN:
                a = env.action_space.sample()
                actions.append(a)
                nxt, _, done, trunc, _ = env.step(a)
                if done or trunc:
                    break
                state = nxt
        if len(latents) == SEQ_LEN+1:
            seqs.append((
                np.stack(latents),     # (T+1,latent)
                np.stack(actions)      # (T,action)
            ))
    return seqs

def train_rnn(rnn, sequences):
    optimizer = optim.Adam(rnn.parameters(), lr=RNN_LR)
    mse       = nn.MSELoss()
    for epoch in range(1, RNN_EPOCHS+1):
        random.shuffle(sequences)
        total_loss = 0
        for i in range(0, len(sequences), RNN_BATCH):
            batch = sequences[i:i+RNN_BATCH]
            L = np.stack([b[0] for b in batch])  # (B, T+1, latent)
            A = np.stack([b[1] for b in batch])  # (B, T, action)
            L = torch.from_numpy(L[:, :-1]).float().to(device)  # input latents
            A = torch.from_numpy(A).float().to(device)
            target = torch.from_numpy(L.cpu().numpy()[:,1:]).float().to(device)  # next latents
            pred   = rnn(L, A)
            loss   = mse(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * L.size(0)
        print(f"[RNN] Epoch {epoch}/{RNN_EPOCHS}  avg_mse: {total_loss/len(sequences):.6f}")

# -----------------------------------------------------------------------------
#  Phase 3: Controller via REINFORCE in latent space
# -----------------------------------------------------------------------------
class Controller(nn.Module):
    def __init__(self, hidden_dim=RNN_HIDDEN, action_dim=ACTION_DIM):
        super().__init__()
        self.fc     = nn.Linear(hidden_dim, action_dim)
        self.logstd = nn.Parameter(torch.zeros(action_dim))

    def forward(self, h):
        mu  = self.fc(h)
        std = torch.exp(self.logstd)
        return torch.distributions.Normal(mu, std)

def train_controller(env, controller, vae, rnn):
    optimizer = optim.Adam(controller.parameters(), lr=POLICY_LR)
    for ep in range(1, POLICY_EPISODES+1):
        state, _ = env.reset()
        # reset RNN hidden state
        h = (torch.zeros(1,1,RNN_HIDDEN).to(device),
             torch.zeros(1,1,RNN_HIDDEN).to(device))
        prev_action = torch.zeros(ACTION_DIM, device=device)
        logps, rewards = [], []

        for t in range(POLICY_MAX_STEPS):
            z_mu, _ = vae.encode(preprocess(state))
            # run one step of RNN
            inp = torch.cat([z_mu, prev_action.unsqueeze(0)], dim=-1).unsqueeze(0)
            out, h = rnn.lstm(inp, h)
            h0 = out[0,0]  # (hidden_dim,)

            dist   = controller(h0)
            a      = dist.rsample()
            logp   = dist.log_prob(a).sum()
            action = torch.tanh(a)

            nxt, reward, done, trunc, _ = env.step(action.cpu().numpy())
            logps.append(logp)
            rewards.append(reward)
            prev_action = action.detach()
            state = nxt
            if done or trunc:
                break

        # compute loss
        returns = discounted_returns(rewards)
        policy_loss = 0
        for lp, G in zip(logps, returns):
            policy_loss += -lp * G
        policy_loss = policy_loss / len(logps)

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        total_reward = sum(rewards)
        print(f"[POL] Ep {ep}/{POLICY_EPISODES} | R = {total_reward:.2f}")

# -----------------------------------------------------------------------------
#  Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    env = gym.make("CarRacing-v3", render_mode=None)
    # Phase 1
    vae = VAE().to(device)
    print("Collecting random frames for VAE...")
    frames = collect_random_frames(env, n_frames=20000)
    print("Training VAE...")
    train_vae(vae, frames)

    # Phase 2
    rnn = MDNRNN().to(device)
    print("Collecting random sequences for MDN-RNN...")
    seqs = collect_random_sequences(env, vae)
    print("Training MDN-RNN...")
    train_rnn(rnn, seqs)

    # Phase 3
    controller = Controller().to(device)
    print("Training Controller via REINFORCE...")
    train_controller(env, controller, vae, rnn)

    env.close()

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import os

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x): return self.net(x)

class DQNAgent:
    def __init__(self, state_dim=6, action_dim=4, device=None):
        # device selection: allow override, otherwise use CUDA if available
        self.device = torch.device(device) if device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self._init_model()
        print(self.device)
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        self.batch_size = 64

    def _init_model(self):
        # create models and move them to device
        self.model = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def act(self, state, greedy=False, valid_actions=None):
        if valid_actions is not None and not valid_actions.any():
            valid_actions = None
        if not greedy and random.random() < self.epsilon:
            if valid_actions is not None:
                choices = np.flatnonzero(valid_actions)
                return int(np.random.choice(choices))
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.model(state_tensor)
            q_values = q.squeeze(0)
            if valid_actions is not None:
                mask = torch.tensor(valid_actions, dtype=torch.bool, device=self.device)
                masked_q = q_values.clone()
                masked_q[~mask] = -1e9
                return int(torch.argmax(masked_q).item())
            return int(torch.argmax(q_values).item())

    def remember(self, s, a, r, s_, done):
        self.memory.append((np.array(s, dtype=np.float32), int(a), float(r), np.array(s_, dtype=np.float32), bool(done)))

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # convert to tensors on device
        states = torch.tensor(np.stack(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.stack(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q = self.target(next_states).max(1)[0].detach()
        target_q = rewards + self.gamma * next_q * (1 - dones)
        loss = nn.functional.mse_loss(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target(self):
        self.target.load_state_dict(self.model.state_dict())

    def boost_exploration(self, epsilon_floor=0.3):
        """Increase epsilon when switching to a fresh maze to promote exploration."""
        epsilon_floor = min(max(epsilon_floor, self.epsilon_min), 1.0)
        self.epsilon = max(self.epsilon, epsilon_floor)

    def save(self, path="dqn_model.pth"):
        # save CPU tensors for portability
        cpu_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
        torch.save(cpu_state, path)

    def load(self, path="dqn_model.pth"):
        """
        Load weights safely. Do NOT change self.state_dim here.
        Try exact load first; if it fails, copy only tensors whose shapes match
        the current model to allow partial weight reuse across different architectures.
        """
        if not os.path.exists(path):
            print("[INFO] No saved model found. Starting fresh.")
            return

        try:
            saved_state = torch.load(path, map_location="cpu")
        except Exception as e:
            print(f"[WARN] Could not read checkpoint: {e}. Starting fresh.")
            return

        # Try exact load first (PyTorch will handle device copy)
        try:
            self.model.load_state_dict(saved_state)
            self.model.to(self.device)
            self.update_target()
            self.target.to(self.device)
            print("[INFO] Model loaded (exact match).")
            return
        except Exception:
            pass

        # Partial load: copy matching-shape tensors only (do not modify self.state_dim)
        try:
            current_state = self.model.state_dict()
            copied = 0
            skipped = 0
            for k, v in saved_state.items():
                if k in current_state and v.shape == current_state[k].shape:
                    # copy into same device as current tensor
                    current_state[k] = v.to(current_state[k].device)
                    copied += 1
                else:
                    skipped += 1
            # load the merged state dict and move model to device
            self.model.load_state_dict(current_state)
            self.model.to(self.device)
            self.update_target()
            self.target.to(self.device)
            print(f"[INFO] Partial model load complete. Copied tensors: {copied}, Skipped tensors: {skipped}")
        except Exception as e:
            print(f"[WARN] Partial load failed: {e}. Starting fresh.")

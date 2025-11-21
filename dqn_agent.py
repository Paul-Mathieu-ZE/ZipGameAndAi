import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import os

class DQN(nn.Module):
    def __init__(self, grid_shape, meta_dim, output_dim):
        super().__init__()
        channels, height, width = grid_shape
        self.meta_dim = meta_dim
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        with torch.no_grad():
            dummy = torch.zeros(1, channels, height, width)
            conv_out = self.conv(dummy)
            conv_flat = conv_out.view(1, -1).shape[1]
        self.conv_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_flat, 256),
            nn.ReLU()
        )
        self.meta_head = None
        meta_out = 0
        if meta_dim > 0:
            self.meta_head = nn.Sequential(
                nn.Linear(meta_dim, 64),
                nn.ReLU()
            )
            meta_out = 64
        self.head = nn.Sequential(
            nn.Linear(256 + meta_out, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, grid, meta=None):
        conv_feat = self.conv_head(self.conv(grid))
        if self.meta_head is not None and meta is not None:
            meta_feat = self.meta_head(meta)
            x = torch.cat([conv_feat, meta_feat], dim=1)
        else:
            x = conv_feat
        return self.head(x)

class DQNAgent:
    def __init__(self, state_dim=6, action_dim=4, grid_shape=(3, 20, 20), meta_dim=8, device=None):
        # device selection: allow override, otherwise use CUDA if available
        self.device = torch.device(device) if device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.grid_shape = tuple(grid_shape)
        self.meta_dim = int(meta_dim)
        self._init_model()
        print(self.device)
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        self.batch_size = 64
        self.global_steps = 0

    def _init_model(self):
        # create models and move them to device
        self.model = DQN(self.grid_shape, self.meta_dim, self.action_dim).to(self.device)
        self.target = DQN(self.grid_shape, self.meta_dim, self.action_dim).to(self.device)
        self.target.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def _decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def _split_state(self, state):
        state = np.asarray(state, dtype=np.float32)
        meta = state[:self.meta_dim] if self.meta_dim > 0 else None
        grid = state[self.meta_dim:].reshape(self.grid_shape)
        return meta, grid

    def act(self, state, greedy=False, valid_actions=None):
        if valid_actions is not None and not valid_actions.any():
            valid_actions = None
        action = None
        if not greedy and random.random() < self.epsilon:
            if valid_actions is not None:
                choices = np.flatnonzero(valid_actions)
                action = int(np.random.choice(choices))
            else:
                action = random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                meta, grid = self._split_state(state)
                grid_tensor = torch.tensor(grid, dtype=torch.float32, device=self.device).unsqueeze(0)
                meta_tensor = torch.tensor(meta, dtype=torch.float32, device=self.device).unsqueeze(0) if meta is not None else None
                q = self.model(grid_tensor, meta_tensor)
                q_values = q.squeeze(0)
                if valid_actions is not None:
                    mask = torch.tensor(valid_actions, dtype=torch.bool, device=self.device)
                    masked_q = q_values.clone()
                    masked_q[~mask] = -1e9
                    action = int(torch.argmax(masked_q).item())
                else:
                    action = int(torch.argmax(q_values).item())
        if not greedy:
            self.global_steps += 1
            self._decay_epsilon()
        return action

    def remember(self, s, a, r, s_, done):
        self.memory.append((np.array(s, dtype=np.float32), int(a), float(r), np.array(s_, dtype=np.float32), bool(done)))

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # convert to tensors on device
        states_np = np.stack(states)
        next_states_np = np.stack(next_states)

        state_meta = torch.tensor(states_np[:, :self.meta_dim], dtype=torch.float32, device=self.device) if self.meta_dim > 0 else None
        state_grid = torch.tensor(states_np[:, self.meta_dim:], dtype=torch.float32, device=self.device).view(-1, *self.grid_shape)
        next_meta = torch.tensor(next_states_np[:, :self.meta_dim], dtype=torch.float32, device=self.device) if self.meta_dim > 0 else None
        next_grid = torch.tensor(next_states_np[:, self.meta_dim:], dtype=torch.float32, device=self.device).view(-1, *self.grid_shape)

        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        q_values = self.model(state_grid, state_meta).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q = self.target(next_grid, next_meta).max(1)[0].detach()
        target_q = rewards + self.gamma * next_q * (1 - dones)
        loss = nn.functional.mse_loss(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        

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

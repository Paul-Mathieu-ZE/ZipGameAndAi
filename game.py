from concurrent.futures import thread
import board_gen 
from board_gen import *
import random
import tkinter as tk
from tkinter import ttk
import numpy as np
from board_gen import Board
from dqn_agent import DQNAgent
import threading
from threading import Thread
import time 

CELL_SIZE = 20

class MazeEnv:
    def __init__(self, width, height):
        self.width, self.height = width, height
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.white_distance = None
        self.reset()

    def reset(self):
        b = Board(sizeX=self.height, sizeY=self.width)
        self.board = np.array(b.board, dtype=np.int8)
        s = np.argwhere(self.board == 2)
        g = np.argwhere(self.board == 3)
        if len(s) == 0 or len(g) == 0:
            return self.reset()
        self.start = tuple(s[0])
        self.goal = tuple(g[0])
        self.agent_pos = self.start
        # compute white distance at reset so callers can read it
        self.white_distance = self.compute_white_distance()
        return self._get_state()

    def step(self, action):
        dx, dy = self.actions[action]
        x, y = self.agent_pos
        nx, ny = x + dx, y + dy
        if not (0 <= nx < self.height and 0 <= ny < self.width):
            return self._get_state(), -0.05, False
        if int(self.board[nx, ny]) == 1:
            return self._get_state(), -0.1, False
        self.agent_pos = (nx, ny)
        reward = 1.0 if self.agent_pos == self.goal else -0.01
        # update white_distance when agent moves (optional small cost to recompute is fine)
        self.white_distance = self.compute_white_distance()
        return self._get_state(), reward, self.agent_pos == self.goal

    def compute_white_distance(self):
        """Return shortest path length in white/free cells from agent (or start) to goal using BFS.
           Returns None if unreachable, or integer steps if reachable.
        """
        from collections import deque
        sx, sy = self.agent_pos if hasattr(self, "agent_pos") and self.agent_pos is not None else self.start
        gx, gy = self.goal
        visited = np.zeros_like(self.board, dtype=bool)
        q = deque()
        q.append((sx, sy, 0))
        visited[sx, sy] = True
        while q:
            x, y, d = q.popleft()
            if (x, y) == (gx, gy):
                return d
            for dx, dy in self.actions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.height and 0 <= ny < self.width and not visited[nx, ny] and int(self.board[nx, ny]) != 1:
                    visited[nx, ny] = True
                    q.append((nx, ny, d + 1))
        return None

    def _get_state(self):
        """
        State = [ax_norm, ay_norm, gx_norm, gy_norm, norm_shortest_white_distance] + flattened wall_map
        shortest_white_distance = shortest path length from agent to goal using only non-wall cells (BFS)
        """
        ax, ay = self.agent_pos
        gx, gy = self.goal

        shortest = self.white_distance if self.white_distance is not None else self.compute_white_distance()
        if shortest is None:
            shortest = self.width * self.height  # sentinel large
        norm_dist = float(shortest) / float(self.width * self.height)

        meta = np.array([
            ax / float(self.height),
            ay / float(self.width),
            gx / float(self.height),
            gy / float(self.width),
            norm_dist
        ], dtype=np.float32)

        wall_map = (self.board == 1).astype(np.float32).ravel()
        return np.concatenate([meta, wall_map]).astype(np.float32)


    def get_path(self, agent):
        self.agent_pos = self.start
        path = [self.start]
        for _ in range(500):
            state = self._get_state()
            action = agent.act(state, greedy=True)
            _, _, done = self.step(action)
            path.append(self.agent_pos)
            if done: break
        return path




class MazeApp:
    def __init__(self, root):
        self.root = root
        # helper to schedule UI updates on the Tk main thread from worker threads
        self._ui = lambda fn, *a, **kw: self.root.after(0, lambda: fn(*a, **kw))
        # ensure commonly used attributes exist (won't override if already set later)
        if not hasattr(self, "canvas"):
            self.canvas = tk.Canvas(self.root)
        if not hasattr(self, "stats"):
            self.stats = ttk.Label(self.root, text="")
        if not hasattr(self, "agent"):
            self.agent = None
        if not hasattr(self, "env"):
            self.env = None
        # graceful shutdown support for training thread
        self.stop_event = threading.Event()
        self.training_thread = None
        # ensure window close goes through our handler so background threads stop
        try:
            self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        except Exception:
            pass

        self.root.title("Maze AI Trainer")
        self.canvas = tk.Canvas(root, width=600, height=600, bg="white")
        self.canvas.pack()
        self.stats = ttk.Label(root, text="Stats:"); self.stats.pack()
        self.setup_controls()

    def setup_controls(self):
        frame = ttk.Frame(self.root); frame.pack(pady=10)
        self.width_entry = ttk.Entry(frame, width=5); self.width_entry.insert(0, "20")
        self.height_entry = ttk.Entry(frame, width=5); self.height_entry.insert(0, "20")
        ttk.Label(frame, text="Width:").grid(row=0, column=0); self.width_entry.grid(row=0, column=1)
        ttk.Label(frame, text="Height:").grid(row=0, column=2); self.height_entry.grid(row=0, column=3)
        ttk.Button(frame, text="Generate", command=self.generate_maze).grid(row=0, column=4)
        ttk.Button(frame, text="Train", command=self.start_training_thread).grid(row=0, column=5)
        ttk.Button(frame, text="Show Path", command=self.show_path).grid(row=0, column=6)

    def generate_maze(self):
        w, h = int(self.width_entry.get()), int(self.height_entry.get())
        self.env = MazeEnv(w, h)
        if not self.agent:
            self.agent = DQNAgent()
            self.agent.load()
        self.draw_static_maze()
        self.draw_agent()

    def draw_static_maze(self):
        # draw the current env grid and force a UI refresh (must run on main thread)
        if not self.env:
            return
        rows, cols = int(self.env.height), int(self.env.width)
        # set canvas size to match maze
        self.canvas.config(width=cols * CELL_SIZE, height=rows * CELL_SIZE)
        # remove only maze items (keep other tags like "path", "agent", etc.)
        try:
            self.canvas.delete("maze")
        except Exception:
            self.canvas.delete("all")
        # draw cells row-major (i = row, j = col)
        for i in range(rows):
            for j in range(cols):
                cell = int(self.env.board[i, j])
                x0 = j * CELL_SIZE
                y0 = i * CELL_SIZE
                x1 = x0 + CELL_SIZE
                y1 = y0 + CELL_SIZE
                if cell == 1:
                    color = "black"
                elif cell == 2:
                    color = "green"
                elif cell == 3:
                    color = "red"
                else:
                    color = "white"
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="#ccc", tags="maze")
        # ensure the mainloop processes pending drawing events immediately
        try:
            self.root.update_idletasks()
            self.root.update()
        except Exception:
            # update may fail if called during teardown; ignore
            pass
        # optional log to console if present
        if hasattr(self, "console") and self.console is not None:
            try:
                self.console.log("[INFO] Maze drawn\n")
            except Exception:
                pass

    def draw_agent(self):
        self.canvas.delete("agent")
        ax, ay = self.env.agent_pos
        x0 = ay * CELL_SIZE + 4
        y0 = ax * CELL_SIZE + 4
        x1 = x0 + CELL_SIZE - 8
        y1 = y0 + CELL_SIZE - 8
        self.canvas.create_oval(x0, y0, x1, y1, fill="blue", outline="", tags="agent")

    def draw_path(self, path):
        for px, py in path:
            x0 = py * CELL_SIZE + 8
            y0 = px * CELL_SIZE + 8
            x1 = x0 + 4
            y1 = y0 + 4
            self.canvas.create_oval(x0, y0, x1, y1, fill="cyan", outline="")


    def on_close(self):
        """Signal threads to stop, wait briefly, then destroy the UI."""
        # set stop flag so training loop can exit
        self.stop_event.set()
        # if there's a training thread, wait a short while for it to finish
        if getattr(self, "training_thread", None) and self.training_thread.is_alive():
            try:
                self.training_thread.join(timeout=1.0)
            except Exception:
                pass
        try:
            self.root.destroy()
        except Exception:
            pass

    def start_training_thread(self):
        if self.training_thread and self.training_thread.is_alive():
            return
        # clear any previous stop request and start daemon thread
        self.stop_event.clear()
        self.training_thread = Thread(target=self.train_on_multiple_mazes, daemon=True)
        self.training_thread.start()

    def train_on_multiple_mazes(self, mazes=100, episodes_per_maze=100):
        for maze_index in range(mazes):
            # allow graceful exit requested from UI
            if self.stop_event.is_set():
                return
            # create valid maze on worker thread
            while True:
                self.env = MazeEnv(width=int(self.width_entry.get()), height=int(self.height_entry.get()))
                if self.env.start and self.env.goal and self.env.start != self.env.goal:
                    break

            # compute required state dim (5 metadata + width*height flattened wall map)
            required_state_dim = self.env.width * self.env.height + 5
            # Always ensure agent exists and uses required_state_dim before any train() calls.
            if (self.agent is None) or (self.agent.state_dim != required_state_dim):
                self.agent = DQNAgent(state_dim=required_state_dim, action_dim=4)
                # load will attempt exact or partial load but will NOT change state_dim
                self.agent.load()
                # schedule UI updates on main thread
                self._ui(self.canvas.delete, "all")
                self._ui(self.draw_static_maze)
                self.env.agent_pos = self.env.start
                self._ui(self.draw_agent)

                # notify UI that a new maze was generated / drawn
                # update stats label and optionally log to console widget if present
                # show initial stats for the new maze (no episode yet)
                init_white = self.env.white_distance if getattr(self.env, "white_distance", None) is not None else "N/A"
                self._ui(self.stats.config, {"text": f"Maze {maze_index+1}/{mazes} generated | WhiteDist: {init_white} | Eps: {self.agent.epsilon:.3f}"})
                if hasattr(self, "console") and self.console is not None:
                    self._ui(self.console.log, f"[INFO] Generated maze {maze_index+1}/{mazes} (white_dist={init_white})")
            
            solved = False
            for ep in range(episodes_per_maze):
                # allow graceful exit requested from UI
                self.env.reset()
                if self.stop_event.is_set():
                    return
                state = self.env._get_state()
                total_reward = 0.0
                for step in range((self.env.width * self.env.height)//2):
                    if self.stop_event.is_set():
                        return
                    action = self.agent.act(state)
                    next_state, reward, done = self.env.step(action)
                    self.agent.remember(state, action, reward, next_state, done)
                    self.agent.train()
                    state = next_state
                    total_reward += reward
                    # only update agent drawing (fast) on main thread
                    self._ui(self.draw_agent)
                    if done:
                        solved = True
                        break
                    time.sleep(0.01)  # slight delay to visualize agent movement
                self.agent.update_target()
                
                # update stats label on main thread
                # compute white distance display (fresh value)
                current_white = self.env.white_distance if getattr(self.env, "white_distance", None) is not None else self.env.compute_white_distance()
                white_disp = current_white if current_white is not None else "inf"
                # update stats label on main thread with full info
                self._ui(self.stats.config, {"text":
                    f"Maze {maze_index+1}/{mazes} | Ep {ep+1}/{episodes_per_maze} | Reward: {total_reward:.2f} | Eps: {self.agent.epsilon:.3f} | WhiteDist: {white_disp}"
                })
                

                if solved:
                    # save and show final path on UI thread
                    self.agent.save()
                    self._ui(self.show_path)
                    break
                if solved:
                    self.agent.save()
                    self.show_path()
                else:
                    print(f"[INFO] Maze {maze_index+1} skipped (unsolved)")
                self.agent.save()
        


            



    def show_path(self):
        path = self.env.get_path(self.agent)
        self.draw_static_maze(path)

      


if __name__ == "__main__":
    root = tk.Tk()
    app = MazeApp(root)
    root.mainloop()



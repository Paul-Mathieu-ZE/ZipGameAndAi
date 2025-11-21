from board_gen import *
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
        self.board = None
        self.start = None
        self.goal = None
        self.white_distance = None
        self.last_distance = None
        self.prev_move = (0, 0)
        self.reset(regenerate=True)

    def _generate_board(self):
        while True:
            b = Board(sizeX=self.height, sizeY=self.width)
            board = np.array(b.board, dtype=np.int8)
            s = np.argwhere(board == 2)
            g = np.argwhere(board == 3)
            if len(s) == 0 or len(g) == 0:
                continue
            start = tuple(s[0])
            goal = tuple(g[0])
            if start == goal:
                continue
            self.board = board
            self.start = start
            self.goal = goal
            break

    def _reset_agent_state(self):
        self.agent_pos = self.start
        self.prev_move = (0, 0)
        self.white_distance = self.compute_white_distance()
        max_dist = self.width * self.height
        self.last_distance = self.white_distance if self.white_distance is not None else max_dist

    def reset(self, regenerate=False):
        if regenerate or self.board is None:
            self._generate_board()
        self._reset_agent_state()
        return self._get_state()

    def step(self, action):
        dx, dy = self.actions[action]
        x, y = self.agent_pos
        nx, ny = x + dx, y + dy
        invalid_penalty = -0.2
        if not (0 <= nx < self.height and 0 <= ny < self.width):
            self.prev_move = (0, 0)
            return self._get_state(), invalid_penalty, False
        if int(self.board[nx, ny]) == 1:
            self.prev_move = (0, 0)
            return self._get_state(), invalid_penalty, False
        self.agent_pos = (nx, ny)
        self.prev_move = (dx, dy)
        previous = self.last_distance if self.last_distance is not None else (self.width * self.height)
        # update white_distance when agent moves (optional small cost to recompute is fine)
        self.white_distance = self.compute_white_distance()
        current = self.white_distance if self.white_distance is not None else (self.width * self.height)
        progress = previous - current
        # shaped reward encourages moving closer to goal, mild penalty for wandering
        reward = -0.02 + 0.1 * np.clip(progress, -1.0, 1.0)
        self.last_distance = current
        done = self.agent_pos == self.goal
        if done:
            reward = 5.0
        return self._get_state(), reward, done

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
        State = meta (agent/goal/prev move info) + flattened wall, agent, goal maps.
        """
        ax, ay = self.agent_pos
        gx, gy = self.goal

        shortest = self.white_distance if self.white_distance is not None else self.compute_white_distance()
        if shortest is None:
            shortest = self.width * self.height  # sentinel large
        norm_dist = float(shortest) / float(self.width * self.height)
        norm_last = float(self.last_distance if self.last_distance is not None else self.width * self.height) / float(self.width * self.height)
        prev_dx, prev_dy = self.prev_move

        meta = np.array([
            ax / float(self.height),
            ay / float(self.width),
            gx / float(self.height),
            gy / float(self.width),
            norm_dist,
            norm_last,
            prev_dx,
            prev_dy
        ], dtype=np.float32)

        wall_map = (self.board == 1).astype(np.float32).ravel()
        agent_map = np.zeros_like(self.board, dtype=np.float32)
        agent_map[ax, ay] = 1.0
        goal_map = np.zeros_like(self.board, dtype=np.float32)
        goal_map[gx, gy] = 1.0
        return np.concatenate([meta, wall_map, agent_map.ravel(), goal_map.ravel()]).astype(np.float32)

    @property
    def state_dim(self):
        # meta (8 values) + three flattened maps (wall/agent/goal)
        return int(8 + 3 * self.width * self.height)

    def get_valid_actions(self):
        valids = []
        x, y = self.agent_pos
        for dx, dy in self.actions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.height and 0 <= ny < self.width and int(self.board[nx, ny]) != 1:
                valids.append(True)
            else:
                valids.append(False)
        return np.array(valids, dtype=bool)


    def get_path(self, agent):
        self.agent_pos = self.start
        path = [self.start]
        for _ in range(500):
            state = self._get_state()
            action = agent.act(state, greedy=True, valid_actions=self.get_valid_actions())
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
        self.training_stats = {"mazes": 0, "solved": 0, "episodes": 0}
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
        required_state_dim = self.env.state_dim
        if (self.agent is None) or (self.agent.state_dim != required_state_dim):
            self.agent = DQNAgent(state_dim=required_state_dim, action_dim=len(self.env.actions))
            self.agent.load()
        else:
            self.agent.boost_exploration()
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

            self.training_stats["mazes"] += 1

            # compute required state dim (meta + stacked maps)
            required_state_dim = self.env.state_dim
            # Always ensure agent exists and uses required_state_dim before any train() calls.
            if (self.agent is None) or (self.agent.state_dim != required_state_dim):
                self.agent = DQNAgent(state_dim=required_state_dim, action_dim=len(self.env.actions))
                # load will attempt exact or partial load but will NOT change state_dim
                self.agent.load()
            else:
                self.agent.boost_exploration()

            # whenever a new maze is created (even if the dimensions match), redraw the UI
            self.env.agent_pos = self.env.start
            self._ui(self.canvas.delete, "all")
            self._ui(self.draw_static_maze)
            self._ui(self.draw_agent)

            # notify UI that a new maze was generated / drawn
            # update stats label and optionally log to console widget if present
            # show initial stats for the new maze (no episode yet)
            init_white = self.env.white_distance if getattr(self.env, "white_distance", None) is not None else "N/A"
            if hasattr(self, "console") and self.console is not None:
                self._ui(self.console.log, f"[INFO] Generated maze {maze_index+1}/{mazes} (white_dist={init_white})")
            
            solved = False
            episode_rewards = []
            for ep in range(episodes_per_maze):
                # allow graceful exit requested from UI
                self.env.reset()
                if self.stop_event.is_set():
                    return
                state = self.env._get_state()
                total_reward = 0.0
                estimated = self.env.white_distance if self.env.white_distance is not None else (self.env.width * self.env.height)
                max_steps = max(int(estimated * 3), self.env.width * self.env.height // 2, self.env.width + self.env.height)
                for step in range(max_steps):
                    if self.stop_event.is_set():
                        return
                    valid_actions = self.env.get_valid_actions()
                    action = self.agent.act(state, valid_actions=valid_actions)
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
                    self._ui(self.stats.config, {"text": f"Maze {maze_index+1}/{mazes} generated | WhiteDist: {init_white} | Eps: {self.agent.epsilon:.3f} | Reward: {total_reward:.2f}"})
                    time.sleep(0.01)  # slight delay to visualize agent movement
                episode_rewards.append(total_reward)
                self.training_stats["episodes"] += 1
                self.agent.update_target()
                
                # update stats label on main thread
                # compute white distance display (fresh value)
                current_white = self.env.white_distance if getattr(self.env, "white_distance", None) is not None else self.env.compute_white_distance()
                white_disp = current_white if current_white is not None else "inf"
                success_rate = self.training_stats["solved"] / max(1, self.training_stats["mazes"])
                # update stats label on main thread with full info
                self._ui(self.stats.config, {"text":
                    f"Maze {maze_index+1}/{mazes} | Ep {ep+1}/{episodes_per_maze} | Reward: {total_reward:.2f} | Eps: {self.agent.epsilon:.3f} | WhiteDist: {white_disp} | SR: {success_rate:.2f}"
                })
                
                if solved:
                    # save and show final path on UI thread
                    self.training_stats["solved"] += 1
                    self.agent.save()
                    self._ui(self.show_path)
                    break
            average_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
            result = "solved" if solved else "unsolved"
            summary_text = f"Maze {maze_index+1}/{mazes} {result} | avg reward {average_reward:.2f}"
            if hasattr(self, "console") and self.console is not None:
                self._ui(self.console.log, f"[INFO] {summary_text}")
            else:
                print(f"[INFO] {summary_text}")
            self.agent.save()
        


            



    def show_path(self):
        if not self.env or not self.agent:
            return
        path = self.env.get_path(self.agent)
        self.draw_path(path)

      


if __name__ == "__main__":
    root = tk.Tk()
    app = MazeApp(root)
    root.mainloop()



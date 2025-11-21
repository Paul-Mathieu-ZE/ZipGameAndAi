from board_gen import *
import pygame
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




class MazeGame:
    PANEL_HEIGHT = 120
    BG_COLOR = (18, 18, 22)
    GRID_COLOR = (200, 200, 200)

    def __init__(self):
        pygame.init()
        self.window_width = 900
        self.window_height = 900
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Maze AI Trainer (Pygame)")
        self.font = pygame.font.SysFont("consolas", 18)
        self.clock = pygame.time.Clock()

        self.width = 20
        self.height = 20
        self.env = MazeEnv(self.width, self.height)
        self.agent = DQNAgent(state_dim=self.env.state_dim, action_dim=len(self.env.actions))
        self.agent.load()

        self.training_stats = {"mazes": 0, "solved": 0, "episodes": 0}
        self.training_thread = None
        self.stop_event = threading.Event()
        self.running = True

        self.status_lock = threading.Lock()
        self.status_message = "G: generate | T: train | Arrows: resize | ESC: quit"

    def set_status(self, text):
        with self.status_lock:
            self.status_message = text

    def get_status(self):
        with self.status_lock:
            return self.status_message

    def run(self):
        try:
            while self.running:
                self.handle_events()
                self.draw()
                pygame.display.flip()
                self.clock.tick(60)
        finally:
            self.shutdown()

    def shutdown(self):
        self.stop_event.set()
        if self.training_thread and self.training_thread.is_alive():
            self.training_thread.join(timeout=1.0)
        pygame.quit()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_g:
                    self.generate_maze()
                elif event.key == pygame.K_t:
                    self.start_training_thread()
                elif event.key == pygame.K_UP:
                    self.adjust_height(1)
                elif event.key == pygame.K_DOWN:
                    self.adjust_height(-1)
                elif event.key == pygame.K_RIGHT:
                    self.adjust_width(1)
                elif event.key == pygame.K_LEFT:
                    self.adjust_width(-1)

    def adjust_width(self, delta):
        new_width = int(np.clip(self.width + delta, 5, 40))
        if new_width != self.width:
            self.width = new_width
            self.set_status(f"Width set to {self.width}. Press G to regenerate maze.")

    def adjust_height(self, delta):
        new_height = int(np.clip(self.height + delta, 5, 40))
        if new_height != self.height:
            self.height = new_height
            self.set_status(f"Height set to {self.height}. Press G to regenerate maze.")

    def draw(self):
        self.screen.fill(self.BG_COLOR)
        self.draw_panel()
        self.draw_maze()
        self.draw_agent()

    def draw_panel(self):
        pygame.draw.rect(self.screen, (25, 25, 30), (0, 0, self.window_width, 80))
        lines = [
            f"Size: {self.width}x{self.height}  Cells: {self.width * self.height}",
            f"Epsilon: {self.agent.epsilon:.3f} | Episodes: {self.training_stats['episodes']} | Mazes: {self.training_stats['mazes']} | Solved: {self.training_stats['solved']}",
            self.get_status()
        ]
        for idx, line in enumerate(lines):
            text = self.font.render(line, True, (230, 230, 230))
            self.screen.blit(text, (20, 10 + idx * 22))

    def draw_maze(self):
        if not self.env or self.env.board is None:
            return
        rows, cols = self.env.height, self.env.width
        maze_width = cols * CELL_SIZE
        maze_height = rows * CELL_SIZE
        offset_x = max((self.window_width - maze_width) // 2, 20)
        offset_y = 100

        colors = {
            0: (240, 240, 240),
            1: (20, 20, 20),
            2: (0, 160, 90),
            3: (200, 40, 40)
        }
        for i in range(rows):
            for j in range(cols):
                cell = int(self.env.board[i, j])
                rect = pygame.Rect(offset_x + j * CELL_SIZE, offset_y + i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(self.screen, colors.get(cell, (240, 240, 240)), rect)
                pygame.draw.rect(self.screen, self.GRID_COLOR, rect, 1)

    def draw_agent(self):
        if not self.env:
            return
        rows, cols = self.env.height, self.env.width
        maze_width = cols * CELL_SIZE
        offset_x = max((self.window_width - maze_width) // 2, 20)
        offset_y = 100
        ax, ay = self.env.agent_pos
        center_x = offset_x + ay * CELL_SIZE + CELL_SIZE // 2
        center_y = offset_y + ax * CELL_SIZE + CELL_SIZE // 2
        pygame.draw.circle(self.screen, (50, 120, 255), (center_x, center_y), CELL_SIZE // 3)

    def generate_maze(self):
        self.env = MazeEnv(self.width, self.height)
        required_state_dim = self.env.state_dim
        if (self.agent is None) or (self.agent.state_dim != required_state_dim):
            self.agent = DQNAgent(state_dim=required_state_dim, action_dim=len(self.env.actions))
            self.agent.load()
        else:
            self.agent.boost_exploration()
        self.set_status(f"Generated new maze {self.width}x{self.height}. Press T to train.")

    def start_training_thread(self):
        if self.training_thread and self.training_thread.is_alive():
            return
        self.stop_event.clear()
        self.training_thread = Thread(target=self.train_on_multiple_mazes, daemon=True)
        self.training_thread.start()
        self.set_status("Training started (background).")

    def train_on_multiple_mazes(self, mazes=100, episodes_per_maze=100):
        for maze_index in range(mazes):
            if self.stop_event.is_set():
                return
            while True:
                self.env = MazeEnv(self.width, self.height)
                if self.env.start and self.env.goal and self.env.start != self.env.goal:
                    break

            self.training_stats["mazes"] += 1
            required_state_dim = self.env.state_dim
            if (self.agent is None) or (self.agent.state_dim != required_state_dim):
                self.agent = DQNAgent(state_dim=required_state_dim, action_dim=len(self.env.actions))
                self.agent.load()
            else:
                self.agent.boost_exploration()

            self.env.agent_pos = self.env.start
            self.set_status(f"Maze {maze_index+1}/{mazes} generated. Training...")

            maze_solved_any = False
            solved_streak = 0
            episode_rewards = []
            best_reward = -float("inf")
            stagnation_counter = 0
            stagnation_patience = 15
            exit_reason = ""

            for ep in range(episodes_per_maze):
                if self.stop_event.is_set():
                    return
                self.env.reset()
                state = self.env._get_state()
                total_reward = 0.0
                estimated = self.env.white_distance if self.env.white_distance is not None else (self.env.width * self.env.height)
                max_steps = max(int(estimated * 3), self.env.width * self.env.height // 2, self.env.width + self.env.height)
                solved_this_episode = False
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
                    if done:
                        solved_this_episode = True
                        break
                    time.sleep(0.005)

                episode_rewards.append(total_reward)
                self.training_stats["episodes"] += 1
                self.agent.update_target()

                success_rate = self.training_stats["solved"] / max(1, self.training_stats["mazes"])
                rolling_reward = float(np.mean(episode_rewards[-5:])) if episode_rewards else 0.0
                self.set_status(
                    f"Maze {maze_index+1}/{mazes} | Ep {ep+1}/{episodes_per_maze} | Reward {total_reward:.2f} | Avg5 {rolling_reward:.2f} | SR {success_rate:.2f}"
                )

                if solved_this_episode:
                    solved_streak += 1
                    if not maze_solved_any:
                        maze_solved_any = True
                        self.training_stats["solved"] += 1
                    if solved_streak >= 5:
                        exit_reason = "5 consecutive solves"
                        break
                else:
                    solved_streak = 0

                if total_reward > best_reward + 0.1:
                    best_reward = total_reward
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
                    if stagnation_counter >= stagnation_patience:
                        exit_reason = f"plateau ({stagnation_patience} eps without gain)"
                        break

            average_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
            result = "solved" if maze_solved_any else "unsolved"
            if not exit_reason:
                exit_reason = "max episodes reached"
            summary_text = f"Maze {maze_index+1}/{mazes} {result} | avg reward {average_reward:.2f} | streak {solved_streak} | {exit_reason}"
            print(f"[INFO] {summary_text}")
            self.set_status(summary_text)
            self.agent.save()


if __name__ == "__main__":
    game = MazeGame()
    game.run()
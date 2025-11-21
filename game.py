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
    META_DIM = 8

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
        if progress < 0:
            reward = -0.02 * abs(progress)
        else:
            reward = 0.3*abs(progress)
        self.last_distance = current
        done = self.agent_pos == self.goal
        if done:
            reward = 10.0
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

        wall_map = (self.board == 1).astype(np.float32)
        agent_map = np.zeros_like(self.board, dtype=np.float32)
        agent_map[ax, ay] = 1.0
        goal_map = np.zeros_like(self.board, dtype=np.float32)
        goal_map[gx, gy] = 1.0
        stacked = np.stack([wall_map, agent_map, goal_map], axis=0).astype(np.float32)
        return np.concatenate([meta, stacked.ravel()]).astype(np.float32)

    @property
    def state_dim(self):
        return int(self.meta_dim + np.prod(self.grid_shape))

    @property
    def meta_dim(self):
        return self.META_DIM

    @property
    def grid_shape(self):
        return (3, int(self.height), int(self.width))

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
        self.agent = DQNAgent(
            state_dim=self.env.state_dim,
            action_dim=len(self.env.actions),
            grid_shape=self.env.grid_shape,
            meta_dim=self.env.meta_dim
        )
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

    def _create_env(self):
        while True:
            env = MazeEnv(self.width, self.height)
            if env.start and env.goal and env.start != env.goal:
                return env

    def generate_maze(self):
        self.env = self._create_env()
        required_state_dim = self.env.state_dim
        if (self.agent is None) or (self.agent.state_dim != required_state_dim):
            self.agent = DQNAgent(
                state_dim=required_state_dim,
                action_dim=len(self.env.actions),
                grid_shape=self.env.grid_shape,
                meta_dim=self.env.meta_dim
            )
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
        parallel_envs = max(2, min(8, (self.width * self.height) // 50))
        for maze_index in range(mazes):
            if self.stop_event.is_set():
                return
            envs = [self._create_env() for _ in range(parallel_envs)]
            states = [env._get_state() for env in envs]
            self.env = envs[0]

            self.training_stats["mazes"] += 1
            required_state_dim = self.env.state_dim
            if (self.agent is None) or (self.agent.state_dim != required_state_dim):
                self.agent = DQNAgent(
                    state_dim=required_state_dim,
                    action_dim=len(self.env.actions),
                    grid_shape=self.env.grid_shape,
                    meta_dim=self.env.meta_dim
                )
                self.agent.load()
            else:
                self.agent.boost_exploration()

            self.set_status(f"Maze {maze_index+1}/{mazes} generated | Parallel envs: {parallel_envs}")

            maze_solved_any = False
            per_env_streaks = [0] * parallel_envs
            per_env_rewards = [0.0] * parallel_envs
            per_env_lengths = [0] * parallel_envs
            episode_rewards = []
            best_reward = -float("inf")
            stagnation_counter = 0
            stagnation_patience = 25
            exit_reason = ""
            warmup_episodes = 3
            episodes_completed = 0
            max_steps = max(int(self.width * self.height * 1.5), self.width + self.height)

            while episodes_completed < episodes_per_maze:
                if self.stop_event.is_set():
                    return

                actions = []
                valid_masks = []
                for idx, env in enumerate(envs):
                    if per_env_lengths[idx] == 0 and episodes_completed < warmup_episodes:
                        self.agent.boost_exploration(0.5)
                    valid = env.get_valid_actions()
                    valid_masks.append(valid)
                    actions.append(self.agent.act(states[idx], valid_actions=valid))

                batch_transitions = []
                episodes_finished_this_iter = 0
                break_loop = False

                for idx, env in enumerate(envs):
                    curr_state = states[idx]
                    next_state, reward, done = env.step(actions[idx])
                    per_env_lengths[idx] += 1
                    per_env_rewards[idx] += reward
                    timeout = False
                    if per_env_lengths[idx] >= max_steps and not done:
                        done = True
                        timeout = True

                    batch_transitions.append((curr_state, actions[idx], reward, next_state, done))

                    if done:
                        episodes_completed += 1
                        episodes_finished_this_iter += 1
                        episode_total = per_env_rewards[idx]
                        episode_rewards.append(episode_total)

                        if not timeout and episode_total > 0:
                            per_env_streaks[idx] += 1
                            if not maze_solved_any:
                                maze_solved_any = True
                                self.training_stats["solved"] += 1
                        else:
                            per_env_streaks[idx] = 0

                        if per_env_streaks[idx] >= 5:
                            exit_reason = f"Env {idx+1} solved 5 in a row"
                            break_loop = True

                        if episode_total > best_reward + 0.1:
                            best_reward = episode_total
                            stagnation_counter = 0
                        else:
                            stagnation_counter += 1

                        env.reset(regenerate=False)
                        states[idx] = env._get_state()
                        per_env_rewards[idx] = 0.0
                        per_env_lengths[idx] = 0
                    else:
                        states[idx] = next_state

                for transition in batch_transitions:
                    self.agent.remember(*transition)

                self.training_stats["episodes"] += episodes_finished_this_iter
                self.agent.train()
                self.agent.update_target()

                success_rate = self.training_stats["solved"] / max(1, self.training_stats["mazes"])
                rolling_reward = float(np.mean(episode_rewards[-parallel_envs:])) if episode_rewards else 0.0
                self.set_status(
                    f"Maze {maze_index+1}/{mazes} | Episodes {episodes_completed}/{episodes_per_maze} | Avg {rolling_reward:.2f} | SR {success_rate:.2f}"
                )

                if break_loop:
                    break

                if stagnation_counter >= stagnation_patience:
                    exit_reason = f"plateau ({stagnation_patience} eps without gain)"
                    break

            average_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
            result = "solved" if maze_solved_any else "unsolved"
            if not exit_reason:
                exit_reason = "max episodes reached"
            max_streak = max(per_env_streaks) if per_env_streaks else 0
            summary_text = f"Maze {maze_index+1}/{mazes} {result} | avg reward {average_reward:.2f} | streak {max_streak} | {exit_reason}"
            print(f"[INFO] {summary_text}")
            self.set_status(summary_text)
            self.agent.save()


if __name__ == "__main__":
    game = MazeGame()
    game.run()
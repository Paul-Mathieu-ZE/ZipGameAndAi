import numpy as np, random, os
from collections import deque

class Board:
    _counter = 0
    _boards_dir = "boards"

    def __init__(self, sizeX=None, sizeY=None, board_data=None):
        self.start_pos = None
        self.goal_pos = None
        if board_data is not None:
            self.board = board_data
            self.sizeX, self.sizeY = board_data.shape
        elif sizeX and sizeY:
            self.sizeX, self.sizeY = sizeX, sizeY
            self.board = self.generate_maze()
        else: raise ValueError("Provide size or board_data.")
        if not os.path.exists(Board._boards_dir): os.makedirs(Board._boards_dir)

    def generate_maze(self):
        r, c = self.sizeX, self.sizeY
        maze = np.ones((r, c), dtype=int)

        def carve(x, y):
            maze[x, y] = 0
            directions = [(0, 2), (2, 0), (-2, 0), (0, -2)]
            random.shuffle(directions)
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 1 <= nx < r - 1 and 1 <= ny < c - 1 and maze[nx, ny] == 1:
                    maze[x + dx // 2, y + dy // 2] = 0
                    carve(nx, ny)

        carve(1, 1)
        return self._add_start_and_goal(maze)


    def _add_start_and_goal(self, board):
        r, c = board.shape
        def rand_open(x1, y1, x2, y2):
            for _ in range(1000):
                x, y = random.randint(x1, x2), random.randint(y1, y2)
                if board[x, y] == 0: return x, y
            return None
        s = rand_open(1, 1, r//3, c//3)
        g = rand_open(2*r//3, 2*c//3, r-2, c-2)
        if s and g:
            board[s[0], s[1]] = 2
            board[g[0], g[1]] = 3
            self.start_pos = s
            self.goal_pos = g
        return board


    def _save_board(self):
        path = os.path.join(Board._boards_dir, f"board_{self.index}.npy")
        np.save(path, self.board)

    @classmethod
    def load_board(cls, index):
        path = os.path.join(cls._boards_dir, f"board_{index}.npy")
        if not os.path.exists(path):
            return None
        board_data = np.load(path)
        b = cls(board_data=board_data)
        b.index = index

        # Restore start and goal positions
        start = np.argwhere(board_data == 2)
        goal = np.argwhere(board_data == 3)
        b.start_pos = tuple(start[0]) if len(start) > 0 else None
        b.goal_pos = tuple(goal[0]) if len(goal) > 0 else None

        return b


    @classmethod
    def list_boards(cls):
        if not os.path.exists(cls._boards_dir): return []
        return [(int(f.split("_")[1].split(".")[0]), np.load(os.path.join(cls._boards_dir,f)).shape)
                for f in sorted(os.listdir(cls._boards_dir)) if f.endswith(".npy")]

    def display(self):
        print(f"Board index: {self.index}")
        print(f"Start: {self.start_pos}, Goal: {self.goal_pos}")
        for row in self.board: print(" ".join(str(cell) for cell in row))

def board_generator(sizeX, sizeY, number):
    for _ in range(number):
        b = Board(sizeX, sizeY)
        b.index = Board._counter; Board._counter += 1
        b._save_board(); yield b


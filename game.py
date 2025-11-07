import board_gen 
from board_gen import *
import random
import tkinter as tk
from tkinter import ttk
import numpy as np
from board_gen import Board

CELL_SIZE = 20

class MazeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Générateur de Labyrinthe")

        self.player_pos = None
        self.controls = ttk.Frame(root)
        self.controls.pack(pady=10)
        
        self.current_board = None
        ttk.Label(self.controls, text="Largeur:").grid(row=0, column=0)
        self.width_entry = ttk.Entry(self.controls, width=5)
        self.width_entry.insert(0, "20")
        self.width_entry.grid(row=0, column=1)

        ttk.Label(self.controls, text="Hauteur:").grid(row=0, column=2)
        self.height_entry = ttk.Entry(self.controls, width=5)
        self.height_entry.insert(0, "20")
        self.height_entry.grid(row=0, column=3)

        self.generate_button = ttk.Button(self.controls, text="Générer", command=self.generate_maze)
        self.generate_button.grid(row=0, column=4, padx=10)
        
        self.save_button = ttk.Button(self.controls, text= "Save ", command=self.save_maze)
        self.save_button.grid(row=0,column= 5, padx=10)
        

        self.canvas = tk.Canvas(root, width=600, height=600, bg="white")
        self.canvas.pack()

    def generate_maze(self):
        try:
            w = int(self.width_entry.get())
            h = int(self.height_entry.get())
        except ValueError:
            print("Taille invalide.")
            return
        board = Board(sizeX=h, sizeY=w)
        
        self.current_board = board
        self.draw_board(board.board)
        start = np.argwhere(board == 2)
    def save_maze(self):
        if self.current_board:
            self.current_board.index = Board._counter
            Board._counter += 1
            self.current_board._save_board()
            print(f"Labyrinthe sauvegardé avec l'index {self.current_board.index}")
        else:
            print("Aucun labyrinthe à sauvegarder.")

    def draw_board(self, board):
        self.canvas.delete("all")
        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                x0 = j * CELL_SIZE
                y0 = i * CELL_SIZE
                x1 = x0 + CELL_SIZE
                y1 = y0 + CELL_SIZE

                val = board[i, j]
                color = {
                    1: "black",
                    0: "white",
                    2: "green",
                    3: "red"
                }.get(val, "gray")

                self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="gray")

if __name__ == "__main__":
    root = tk.Tk()
    app = MazeApp(root)
    root.mainloop()



class Position:
    def __init__(self,pos_x,pos_y):
        self.pos_x = pos_x
        self.pos_y = pos_y
    def __str__(self):
        return f"({self.pos_x}, {self.pos_y})"

        
        

def Game():
    b = Board.load_board(1)
    if not b:
        print("Board not found.")
        return
    b.display()
    ac = Position(b.start_pos[0], b.start_pos[1])
    gc = Position(b.goal_pos[0], b.goal_pos[1])
    print("Start position:", ac.pos_x, ac.pos_y)
    print("Goal position:", gc.pos_x, gc.pos_y)

    # Manhattan distance
    manhattan = abs(ac.pos_x - gc.pos_x) + abs(ac.pos_y - gc.pos_y)
    print("Manhattan distance:", manhattan)



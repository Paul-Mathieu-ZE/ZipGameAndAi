import board_gen 
from board_gen import *
import random

class Postition:
    def __init__(self,pos_x,pos_y):
        self.pos_x = pos_x
        self.pos_y = pos_y
        

def Game():
    b = Board.load_board(random.randrange(0,300))
    b.display()
    ac = Postition(b.start_pos[0],b.start_pos[1])
    print("Start position")
     
    
Game()
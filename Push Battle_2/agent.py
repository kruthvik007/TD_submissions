import random
from PushBattle import Game, PLAYER1, PLAYER2, EMPTY, BOARD_SIZE, NUM_PIECES, _torus
import numpy as np
'''
This is a sample implementation of an agent that just plays a random valid move every turn.
I would not recommend using this lol, but you are welcome to use functions and the structure of this.
'''

class AngryAgent:
    def __init__(self, player=PLAYER1):
        self.player = player
    
    # given the game state, gets all of the possible moves
    def get_possible_moves(self, game):
        """Returns list of all possible moves in current state."""
        moves = []
        current_pieces = game.p1_pieces if game.current_player == PLAYER1 else game.p2_pieces
        
        if current_pieces < NUM_PIECES:
            # placement moves
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    if game.board[r][c] == EMPTY:
                        moves.append((r, c))
        else:
            # movement moves
            for r0 in range(BOARD_SIZE):
                for c0 in range(BOARD_SIZE):
                    if game.board[r0][c0] == game.current_player:
                        for r1 in range(BOARD_SIZE):
                            for c1 in range(BOARD_SIZE):
                                if game.board[r1][c1] == EMPTY:
                                    moves.append((r0, c0, r1, c1))
        return moves
        
    def get_best_move(self, game):
        """Returns a random valid move."""
        possible_moves = self.get_possible_moves(game)
        
        
            # Create masks directly based on the game board
        opponent_P = (game.board == (-game.current_player)).astype(int)  # Mask for opponent positions
        player_P = (game.board == game.current_player).astype(int)       # Mask for player positions
        Good_Move = np.zeros((8, 8), dtype=float)

        
     
        
        for i in range(8):
            for j in range(8):
                if opponent_P[i][j] == 1:
                    
                    L = 0.0
                    R = 0.0
                    T = 0.0
                    D = 0.0
                    a = 0.0
                    
                    
                    if i == 0 | i == 7:
                        if j != 0 | j != 7:
                            R -= 1
                            L -= 1
                            
                        a += 1
                        
                    elif j == 0 | j == 7:
                        if i != 0 | i != 7:
                            T -= 1
                            D -= 1
                            
                        a += 1
                        
                    
                    # Check if each neighboring cell is within bounds before updating
                    if i - 1 >= 0 and j - 1 >= 0:  # Check top-left
                        Good_Move[i - 1][j - 1] += 1 + a
                    if i + 1 < 8 and j + 1 < 8:  # Check bottom-right
                        Good_Move[i + 1][j + 1] += 1 + a
                    if i + 1 < 8 and j - 1 >= 0:  # Check bottom-left
                        Good_Move[i + 1][j - 1] += 1 + a
                    if i - 1 >= 0 and j + 1 < 8:  # Check top-right
                        Good_Move[i - 1][j + 1] += 1 + a
                    if i + 1 < 8:  # Check bottom
                        Good_Move[i + 1][j]     += 1 + a + D
                    if i - 1 >= 0:  # Check top
                        Good_Move[i - 1][j]     += 1 + a + T
                    if j + 1 < 8:  # Check right
                        Good_Move[i][j + 1]     += 1 + a + R
                    if j - 1 >= 0:  # Check left
                        Good_Move[i][j - 1]     += 1 + a + L
                       
        
        Max = (0,0)
                    
        for i in range(8):
            for j in range(8):
                
                if opponent_P[i][j] == 1 | player_P[i][j] == 1:
                    Good_Move[i][j] = 0
                
                if Good_Move[i][j] > Good_Move[Max[0]][Max[1]]:
                    Max = (i,j)
            
        for M in possible_moves:
            if len(M) == 2:
                if M == Max:
                    return M
                    
            elif M[2] == Max[0] & M[3] == Max[1]:
                return M
                

            
import numpy as np

class TicTacToe:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros(9, dtype=int)  # 0=vazio, 1=IA, -1=humano
        self.current_player = 1
        return self.board

    def available_moves(self):
        return np.where(self.board == 0)[0]

    def step(self, action):
        self.board[action] = self.current_player
        reward, done = self.check_end()
        self.current_player *= -1
        return self.board.copy(), reward, done

    def check_end(self):
        b = self.board.reshape(3,3)
        lines = np.concatenate([b.sum(axis=0), b.sum(axis=1),
                                [b.trace(), np.fliplr(b).trace()]])
        if 3 in lines: return 1, True
        if -3 in lines: return -1, True
        if not self.available_moves().size: return 0, True
        return 0, False

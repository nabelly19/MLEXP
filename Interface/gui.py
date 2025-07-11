import tkinter as tk
from engine import TicTacToe
from Library.q_model import build_model

class TicTacToeGUI:
    def __init__(self):
        self.env = TicTacToe()
        self.model = build_model()
        self.model.load_weights('../Data/model_weights.h5')
        self.root = tk.Tk()
        self.buttons = []
        for i in range(9):
            btn = tk.Button(self.root, text='', width=6, height=3,
                            command=lambda i=i: self.human_move(i))
            btn.grid(row=i//3, column=i%3)
            self.buttons.append(btn)
        self.root.mainloop()

    def human_move(self, idx):
        if self.env.board[idx] != 0: return
        self.env.step(idx)
        self.update_buttons()
        state = self.env.board.copy()
        qs = self.model.predict(state.reshape(1,-1))[0]
        mask = (state != 0); qs[mask] = -np.inf
        ai_action = np.argmax(qs)
        self.env.step(ai_action)
        self.update_buttons()

    def update_buttons(self):
        for i,btn in enumerate(self.buttons):
            val = self.env.board[i]
            btn.config(text='X' if val== -1 else 'O' if val==1 else '')

if __name__ == "__main__":
    TicTacToeGUI()

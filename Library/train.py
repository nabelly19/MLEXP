import numpy as np
from engine import TicTacToe
from q_model import build_model
import os

def train(models_path='Data/model_weights.h5', episodes=5000, gamma=0.95, eps=1.0, eps_decay=0.995):
    env = TicTacToe()
    model = build_model()
    if os.path.exists(models_path):
        model.load_weights(models_path)

    for ep in range(episodes):
        state = env.reset()
        done = False
        while not done:
            if np.random.rand() < eps:
                action = np.random.choice(env.available_moves())
            else:
                qs = model.predict(state.reshape(1,-1))[0]
                mask = (state != 0)
                qs[mask] = -np.inf
                action = np.argmax(qs)
            next_state, reward, done = env.step(action)
            target = reward
            if not done:
                qnext = model.predict(next_state.reshape(1,-1))[0]
                target += gamma * np.max(qnext)
            qs_target = model.predict(state.reshape(1,-1))[0]
            qs_target[action] = target
            model.fit(state.reshape(1,-1), qs_target.reshape(1,-1), verbose=0)
            state = next_state

        eps *= eps_decay
        if ep % 100 == 0:
            model.save_weights(models_path)
            print(f"[Ep {ep}] eps={eps:.3f}")
    model.save_weights(models_path)

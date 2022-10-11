import time
import numpy as np
import torch as T
from NumPy2048 import TerminalGame
from dqn import Agent
from main_dqn_2048 import board_transform

def main():
    env = TerminalGame()
    agent = Agent(
        gamma=0.999,
        epsilon=0.05,
        lr=0.00001,
        input_dims=(1, env.height, env.width),
        batch_size=128,
        n_actions=4,
        max_mem_size=500000,
        eps_decay=1e-6,
        eps_end=0.05,
    )
    agent.Q_eval_target.load_state_dict(
        T.load('./saved_models/dqn_29000_games')
    )
    agent.Q_eval_target.eval()

    env.draw_game()
    observation = board_transform(env.board)
    done = False
    while not done:
        action = agent.choose_action(observation)
        observation, _, done = env.step(action)
        observation = board_transform(observation)
        time.sleep(0.2)
        env.draw_game()


if __name__ == '__main__':
    main()

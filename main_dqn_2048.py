import numpy as np
from NumPy2048 import CoreGame
from dqn import Agent
from utils import plot_learning


def main():
    env = CoreGame()
    agent = Agent(
        gamma=0.99,
        epsilon=1.0,
        lr=0.0001,
        input_dims=(env.height * env.width,),
        batch_size=64,
        n_actions=4,
        eps_decay=1e-5,
    )
    scores, eps_history = [], []
    n_games = 10000

    for i in range(n_games):
        done = False
        env.reset()
        observation = env.board
        observation = observation.flatten()
        cum_reward = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done = env.step(action)
            observation_ = observation_.flatten()
            agent.store_transition(
                observation, action, reward, observation_, done
            )
            cum_reward += reward
            agent.learn()
            observation = observation_
        scores.append(env.score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])
        print(
            f'''
            Episode: {i}, Score: {env.score}, Cumulative reward: {cum_reward},
            Average score: {avg_score}, Epsilon: {agent.epsilon}

            '''
        )
        print(
            f'Action sequence: {"".join([i[0] for i in env.action_history])}\n'
        )
        env.action_history = []

    x = [i + 1 for i in range(n_games)]
    filename = '2048.png'
    plot_learning(x, scores, eps_history, filename)


if __name__ == '__main__':
    main()

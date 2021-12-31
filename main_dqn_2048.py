import numpy as np
from NumPy2048 import CoreGame
from dqn import Agent
from utils import plot_learning


def main():
    env = CoreGame()
    agent = Agent(
        gamma=0.999,
        epsilon=1.0,
        lr=0.00001,
        input_dims=(env.height * env.width,),
        batch_size=128,
        n_actions=4,
        max_mem_size=500000,
        eps_decay=1e-6,
        eps_end=0.05,
    )
    scores, eps_history = [], []
    N_GAMES = 50000
    TARGET_UPDATE = 500

    for i_episode in range(N_GAMES):
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
            Episode: {i_episode}, Score: {env.score}, Cumulative reward: {cum_reward},
            Average score: {avg_score}, Epsilon: {agent.epsilon}

            '''
        )
        print(
            f'Action sequence: {"".join([i[0] for i in env.action_history])}\n'
        )
        env.action_history = []

        if i_episode % TARGET_UPDATE == 0:
            agent.Q_eval_target.load_state_dict(agent.Q_eval.state_dict())

            x = [i + 1 for i in range(i_episode + 1)]
            filename = '2048.png'
            plot_learning(x, scores, eps_history, filename)


if __name__ == '__main__':
    main()

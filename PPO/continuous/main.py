import gym
import numpy as np
import torch
from agent import Agent


# action adapter idea taken from:
# https://github.com/XinJingHao/PPO-Continuous-Pytorch
def action_adapter(a, max_a):
    return 2 * (a-0.5)*max_a


def clip_reward(x):
    if x < -1:
        return -1
    elif x > 1:
        return 1
    else:
        return x


if __name__ == '__main__':
    # env_id = 'BipedalWalker-v3'
    # env_id = 'LunarLanderContinuous-v2'
    env_id = 'Pendulum-v0'
    random_seed = 0
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    env = gym.make(env_id)
    eval_env = gym.make(env_id)
    env.seed(random_seed)
    eval_env.seed(random_seed)
    N = 2048
    batch_size = 64
    n_epochs = 10
    alpha = 0.0003
    max_action = env.action_space.high[0]
    agent = Agent(n_actions=env.action_space.shape[0], batch_size=batch_size,
                  alpha=alpha, n_epochs=n_epochs,
                  input_dims=env.observation_space.shape)

    score_history = []
    max_steps = 1_000_000
    total_steps = 0
    traj_length = 0
    episode = 1

    # for i in range(n_games):
    while total_steps < max_steps:
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action, prob = agent.choose_action(observation)
            act = action_adapter(action, max_action)
            observation_, reward, done, info = env.step(act)
            r = clip_reward(reward)
            total_steps += 1
            traj_length += 1
            score += reward
            agent.remember(observation, observation_, action,
                           prob, r, done)
            if traj_length % N == 0:
                agent.learn()
                traj_length = 0
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        print('{} Episode {} total steps {} avg score {:.1f}'.
              format(env_id, episode, total_steps, avg_score))
        episode += 1

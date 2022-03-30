import numpy as np
from agent import Agent
from vec_env import make_vec_envs


# action adapter idea taken from:
# https://github.com/XinJingHao/PPO-Continuous-Pytorch
def action_adapter(a, max_a):
    return 2 * (a-0.5)*max_a


def clip_reward(x):
    rewards = []
    for r in x:
        if r < -1:
            rewards.append(-1.0)
        elif r > 1:
            rewards.append(1.0)
        else:
            rewards.append(r)
    return rewards


if __name__ == '__main__':
    env_id = 'BipedalWalker-v3'
    # env_id = 'LunarLanderContinuous-v2'
    # env_id = 'Pendulum-v0'
    random_seed = 0
    n_procs = 16
    env = make_vec_envs(env_id, random_seed, n_procs)
    N = int(2048 // n_procs)
    batch_size = 64
    n_epochs = 10
    alpha = 0.0003
    max_action = env.action_space.high[0]
    n_actions = env.action_space.shape[0]
    agent = Agent(n_actions=n_actions, batch_size=batch_size,
                  alpha=alpha, n_epochs=n_epochs, T=N,
                  input_dims=env.observation_space.shape,
                  n_procs=n_procs)

    score_history = []
    max_steps = 1_000_000
    total_steps = 0
    traj_length = 0
    episode = 1
    while total_steps < max_steps:
        observation = env.reset()
        done = [False] * n_procs
        score = [0] * n_procs
        while not any(done):
            action, prob = agent.choose_action(observation)
            act = action_adapter(action, max_action).reshape(
                    (n_procs, n_actions))
            observation_, reward, done, info = env.step(act)
            r = clip_reward(reward)
            total_steps += 1
            traj_length += 1
            score += reward
            mask = [0.0 if d else 1.0 for d in done]
            agent.remember(observation, observation_,
                           action.reshape((n_procs, n_actions)),
                           prob.reshape((n_procs, n_actions)), r, mask)

            if traj_length % N == 0:
                # print('traj length', traj_length, 'n', N, 'mem cntr', agent.memory.mem_cntr)
                agent.learn()
                traj_length = 0
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        print('{} Episode {} total steps {} avg score {:.1f}'.
              format(env_id, episode, total_steps, avg_score))
        episode += 1
    env.close()

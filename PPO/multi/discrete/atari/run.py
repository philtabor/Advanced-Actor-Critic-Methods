import numpy as np
from agent import Agent
from vec_env import make_vec_envs


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
    env_id = 'PongNoFrameskip-v4'
    random_seed = 0
    n_procs = 8
    env = make_vec_envs(env_id, random_seed, n_procs)
    N = 128
    batch_size = 32
    n_epochs = 4
    alpha = 2.5E-4
    n_actions = env.action_space.n
    max_steps = 1e6
    agent = Agent(n_actions=n_actions, batch_size=batch_size,
                  alpha=alpha, n_epochs=n_epochs, T=N,
                  input_dims=env.observation_space.shape,
                  n_procs=n_procs, policy_clip=0.1, entropy_c=1e-2,
                  max_steps=max_steps, lr_decay=True)

    score_history = []
    total_steps = 0
    traj_length = 0
    episode = 1

    while total_steps < max_steps:
        observation = env.reset()
        done = [False] * n_procs
        score = [0] * n_procs
        while not any(done):
            action, prob, v = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            r = clip_reward(reward)
            total_steps += 1
            traj_length += 1
            score += reward
            mask = [0.0 if d else 1.0 for d in done]
            agent.remember(observation, observation_, action, prob, r, mask, v)
            if traj_length % N == 0:
                agent.learn()
                traj_length = 0
            observation = observation_
        score_history.append(score)
        score = np.mean(score)
        avg_score = np.mean(score_history[-100:])
        print('{} Episode {} total steps {} score {:.1f} '
              'avg score {:.1f}'.
              format(env_id, episode, total_steps, score, avg_score))
        episode += 1
    env.close()

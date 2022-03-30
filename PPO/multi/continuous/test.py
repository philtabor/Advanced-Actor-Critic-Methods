from vec_env import make_vec_envs

if __name__ == '__main__':
    seed = 0
    n_processes = 8
    env = make_vec_envs('Pendulum-v0', seed, n_processes)
    obs = env.reset()
    print(obs)
    env.close()
    print(env.reset())

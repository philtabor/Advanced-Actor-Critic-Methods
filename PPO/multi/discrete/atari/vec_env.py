from mpi4py import MPI
import multiprocessing as mp
import gym
import numpy as np
import torch as T
from wrappers import RepeatActionAndMaxFrame, PreprocessFrame


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError


class SubprocVecEnv:
    def __init__(self, env_fns, spaces=None):
        self.closed = False
        nenvs = len(env_fns)
        mp.set_start_method('forkserver')
        self.remotes, self.work_remotes = zip(*[mp.Pipe()
                                                for _ in range(nenvs)])
        self.ps = [mp.Process(target=worker, args=(work_remote, remote,
                              CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in
                   zip(self.work_remotes, self.remotes, env_fns)]

        for p in self.ps:
            p.daemon = True
            p.start()

        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        self.observation_space = observation_space
        self.action_space = action_space

    def step_async(self, actions):
        assert not self.closed, "trying to operate after calling close()"
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        assert not self.closed, "trying to operate after calling close()"
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close_extras(self):
        if self.closed:
            return
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    def close(self):
        if self.closed:
            return
        self.close_extras()
        self.closed = True

    def step(self, actions):
        obs, reward, dones, info = self.step_async(actions)
        return obs, reward, dones, info

    def __del__(self):
        if not self.closed:
            self.close()


class CloudpickleWrapper:
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


def make_env(env_id, seed, rank, repeat=4, clip_rewards=False,
             no_ops=0, fire_first=False, shape=(84, 84, 1)):
    def _thunk():
        env = gym.make(env_id)
        env = RepeatActionAndMaxFrame(env, repeat, clip_rewards,
                                      no_ops, fire_first)
        env = PreprocessFrame(shape, env)
        env.seed(seed + rank)
        return env
    return _thunk


def make_vec_envs(env_name, seed, num_processes):
    mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
    seed = seed + 10000 * mpi_rank if seed is not None else None
    set_global_seeds(seed)
    envs = [make_env(env_name, seed, i) for i in range(num_processes)]

    if len(envs) > 1:
        envs = SubprocVecEnv(envs)

    return envs


def set_global_seeds(seed):
    import random
    np.random.seed(seed)
    random.seed(seed)
    T.manual_seed(seed)
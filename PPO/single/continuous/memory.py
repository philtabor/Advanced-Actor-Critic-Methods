import numpy as np


class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.new_states = []

        self.batch_size = batch_size

    def recall(self):
        return np.array(self.states),\
            np.array(self.new_states),\
            np.array(self.actions),\
            np.array(self.probs),\
            np.array(self.rewards),\
            np.array(self.dones)

    def generate_batches(self):
        n_states = len(self.states)
        n_batches = int(n_states // self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i*self.batch_size:(i+1)*self.batch_size]
                   for i in range(n_batches)]
        return batches

    def store_memory(self, state, state_, action, probs, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.rewards.append(reward)
        self.dones.append(done)
        self.new_states.append(state_)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.new_states = []

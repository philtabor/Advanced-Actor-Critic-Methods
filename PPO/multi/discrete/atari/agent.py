import torch as T
import torch.nn.functional as F
from memory import PPOMemory
from networks import AtariActorCritic
from torch.distributions import Categorical


class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=3e-4, T=2048,
                 gae_lambda=0.95, policy_clip=0.2, batch_size=64, n_epochs=10,
                 n_procs=8, entropy_c=1e-3, learning_rate=2.5e-4, N=8,
                 max_steps=1e6, lr_decay=False):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.policy_clip_start = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.entropy_coefficient = entropy_c
        self.actor_critic = AtariActorCritic(n_actions, input_dims, alpha)
        self.memory = PPOMemory(batch_size, T=T, N=n_procs)
        self.input_dims = input_dims
        self.lr = learning_rate
        self.start_lr = learning_rate
        self.max_steps = max_steps
        self.T = T
        self.N = n_procs
        self.input_dims = input_dims
        self.lr_decay = lr_decay
        self.n_steps = 0

    def remember(self, state, state_, action, probs, reward, done, v):
        self.memory.store_memory(state, state_, action,
                                 probs, reward, done, v)

    def save_models(self):
        self.actor_critic.save_checkpoint()

    def load_models(self):
        self.actor_critic.load_checkpoint()

    def choose_action(self, observation):
        with T.no_grad():
            state = T.tensor(observation, dtype=T.float).to(
                    self.actor_critic.device)
            pi, v = self.actor_critic(state)
            dist = Categorical(probs=F.softmax(pi, dim=1))
            action = dist.sample()
            probs = dist.log_prob(action)
            action = [a for a in action.flatten().cpu().numpy()]
        return action, probs.flatten().cpu().numpy(), v.flatten().cpu().numpy()

    def calc_adv_and_returns(self, memories):
        states, new_states, r, dones = memories
        states = T.reshape(states, (self.T * self.N, *self.input_dims))
        new_states = T.reshape(new_states, (self.T * self.N, *self.input_dims))
        with T.no_grad():
            _, values = self.actor_critic(states)
            _, values_ = self.actor_critic(new_states)
            values = values.view(self.T, self.N)
            values_ = values_.view(self.T, self.N)
            deltas = r + self.gamma * values_ * dones - values
            deltas = deltas.cpu().numpy()
            dones = dones.cpu().numpy()
            adv = [0]
            for step in reversed(range(deltas.shape[0])):
                advantage = deltas[step] +\
                    self.gamma * self.gae_lambda * adv[-1] * dones[step]
                adv.append(advantage)
            adv.reverse()
            adv = adv[:-1]
            # shape is [128, 8]
            adv = T.tensor(adv).to(self.actor_critic.device)
            returns = adv + values
            adv = (adv - adv.mean()) / (adv.std()+1e-4)
        return adv, returns

    def update_params(self):
        if self.lr_decay:
            total_steps = self.max_steps / self.T
            frac = 1 - self.n_steps / total_steps
            self.lr = self.start_lr * frac
            self.policy_clip = self.policy_clip_start * frac
            for param_group in self.actor_critic.optimizer.param_groups:
                param_group['lr'] = self.lr

    def learn(self):
        state_arr, new_state_arr, action_arr, old_prob_arr,\
            reward_arr, dones_arr, values_arr = \
            self.memory.recall()
        state_arr = T.tensor(state_arr, dtype=T.float).to(
                self.actor_critic.device)
        action_arr = T.tensor(action_arr, dtype=T.float).to(
                self.actor_critic.device)
        old_prob_arr = T.tensor(old_prob_arr, dtype=T.float).to(
                self.actor_critic.device)
        new_state_arr = T.tensor(new_state_arr, dtype=T.float).to(
                self.actor_critic.device)
        r = T.tensor(reward_arr, dtype=T.float).to(
                self.actor_critic.device)
        old_values = T.tensor(values_arr, dtype=T.float).to(
                self.actor_critic.device)
        dones_arr = T.tensor(dones_arr, dtype=T.float).to(
                self.actor_critic.device)

        adv, returns = self.calc_adv_and_returns((state_arr, new_state_arr,
                                                 r, dones_arr))
        for epoch in range(self.n_epochs):
            batches = self.memory.generate_batches()
            for batch in batches:
                states = state_arr[batch].view(-1, *self.input_dims)
                old_probs = old_prob_arr[batch].view(-1)
                actions = action_arr[batch].view(-1)
                advantage = adv[batch].view(-1)
                rets = returns[batch].view(-1)
                pi, critic_value = self.actor_critic(states)
                old_critic_values = old_values[batch].view(-1)
                dist = Categorical(probs=F.softmax(pi, dim=1))
                new_probs = dist.log_prob(actions)
                prob_ratio = T.exp(new_probs - old_probs)
                weighted_probs = advantage * prob_ratio
                weighted_clipped_probs = T.clamp(
                        prob_ratio, 1-self.policy_clip, 1+self.policy_clip) * \
                    advantage
                entropy = dist.entropy()
                actor_loss = -T.min(weighted_probs,
                                    weighted_clipped_probs).mean()
                actor_loss -= self.entropy_coefficient * entropy.mean()
                critic_value = critic_value.view(-1)
                critic_value_clipped = old_critic_values +\
                    (critic_value - old_critic_values).clamp(
                        -self.policy_clip, self.policy_clip)
                critic_loss = (critic_value - rets).pow(2)
                critic_loss_clipped = (critic_value_clipped - rets).pow(2)
                critic_loss = 0.5 * T.max(critic_loss,
                                          critic_loss_clipped).mean()
                total_loss = actor_loss + 0.5 * critic_loss

                self.actor_critic.optimizer.zero_grad()
                total_loss.backward()
                T.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
                self.actor_critic.optimizer.step()
        self.n_steps += 1
        self.memory.clear_memory()
        self.update_params()

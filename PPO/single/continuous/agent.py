import torch as T
from memory import PPOMemory
from networks import ContinuousActorNetwork, ContinuousCriticNetwork


class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=3e-4,
                 gae_lambda=0.95, policy_clip=0.2, batch_size=64, n_epochs=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.entropy_coefficient = 1e-3
        self.actor = ContinuousActorNetwork(n_actions,
                                            input_dims, alpha)
        self.critic = ContinuousCriticNetwork(input_dims, alpha)
        self.memory = PPOMemory(batch_size)

    def remember(self, state, state_, action, probs, reward, done):
        self.memory.store_memory(state, state_, action,
                                 probs, reward, done)

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        with T.no_grad():
            state = T.tensor([observation], dtype=T.float).to(
                    self.actor.device)

            dist = self.actor(state)
            action = dist.sample()
            probs = dist.log_prob(action)

        return action.cpu().numpy().flatten(), probs.cpu().numpy().flatten()

    def calc_adv_and_returns(self, memories):
        states, new_states, r, dones = memories
        with T.no_grad():
            values = self.critic(states)
            values_ = self.critic(new_states)
            deltas = r + self.gamma * values_ - values
            deltas = deltas.cpu().flatten().numpy()
            adv = [0]
            for dlt, mask in zip(deltas[::-1], dones[::-1]):
                advantage = dlt + self.gamma * self.gae_lambda * adv[-1] * \
                            (1 - mask)
                adv.append(advantage)
            adv.reverse()
            adv = adv[:-1]
            adv = T.tensor(adv).float().unsqueeze(1).to(self.critic.device)
            # print('adv', adv)
            returns = adv + values
            # print('returns', returns)
            adv = (adv - adv.mean()) / (adv.std()+1e-4)
        return adv, returns

    def learn(self):
        state_arr, new_state_arr, action_arr, old_prob_arr,\
            reward_arr, dones_arr = \
            self.memory.recall()
        state_arr = T.tensor(state_arr, dtype=T.float).to(
                self.critic.device)
        action_arr = T.tensor(action_arr, dtype=T.float).to(
                self.critic.device)
        old_prob_arr = T.tensor(old_prob_arr, dtype=T.float).to(
                self.critic.device)
        new_state_arr = T.tensor(new_state_arr, dtype=T.float).to(
                self.critic.device)
        r = T.tensor(reward_arr, dtype=T.float).unsqueeze(1).to(
                self.critic.device)
        adv, returns = self.calc_adv_and_returns((state_arr, new_state_arr,
                                                 r, dones_arr))
        for epoch in range(self.n_epochs):
            batches = self.memory.generate_batches()
            for batch in batches:
                states = state_arr[batch]
                old_probs = old_prob_arr[batch]
                actions = action_arr[batch]

                dist = self.actor(states)
                new_probs = dist.log_prob(actions)
                prob_ratio = T.exp(new_probs.sum(1, keepdim=True) - old_probs.
                                   sum(1, keepdim=True))
                weighted_probs = adv[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(
                        prob_ratio, 1-self.policy_clip, 1+self.policy_clip) * \
                    adv[batch]
                entropy = dist.entropy().sum(1, keepdims=True)
                actor_loss = -T.min(weighted_probs,
                                    weighted_clipped_probs)
                actor_loss -= self.entropy_coefficient * entropy
                self.actor.optimizer.zero_grad()
                actor_loss.mean().backward()
                T.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
                self.actor.optimizer.step()

                critic_value = self.critic(states)
                critic_loss = (critic_value - returns[batch]).pow(2).mean()
                self.critic.optimizer.zero_grad()
                critic_loss.backward()
                self.critic.optimizer.step()
        self.memory.clear_memory()

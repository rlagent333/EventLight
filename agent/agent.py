import torch
import numpy as np


class Agent(object):
    def __init__(self, algorithm, config):
        self.alg = algorithm
        self.device = torch.device("cuda" if torch.cuda.
                                   is_available() else "cpu")
        self.config = config
        self.epsilon = self.config['epsilon']

    def sample(self, obs):
        # The epsilon-greedy action selector.
        def sample_random(act_dim):
            # Random samples
            return np.random.randint(0, act_dim)

        obs = torch.FloatTensor(obs).to(self.device)
        logits = self.alg.sample(obs)
        act_dim = logits.shape[-1]
        act_values = logits.cpu().detach().numpy()
        actions = np.argmax(act_values, axis=-1)
        for i in range(obs.shape[0]):
            if np.random.rand() <= self.epsilon:
                actions[i] = sample_random(act_dim)
        return actions

    def predict(self, obs):

        obs = torch.FloatTensor(obs).to(self.device)
        predict_actions = self.alg.predict(obs)
        return predict_actions

    def learn(self, obs, actions, dones, rewards, next_obs):

        obs = torch.FloatTensor(obs).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        next_obs = torch.FloatTensor(next_obs).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)

        Q_loss, pred_values, target_values, max_v_show_values, train_count, lr, epsilon, pred_loss, total_loss, mape = self.alg.learn(
            obs, actions, dones, rewards, next_obs)

        self.alg.sync_target(decay=self.config['decay'])
        self.epsilon = epsilon

        return Q_loss.item(), pred_values.item(), target_values.item(
        ), max_v_show_values.item(), train_count, lr, epsilon, pred_loss.item(), total_loss.item(), mape.item()

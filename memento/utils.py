from collections import namedtuple
import torch

Transition = namedtuple("Transition",
                        ("observation", "action", "reward", "value", "logProb"))


def discounted_cumsum(rewards, discount):

    """Calculates the cummulative sum of discounted rewards
    Arguments:
        rewards {torch.Tensor} -- rewards
        discount {float} -- discount factor
    Returns:
        [type] -- cummulative sum of discounted rewards
    """
    discount **= torch.arange(0, rewards.shape[0])
    disc_cumsum = torch.cumsum(discount * rewards, 0).flip()
    return disc_cumsum


def GAE_Lambda(rewards, values, discount):
    deltas = rewards[:-1] + discount * values[1:] - values[:-1]
    adv = discounted_cumsum(deltas, discount)
    return adv


class VPGBuffer:
    """Vanilla Policy Gradient Buffer
    """
    def __init__(self, obs_dim, act_dim, capacity, gamma=0.99, lam=0.95,
                 device=None, norm_advantage=True):
        obs_shape = (capacity,) + (obs_dim,)
        act_shape = (capacity,)
        if isinstance(act_dim, tuple):
            act_shape += act_dim
        self.buffer = dict(observation=torch.zeros(obs_shape,
                                                   dtype=torch.float32),
                           action=torch.zeros(act_shape, dtype=torch.float32),
                           advantage=torch.zeros(capacity,
                                                 dtype=torch.float32),
                           reward=torch.zeros(capacity, dtype=torch.float32),
                           value=torch.zeros(capacity, dtype=torch.float32),
                           logProb=torch.zeros(capacity, dtype=torch.float32))
        self.gamma = gamma
        self.Lambda = lam
        self.idx = 0
        self.start_idx = 0
        self.capacity = capacity
        self.norm_adv = True
        self.dev = device if device else torch.device("cpu")

    @property
    def shape(self):
        buffer_shape = {}
        for key, val in self.buffer.items():
            buffer_shape[key] = tuple(val.shape)
        return buffer_shape

    def __getitem__(self, item):
        if item == "advantage":
            if self.norm_adv:
                adv_mean, adv_std = torch.std_mean(self.buffer["advantage"])
                self.buffer["advantage"] -= adv_mean
                self.buffer["advantage"] /= adv_std + 1e-6
        return self.buffer["advantage"].to(self.dev)
        return self.buffer[item].to(self.dev)

    def push(self, transition):
        if self.idx < self.capacity:
            self.buffer["observation"][self.idx] = transition.observation.cpu()
            self.buffer["action"][self.idx] = transition.action.cpu()
            self.buffer["reward"][self.idx] = transition.reward
            self.buffer["value"][self.idx] = transition.value.cpu()
            self.buffer["logProb"][self.idx] = transition.logProb.cpu()
            self.idx += 1
        else:
            print("transitions buffer is full")

    def finish(self, last_val=0):
        path_slice = slice(self.start_idx, self.idx)
        rews = torch.cat((self.buffer["rew"][path_slice],
                          last_val.unsqueeze(0).cpu()), dim=0)
        vals = torch.cat((self.buffer["val"][path_slice],
                          last_val.unsqueeze(0).cpu()), dim=0)

        # GAE-Lambda calculation
        self.buffer["adv"][path_slice] = GAE_Lambda(rews, vals,
                                                    self.gamma * self.Lambda)

        # rewards-to-go
        self.buffer["ret"][path_slice] = discounted_cumsum(rews,
                                                           self.gamma)[:-1]

        self.start_idx = self.idx


if __name__ == "__main__":
    agent_buffer = VPGBuffer(128, 1, 1000)
    for _ in range(100):
        rand_1 = torch.zeros(128)
        rand_2 = torch.zeros(1)
        rand_3 = 1.
        transition = Transition(observation=rand_1, action=rand_2,
                                reward=rand_3, value=rand_2, logProb=rand_1)
        agent_buffer.push(transition)
        obs = agent_buffer["observation"]
        adv = agent_buffer["advantage"]

import torch 
import torch.nn as nn
from torch.distributions import Categorical

"""
Heavily follows AND borrows from openai's spinup package
see https://github.com/openai/spinningup
"""


class MLP(nn.Module):
    """Multi-layer Perceptron
    """

    def __init__(self, obs_dim, act_dim, activation=nn.ReLU):
        """[summary]
        
        Arguments:
            obs_dim {[type]} -- [description]
            act_dim {[type]} -- [description]
        
        Keyword Arguments:
            activation {[type]} -- [description] (default: {nn.ReLU})
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, act_dim)
        self.activate = activation()

    def forward(self, x):
        x = self.activate(self.fc1(x))
        x = self.activate(self.fc2(x))
        return x


class Actor(nn.Module):  
    """Base Actor Class
    """

    def distribution(self, obs):
        raise NotImplementedError

    def log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        pi = self.distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self.log_prob_from_distribution(pi, act)
        return pi, logp_a


class CategoricalActor(Actor):
    """Actor with a categorical (stochastic) policy
    """

    def __init__(self, net):
        super(CategoricalActor, self).__init__()
        self.net = net

    def distribution(self, obs):
        logits = self.net(obs)
        return Categorical(logits=logits)

    def log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class Critic(nn.Module):
    """Base Critic Class
    """

    def __init__(self, v_net):
        super(Critic, self).__init__()
        self.net = v_net

    def forward(self, obs):
        return torch.squeeze(self.net(obs), -1)


class ActorCritic(nn.Module):
    """Actor Critic Class, acts as a container for an actor and critic
    """

    def __init__(self, obs_dim, act_dim, actor_type, pi_net=None, v_net=None):
        super(ActorCritic, self).__init__()

        # policy function
        self.pi_net = pi_net if pi_net else MLP(obs_dim, act_dim, 
                                                activation=nn.ReLU)
        if actor_type == 'categorical':
            self.actor = CategoricalActor(self.pi_net)
        else:
            raise NotImplementedError

        # value function
        self.v_net = v_net if v_net else MLP(obs_dim, 1, activation=nn.Tanh)
        self.critic = Critic(v_net=self.v_net)

    def forward(self, obs):
        with torch.no_grad():
            pi = self.actor.distribution(obs)
            a = pi.sample()
            logp_a = self.actor.log_prob_from_distribution(pi, a)
            v = self.critic(obs)
        return a, v, logp_a

    def act(self, obs):
        return self.forward(obs)[0]
        
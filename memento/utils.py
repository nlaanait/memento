import time
from collections import namedtuple

import torch
from gym.spaces import Box, Discrete
from torch.optim import Adam

from .agents import ActorCritic
from .logging import VPGLogger
import gym
import os

"""
Heavily follows AND freely borrows from openai's spinningup
see https://github.com/openai/spinningup
"""
#pylint: disable=no-member

Transition = namedtuple("Transition",
                        ("observation", "action", "reward", "value",
                         "logProb"))


def discounted_cumsum(rewards, discount):

    """Calculates the cummulative sum of discounted rewards

    Arguments:
        rewards {torch.Tensor} -- rewards
        discount {float} -- discount factor
        
    Returns:
        [type] -- cummulative sum of discounted rewards
    """
    discount **= torch.arange(0, rewards.shape[0])
    disc_cumsum = torch.cumsum(discount * rewards, 0).flip(0)
    return disc_cumsum


def GAE_Lambda(rewards, values, discount):
    """General Advantage Estimation
    
    Arguments:
        `rewards` {torch.Tensor} -- [description]
        values {torch.Tensor} -- [description]
        discount {float} -- [description]
    
    Returns:
        [type] -- [description]
    """
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
        self.idx = self.start_idx = 0
        if item == "advantage":
            if self.norm_adv:
                adv_mean, adv_std = torch.std_mean(self.buffer["advantage"])
                self.buffer["advantage"] -= adv_mean
                self.buffer["advantage"] /= adv_std + 1e-6
            return self.buffer["advantage"].to(self.dev)
        return self.buffer[item].to(self.dev)

    def push(self, transition):
        assert self.idx < self.capacity, "Buffer is full"
        self.buffer["observation"][self.idx] = transition.observation.cpu()
        self.buffer["action"][self.idx] = transition.action.cpu()
        self.buffer["reward"][self.idx] = transition.reward
        self.buffer["value"][self.idx] = transition.value.cpu()
        self.buffer["logProb"][self.idx] = transition.logProb.cpu()
        self.idx += 1

    def finish(self, last_val=0):
        path_slice = slice(self.start_idx, self.idx)
        rewards = torch.cat((self.buffer["reward"][path_slice],
                             last_val.unsqueeze(0).cpu()), dim=0)
        vals = torch.cat((self.buffer["value"][path_slice],
                          last_val.unsqueeze(0).cpu()), dim=0)

        # GAE-Lambda calculation
        self.buffer["advantage"][path_slice] = GAE_Lambda(rewards, vals,
                                                         self.gamma * self.Lambda)

        # rewards-to-go
        self.buffer["reward"][path_slice] = discounted_cumsum(rewards,
                                                           self.gamma)[:-1]

        self.start_idx = self.idx


def vpg_train(env_func, device=None, actor_critic=ActorCritic, ac_kwargs=dict(), 
              buf_kwargs=dict(), steps_per_epoch=4000, epochs=50, pi_lr=3e-4,
              vf_lr=1e-3, train_v_iters=80, max_ep_len=1000,
              logger_kwargs=dict(), save_freq=10):
    """
    Vanilla Policy Gradient 

    (with GAE-Lambda for advantage estimation)

    Args:
        env_func : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with a 
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v`` 
            module. The ``step`` method should accept a batch of observations 
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================

            The ``act`` method behaves the same as ``step`` but only returns ``a``.

            The ``pi`` module's forward call should accept a batch of 
            observations and optionally a batch of actions, and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing 
                                           | the log probability, according to 
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================

            The ``v`` module's forward call should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical: 
                                           | make sure to flatten this!)
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to VPG.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """
    # setup tensor device
    device = device if device is not None else torch.device("cpu")

    # instantiate env
    env = env_func()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # Set up logger and save configuration
    logger = VPGLogger(**logger_kwargs)
    logger.save_config(locals())
    logger.log("Environment dimensions: observation={}, action={}"
               .format(obs_dim, act_dim))

    # istantiate actor-critic
    actor_type = None
    if isinstance(env.action_space, Discrete):
        actor_type = 'categorical'
    elif isinstance(env.action_space, Box):
        raise NotImplementedError
    
    ac = actor_critic(obs_dim, act_dim, actor_type, **ac_kwargs)
    ac = ac.to(device)
    
    # save models
    logger.setup_pytorch_saver(ac)

    # setup buffer
    buf = VPGBuffer(obs_dim, act_dim, steps_per_epoch, device=device,
                    **buf_kwargs)
    logger.log("Buffer shapes: {}".format(buf.shape))

    # functions to compute policy and value losses

    def compute_loss_pi(buf):
        # actor (policy)
        obs, act, adv, logp_old = [buf[itm] for itm in ['observation',
                                                        'action',
                                                        'advantage',
                                                        'logProb']]
        pi, logp = ac.actor(obs, act)
        loss_pi = -(logp * adv).mean()

        kl = (logp_old - logp).mean()
        ent = pi.entropy().mean()
        pi_info = dict(kl=kl, ent=ent)
        return loss_pi, pi_info

    def compute_loss_v(buf):
        # critic (value)
        obs, ret = buf['observation'], buf['reward']
        loss_val = ((ac.critic(obs) - ret) ** 2).mean()
        return loss_val

    # optimizers
    actor_optim = Adam(ac.actor.parameters(), lr=pi_lr)
    critic_optim = Adam(ac.critic.parameters(), lr=vf_lr)

    def update(buf):
        pi_l_last, pi_info_last = compute_loss_pi(buf)
        v_l_last = compute_loss_v(buf)
        
        # single step of policy gradient descent
        actor_optim.zero_grad()
        loss_pi, pi_info = compute_loss_pi(buf)
        loss_pi.backward()
        actor_optim.step()

        # value function training
        for _ in range(train_v_iters):
            critic_optim.zero_grad()
            loss_v = compute_loss_v(buf)
            loss_v.backward()
            critic_optim.step()

        # log changes from update
        kl, ent = pi_info['kl'], pi_info_last['ent']
        logger.store(LossPi=pi_l_last, LossV=v_l_last,
                     KL=kl, Entropy=ent,
                     DeltaLossPi=(loss_pi.item() - pi_l_last),
                     DeltaLossV=(loss_v.item() - v_l_last))

    # prepare environement
    start_time = time.time()
    obs, ep_ret, ep_len = env.reset(), 0, 0
    obs = torch.as_tensor(obs, dtype=torch.float32).to(device)

    # collect experience and update
    for epoch in range(epochs):
        for step in range(steps_per_epoch):
            a, v, logp = ac(obs)
            next_obs, reward, done, _ = env.step(a.cpu().numpy())
            ep_ret += reward
            ep_len += 1

            # save
            transition = Transition(observation=obs, action=a,
                                    reward=reward, value=v,
                                    logProb=logp)
            buf.push(transition)
            logger.store(VVals=v)

            # update obs
            obs = torch.as_tensor(next_obs, dtype=torch.float32).to(device)

            timeout = ep_len == max_ep_len
            episode_ended = done or timeout
            epoch_ended = step == steps_per_epoch - 1

            if episode_ended or epoch_ended:
                if epoch_ended and not(episode_ended):
                    logger.log('Warning: trajectory cut off by epoch at %d steps.'
                          % ep_len, flush=True)
                if episode_ended or epoch_ended:
                    _, v, _ = ac(obs)
                else:
                    v = 0
                buf.finish(v)
                if episode_ended:
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                obs, ep_ret, ep_len = env.reset(), 0 , 0
                obs = torch.as_tensor(obs, dtype=torch.float32).to(device)
        
        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, None)

        # VPG update
        update(buf)

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()


if __name__ == "__main__":
    # running the buffer
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
    
    # training an actor-critic policy gradient on Atari
    env_args = "BeamRider-ram-v0"
    env_func = lambda : gym.make(env_args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger_kwargs = dict(output_dir=os.path.join('/tmp',env_args+"_vpg"))
    vpg_train(env_func, device=device, logger_kwargs=logger_kwargs, epochs=250)


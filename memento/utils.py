import os
import time
from collections import namedtuple

import gym
import numpy as np
import scipy
import torch
from gym.spaces import Box, Discrete
from torch.optim import Adam

from .agents import ActorCritic
from .logging import VPGLogger

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

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    x = x.numpy()
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def GAE_Lambda(rewards, values, discount, lam):
    """General Advantage Estimation
    
    Arguments:
        `rewards` {torch.Tensor} -- [description]
        values {torch.Tensor} -- [description]
        discount {float} -- [description]
    
    Returns:
        [type] -- [description]
    """
    deltas = rewards[:-1] + discount * values[1:] - values[:-1]
    adv = discounted_cumsum(deltas, discount * lam)
    return adv


class VPGBuffer:
    """Vanilla Policy Gradient Buffer
    """
    def __init__(self, obs_dim, act_dim, capacity, gamma=0.99, lam=0.95,
                 device=None, norm_advantage=True):
        if not isinstance(obs_dim, tuple):
            obs_dim = (obs_dim, ) 
        obs_shape = (capacity,) + obs_dim
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

    @property
    def device(self):
        return self.dev

    @device.setter
    def device(self, dev):
        if isinstance(dev, torch.device):
            self.dev = dev
        else:
            self.dev = torch.device(dev)

    def __getitem__(self, item):
        self.idx , self.start_idx = 0 , 0
        if item == "advantage":
            if self.norm_adv:
                adv_mean, adv_std = torch.std_mean(self.buffer["advantage"])
                self.buffer["advantage"] -= adv_mean
                self.buffer["advantage"] /= adv_std + 1e-6
            return self.buffer["advantage"].to(self.dev)
        return self.buffer[item].to(self.dev)

    def push(self, transition):
        if self.idx < self.capacity: # only store if buffer is not full 
            self.buffer["observation"][self.idx] = transition.observation.cpu()
            self.buffer["action"][self.idx] = transition.action.cpu()
            self.buffer["reward"][self.idx] = transition.reward
            self.buffer["value"][self.idx] = transition.value.cpu()
            self.buffer["logProb"][self.idx] = transition.logProb.cpu()
            self.idx += 1

    def finish(self, last_val=0):
        if last_val.dim() == 0:
            last_val = last_val.unsqueeze(0)
        path_slice = slice(self.start_idx, self.idx)
        # path_slice_adv = slice(self.start_idx, self.idx-1)
        rewards = torch.cat((self.buffer["reward"][path_slice],
                             last_val.cpu()), dim=0)
        vals = torch.cat((self.buffer["value"][path_slice],
                          last_val.cpu()), dim=0)

        # GAE-Lambda calculation
        deltas = rewards[:-1] + self.gamma * vals[1:] - vals[:-1]
        print(deltas.shape, path_slice, self.buffer['advantage'][path_slice].shape)
        if self.buffer['advantage'][path_slice].shape != deltas.shape:
            path_slice_adv = slice(path_slice.start, path_slice.stop - 1)
        else:
            path_slice_adv = path_slice
        if self.buffer["advantage"][path_slice_adv].size()[0] < 2:
            self.start_idx = self.idx
            return
        # print('deltas shape', deltas.shape, 'advantage shape', self.buffer["advantage"][path_slice].shape)
        # self.buffer["advantage"][path_slice_adv] = GAE_Lambda(rewards, vals,
        #                                                  self.gamma, self.Lambda)
        self.buffer["advantage"][path_slice_adv] = torch.from_numpy(np.ascontiguousarray(
                                                                discount_cumsum(deltas, self.gamma * self.Lambda)))
        # rewards-to-go
        self.buffer["reward"][path_slice_adv] = torch.from_numpy(np.ascontiguousarray(discount_cumsum(rewards,
                                                           self.gamma)[:-1]))
        # self.buffer["reward"][path_slice_adv] = discounted_cumsum(rewards, self.gamma)[:-1]

        self.start_idx = self.idx


def adjust_obs(obs, device=None, crop=[slice(40, None), slice(None, None)]):
    if len(obs.shape) > 1:
        obs = obs.transpose(2,0,1)
        obs = obs[:,crop[0], crop[1]]
        pad_y = (obs.shape[1] % 4) 
        pad_x = (obs.shape[2] % 4)
        obs = np.pad(obs, ((0,0),(0,pad_y), (0, pad_x)))
        obs = np.ascontiguousarray(obs, dtype=np.float32) / 255
    if device is not None:
        obs = torch.as_tensor(obs, dtype=torch.float32).to(device)
    return obs
    

def adjust_obs_atari(obs, device=None):
    obs_arr = np.array(obs)
    obs_arr = np.ascontiguousarray(obs_arr.transpose(2,0,1), dtype=np.float32) / 255
    if device is not None:
        obs = torch.from_numpy(obs_arr).to(device)
        return obs
    return obs_arr


def get_obs_act_dims(env, adjust_obs=adjust_obs):
    obs_shape = env.observation_space.shape
    if len(obs_shape) > 1:
        obs = env.reset()
        obs = adjust_obs(obs)
        obs_shape = obs.shape
    act_shape = env.action_space.n
    return obs_shape, act_shape


def get_buffer_chunks(buffer_size, chunk_size):
    chunks = []
    num_chunks = buffer_size // chunk_size
    for itm in range(num_chunks):
        if itm == num_chunks - 1:
            partition = slice(itm * chunk_size, None)
        else:
            partition = slice(itm * chunk_size, (itm + 1) * chunk_size)
        chunks.append(partition)
    return chunks


def vpg_train(env_func, device=None, actor_critic=ActorCritic, ac_kwargs=dict(), 
              adjust_obs=adjust_obs, buf_kwargs=dict(), steps_per_epoch=4000, epochs=50, 
              pi_lr=3e-4, vf_lr=1e-3, train_v_iters=80, max_ep_len=1000, render=False,
              logger_kwargs=dict(), save_freq=10, batch_size=256):
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
    obs_shape, act_shape = get_obs_act_dims(env, adjust_obs=adjust_obs)

    # Set up logger and save configuration
    logger = VPGLogger(**logger_kwargs)
    logger.save_config(locals())
    logger.log("Environment dimensions: observation={}, action={}"
               .format(obs_shape, act_shape))

    # istantiate actor-critic
    if actor_critic is None:
        actor_type = None
        if isinstance(env.action_space, Discrete):
            actor_type = 'categorical'
        elif isinstance(env.action_space, Box):
            raise NotImplementedError
        ac = actor_critic(obs_shape, act_shape, actor_type, **ac_kwargs)
    else:
        ac = actor_critic
        ac = ac.to(device)
    
    # save models
    logger.setup_pytorch_saver(ac)

    # setup buffer
    try:
        capacity = buf_kwargs.pop('capacity')
    except KeyError:
        capacity = steps_per_epoch
    buf = VPGBuffer(obs_shape, act_shape, capacity, device=device,
                    **buf_kwargs)
    logger.log("Buffer shapes: {}".format(buf.shape))

    # functions to compute policy and value losses

    def compute_loss_pi(obs, act, adv, logp_old):
        pi, logp = ac.actor(obs, act)
        loss_pi = -(logp * adv).mean() 
        kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        pi_info = dict(kl=kl, ent=ent)
        return loss_pi, pi_info

    def compute_loss_v(obs, ret):
        # critic (value)
        pred_ret = ac.critic(obs)
        loss_val = ((pred_ret - ret) ** 2).mean()
        return loss_val

    # optimizers
    actor_optim = Adam(ac.actor.parameters(), lr=pi_lr)
    critic_optim = Adam(ac.critic.parameters(), lr=vf_lr)

    def update(buf, batch=64, pi_l_last=0, v_l_last=0):
        # do updates in batches
        if batch is not None:
            buf_parts =  get_buffer_chunks(buf.idx, batch)
            buf.device = "cpu"
        else:
            buf_parts = [slice(None, None)]
        loss_pi_val = loss_v_val = 0
        for part in buf_parts:
            obs, act, adv, logp_old, ret = [buf[itm][part] for itm in ["observation",
                                                                       "action",
                                                                       "advantage",
                                                                       "logProb",
                                                                       "reward"]]
            obs = obs.to(device)
            act = act.to(device)
            adv = adv.to(device)
            logp_old = logp_old.to(device)
            ret = ret.to(device)
        
            # single step of policy gradient descent
            actor_optim.zero_grad()
            loss_pi, pi_info = compute_loss_pi(obs, act, adv, logp_old)
            loss_pi.backward()
            actor_optim.step()

            # value function training
            for _ in range(train_v_iters):
                critic_optim.zero_grad()
                loss_v = compute_loss_v(obs, ret)
                loss_v.backward()
                critic_optim.step()
            
        
            loss_pi_val += loss_pi.item()
            loss_v_val += loss_v.item()


        # log changes from update
        kl, ent = pi_info['kl'], pi_info['ent']
        logger.store(LossPi=loss_pi_val, LossV=loss_v_val,
                     KL=kl, Entropy=ent,
                     DeltaLossPi=(loss_pi_val - pi_l_last),
                     DeltaLossV=(loss_v_val - v_l_last))
        buf.device = device
        return loss_pi_val, loss_v_val

    # prepare environement
    start_time = time.time()
    obs, ep_ret, ep_len = env.reset(), 0, 0
    obs = adjust_obs(obs, device)

    pi_l_last = 0
    v_l_last = 0
    last_obs = torch.zeros_like(obs)
    # collect experience and update
    for epoch in range(epochs):
        env.reset()
        for step in range(steps_per_epoch):
            if render:
                env.render()
            a, v, logp = ac(obs - last_obs)
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
            # last_obs = obs
            obs = adjust_obs(next_obs, device) 

            timeout = ep_len == max_ep_len
            terminal = done or timeout
            epoch_ended = step == steps_per_epoch - 1

            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    logger.log('Warning: trajectory cut off by epoch at %d steps.'
                          % ep_len)
                if timeout or epoch_ended:
                    print('timed out or epoch ended')
                    _, v, _ = ac(obs)
                else:
                    v = torch.Tensor(0)
                buf.finish(v)
                if terminal:
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                obs, ep_ret, ep_len = env.reset(), 0 , 0
                obs = adjust_obs(obs, device)
        
        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, None)

        # VPG update
        pi_l_last, v_l_last = update(buf, batch=batch_size, pi_l_last=pi_l_last, v_l_last=v_l_last)

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

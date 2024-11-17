import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Union

import gymnasium as gym
import numpy as np
import torch
from tianshou.data import (
    Batch,
    ReplayBuffer,
    ReplayBufferManager,
    VectorReplayBuffer,
    to_numpy,
)
from tianshou.env import BaseVectorEnv, DummyVectorEnv

from fsrl.policy import BasePolicy
from src.models.risk_models import *
from src.datasets.risk_datasets import *
from src.utils import * 
import os


class FastCollector(object):
    """Collector enables the policy to interact with different types of envs with \
    exact number of episodes.

    This collector is a simplified version of Tianshou's `collector
    <https://tianshou.readthedocs.io/en/master/api/tianshou.data.html#collector>`_, so
    it is safe to check their documentation for details. The main change is the \
        support to extract the cost signals from the interaction data.

    :param policy: an instance of the :class:`~fsrl.policy.BasePolicy` class.
    :param env: a ``gym.Env`` environment or an instance of the
        :class:`~tianshou.env.BaseVectorEnv` class.
    :param buffer: an instance of the :class:`~tianshou.data.ReplayBuffer` class. If set
        to None, it will not store the data. Default to None.
    :param function preprocess_fn: a function called before the data has been added to
        the buffer. Default to None.
    :param bool exploration_noise: determine whether the action needs to be modified with
        corresponding policy's exploration noise. If so, "policy. exploration_noise(act,
        batch)" will be called automatically to add the exploration noise into action.
        Default to False.

    .. note::

        Please make sure the given environment has a time limitation (can be done), \
            because we only support the `n_episode` collect option.
    """

    def __init__(
        self,
        args,
        policy: BasePolicy,
        env: Union[gym.Env, BaseVectorEnv],
        buffer: Optional[ReplayBuffer] = None,
        preprocess_fn: Optional[Callable[..., Batch]] = None,
        exploration_noise: bool = False,
        risk_model: Optional[BayesRiskEst] = None,
    ) -> None:
        super().__init__()
        if isinstance(env, gym.Env) and not hasattr(env, "__len__"):
            warnings.warn("Single environment detected, wrap to DummyVectorEnv.")
            self.env = DummyVectorEnv([lambda: env])
        else:
            self.env = env
        
        self.args = args
        self.env_num = len(self.env)
        self.exploration_noise = exploration_noise
        self._assign_buffer(buffer)
        self.policy = policy
        self.preprocess_fn = preprocess_fn
        self._action_space = self.env.action_space
        self.obs_space = self.env.observation_space

        # avoid creating attribute outside __init__
        self.reset(False)

        # Keep count of the number of steps and episodes collected
        self.global_step = 0
        # initializing the risk loss 
        self.risk_loss = None
        self.device = torch.device(args.device)

        # Setup the risk stuff
        if args.use_risk and risk_model is not None:
            self.risk_model = risk_model
            if os.path.exists(args.risk_model_path):
                self.risk_model.load_state_dict(torch.load(args.risk_model_path, map_location=torch.device(args.device)))

            self.risk_model.to(torch.device(args.device))
            
            # if you want to train the risk model
            if args.fine_tune_risk:
                self.initialize_risk_utils()
                # setup the optimizer
                self.opt_risk = torch.optim.Adam(self.risk_model.parameters(), lr=args.risk_lr, eps=1e-10)
                # risk criterion 
                self.risk_criterion = torch.nn.NLLLoss()
                # initalizing risk loss 
            
            self.risk_model.eval()


    def initialize_risk_utils(self):
        # Initializing the risk replay buffer
        self.risk_rb = Batch(
            next_obs = None,
            risks = None, 
            risk_quant= None,
        )
        self.risk_bins = np.array([i*self.args.quantile_size for i in range(self.args.quantile_num+1)])
        # Set number of update steps to 1 in case of sync typpe update
        self.args.risk_update_steps = 1 if self.args.risk_update_type == "sync" else self.args.risk_update_steps
        

    def _assign_buffer(self, buffer: Optional[ReplayBuffer]) -> None:
        """Check if the buffer matches the constraint."""
        if buffer is None:
            buffer = VectorReplayBuffer(self.env_num, self.env_num)
        elif isinstance(buffer, ReplayBufferManager):
            assert buffer.buffer_num >= self.env_num
        else:  # ReplayBuffer or PrioritizedReplayBuffer
            assert buffer.maxsize > 0
            if self.env_num > 1:
                if isinstance(buffer) == ReplayBuffer:
                    buffer_type = "ReplayBuffer"
                    vector_type = "VectorReplayBuffer"
                else:
                    buffer_type = "PrioritizedReplayBuffer"
                    vector_type = "PrioritizedVectorReplayBuffer"
                raise TypeError(
                    f"Cannot use {buffer_type}(size={buffer.maxsize}, ...) to collect "
                    f"{self.env_num} envs,\n\tplease use {vector_type}(total_size="
                    f"{buffer.maxsize}, buffer_num={self.env_num}, ...) instead."
                )
        self.buffer = buffer

    def reset(
        self,
        reset_buffer: bool = True,
        gym_reset_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Reset the environment, statistics, current data and possibly replay memory.

        :param bool reset_buffer: if true, reset the replay buffer that is attached to
            the collector.
        :param gym_reset_kwargs: extra keyword arguments to pass into the environment's
            reset function. Defaults to None (extra keyword arguments)
        """
        # use empty Batch for "state" so that self.data supports slicing convert empty
        # Batch to None when passing data to policy
        self.data = Batch(
            obs={},
            act={},
            rew={},
            cost={},
            terminated={},
            truncated={},
            done={},
            obs_next={},
            info={},
            policy={}
        )
        self.risk_data = Batch(
            obs_next=None,
            cost=None,
        )



        self.reset_env(gym_reset_kwargs)
        if reset_buffer:
            self.reset_buffer()
        self.reset_stat()

    def reset_stat(self) -> None:
        """Reset the statistic variables."""
        self.collect_step, self.collect_episode, self.collect_time = 0, 0, 0.0

    def reset_buffer(self, keep_statistics: bool = False) -> None:
        """Reset the data buffer."""
        self.buffer.reset(keep_statistics=keep_statistics)

    def reset_env(self, gym_reset_kwargs: Optional[Dict[str, Any]] = None) -> None:
        """Reset all of the environments."""
        gym_reset_kwargs = gym_reset_kwargs if gym_reset_kwargs else {}
        rval = self.env.reset(**gym_reset_kwargs)
        returns_info = isinstance(rval, (tuple, list)) and len(rval) == 2 and (
            isinstance(rval[1], dict) or isinstance(rval[1][0], dict)
        )
        if returns_info:
            obs, info = rval
            if self.preprocess_fn:
                processed_data = self.preprocess_fn(
                    obs=obs, info=info, env_id=np.arange(self.env_num)
                )
                obs = processed_data.get("obs", obs)
                info = processed_data.get("info", info)
            self.data.info = info
        else:
            obs = rval
            if self.preprocess_fn:
                obs = self.preprocess_fn(obs=obs,
                                         env_id=np.arange(self.env_num)).get("obs", obs)
        self.data.obs = obs

    def _reset_state(self, id: Union[int, List[int]]) -> None:
        """Reset the hidden state: self.data.state[id]."""
        if hasattr(self.data.policy, "hidden_state"):
            state = self.data.policy.hidden_state  # it is a reference
            if isinstance(state, torch.Tensor):
                state[id].zero_()
            elif isinstance(state, np.ndarray):
                state[id] = None if state.dtype == object else 0
            elif isinstance(state, Batch):
                state.empty_(id)

    def _reset_env_with_ids(
        self,
        local_ids: Union[List[int], np.ndarray],
        global_ids: Union[List[int], np.ndarray],
        gym_reset_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        gym_reset_kwargs = gym_reset_kwargs if gym_reset_kwargs else {}
        rval = self.env.reset(global_ids, **gym_reset_kwargs)
        returns_info = isinstance(rval, (tuple, list)) and len(rval) == 2 and (
            isinstance(rval[1], dict) or isinstance(rval[1][0], dict)
        )
        if returns_info:
            obs_reset, info = rval
            if self.preprocess_fn:
                processed_data = self.preprocess_fn(
                    obs=obs_reset, info=info, env_id=global_ids
                )
                obs_reset = processed_data.get("obs", obs_reset)
                info = processed_data.get("info", info)
            self.data.info[local_ids] = info
        else:
            obs_reset = rval
            if self.preprocess_fn:
                obs_reset = self.preprocess_fn(obs=obs_reset,
                                               env_id=global_ids).get("obs", obs_reset)
        self.data.obs_next[local_ids] = obs_reset
        
        if self.args.use_risk and self.args.fine_tune_risk:
            self.add_risk_data()
            self.risk_data = Batch(
                obs_next=None,
                cost=None,
            )

    def reset_risk_rb(self):
        # resize the replay buffer if its too large 
        if self.args.use_risk and self.args.fine_tune_risk:
            if self.risk_rb.next_obs is not None and self.risk_rb.next_obs.shape[0] > self.args.risk_buffer_size:
                self.risk_rb.next_obs = self.risk_rb.next_obs[-self.args.risk_buffer_size:]
                self.risk_rb.risks = self.risk_rb.risks[-self.args.risk_buffer_size:]
                self.risk_rb.risk_quant = self.risk_rb.risk_quant[-self.args.risk_buffer_size:]
    
    def add_risk_data(self):
        # resize replay buffer if needed
        self.reset_risk_rb()
        costs = torch.Tensor(self.risk_data.cost)
        f_risks = torch.empty_like(costs)
        for i in range(self.env_num):
            f_risks[:, i] = compute_fear(costs[:, i])

        f_risks = f_risks.view(-1, 1)
        f_risks_quant = torch.Tensor(np.apply_along_axis(lambda x: np.histogram(x, bins=self.risk_bins)[0], 1, np.expand_dims(f_risks.cpu().numpy(), 1)))
        f_next_obs = torch.Tensor(self.risk_data.obs_next).reshape(-1, self.obs_space[0].shape[0])
        self.risk_rb.update(
            next_obs=f_next_obs if self.risk_rb.next_obs is None else torch.cat([self.risk_rb.next_obs, f_next_obs], axis=0),
            risks=f_risks_quant if self.risk_rb.risks is None else torch.cat([self.risk_rb.risks, f_risks_quant], axis=0),
            risk_quant=f_risks if self.risk_rb.risk_quant is None else torch.cat([self.risk_rb.risk_quant, f_risks], axis=0),
        )

    def update_risk_model(self):
        if self.args.use_risk and self.args.fine_tune_risk:
            if self.risk_rb.next_obs is not None \
                                 and self.global_step % self.args.risk_update_freq == 0:
                loss = None
                with tqdm.tqdm(total=self.args.risk_update_steps, desc="Update Risk model") as pbar:
                    for _ in range(self.args.risk_update_steps):
                        risk_inds = np.random.choice(range(self.risk_rb.next_obs.size(0)), self.args.risk_batch_size)
                        pred = self.risk_model(self.risk_rb.next_obs[risk_inds,:].to(self.device))
                        risk_loss = self.risk_criterion(pred, torch.argmax(self.risk_rb.risks[risk_inds,:].squeeze(), axis=1).to(self.device))
                        self.opt_risk.zero_grad()
                        risk_loss.backward()
                        self.opt_risk.step()
                        loss = risk_loss if loss is None else loss + risk_loss
                        pbar.update(1)
                self.risk_loss = loss / self.args.risk_update_steps
                # logger.store(**{"risk/risk_loss": risk_loss.item()})
            

    def collect(
        self,
        n_episode: int = 1,
        random: bool = False,
        render: bool = False,
        no_grad: bool = True,
        gym_reset_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Collect a specified number of step or episode.

        To ensure unbiased sampling result with n_episode option, this function will
        first collect ``n_episode - env_num`` episodes, then for the last ``env_num``
        episodes, they will be collected evenly from each env.

        :param int n_episode: how many episodes you want to collect.
        :param bool random: whether to use random policy for collecting data. Default to
            False.
        :param bool render: Whether to render the environment during evaluation, defaults
            to False
        :param bool no_grad: whether to retain gradient in policy.forward(). Default to
            True (no gradient retaining).
        :param gym_reset_kwargs: extra keyword arguments to pass into the environment's
            reset function. Defaults to None (extra keyword arguments)

        .. note::

            We don not support the `n_step` collection method in Tianshou, because using
            `n_episode` only can facilitate the episodic cost computation and better
            evaluate the agent.

        :return: A dict including the following keys

            * ``n/ep`` collected number of episodes.
            * ``n/st`` collected number of steps.
            * ``rew`` mean of episodic rewards.
            * ``len`` mean of episodic lengths.
            * ``total_cost`` cumulative costs in this collect.
            * ``cost`` mean of episodic costs.
            * ``truncated`` mean of episodic truncation.
            * ``terminated`` mean of episodic termination.
        """
        if n_episode is not None:
            assert n_episode > 0
            ready_env_ids = np.arange(min(self.env_num, n_episode))
            self.data = self.data[:min(self.env_num, n_episode)]
        else:
            raise TypeError("Please specify n_episode"
                            "in FastCollector.collect().")

        start_time = time.time()

        step_count = 0
        total_cost = 0
        termination_count = 0
        truncation_count = 0
        episode_count = 0
        episode_rews = []
        episode_lens = []
        episode_start_indices = []

        while True:
            assert len(self.data) == len(ready_env_ids)
            # restore the state: if the last state is None, it won't store
            last_state = self.data.policy.pop("hidden_state", None)

            # get the next action
            if random:
                try:
                    act_sample = [self._action_space[i].sample() for i in ready_env_ids]
                except TypeError:  # envpool's action space is not for per-env
                    act_sample = [self._action_space.sample() for _ in ready_env_ids]
                act_sample = self.policy.map_action_inverse(act_sample)  # type: ignore
                self.data.update(act=act_sample)
            else:
                if no_grad:
                    with torch.no_grad():  # faster than retain_grad version
                        # self.data.obs will be used by agent to get result
                        result = self.policy(self.data, last_state)
                else:
                    result = self.policy(self.data, last_state)
                # update state / act / policy into self.data
                policy = result.get("policy", Batch())
                assert isinstance(policy, Batch)
                state = result.get("state", None)
                if state is not None:
                    policy.hidden_state = state  # save state into buffer
                act = to_numpy(result.act)
                if self.exploration_noise:
                    act = self.policy.exploration_noise(act, self.data)
                self.data.update(policy=policy, act=act)

            # get bounded and remapped actions first (not saved into buffer)
            action_remap = self.policy.map_action(self.data.act)
            # step in env
            result = self.env.step(action_remap, ready_env_ids)
            
            # update number of steps taken in the environment
            self.global_step += len(ready_env_ids)

            # Update the risk model if its there 
            self.update_risk_model()

            if len(result) == 5:
                obs_next, rew, terminated, truncated, info = result
                done = np.logical_or(terminated, truncated)
            elif len(result) == 4:
                obs_next, rew, done, info = result
                if isinstance(info, dict):
                    truncated = info["TimeLimit.truncated"]
                else:
                    truncated = np.array(
                        [
                            info_item.get("TimeLimit.truncated", False)
                            for info_item in info
                        ]
                    )
                terminated = np.logical_and(done, ~truncated)
            else:
                raise ValueError()

            self.data.update(
                obs_next=obs_next,
                rew=rew,
                terminated=terminated,
                truncated=truncated,
                done=done,
                info=info
            )

            if self.preprocess_fn:
                self.data.update(
                    self.preprocess_fn(
                        obs_next=self.data.obs_next,
                        rew=self.data.rew,
                        done=self.data.done,
                        info=self.data.info,
                        policy=self.data.policy,
                        env_id=ready_env_ids,
                    )
                )

            cost = self.data.info.get("cost", np.zeros(rew.shape))
            total_cost += np.sum(cost)
            self.data.update(cost=cost)
            self.risk_data.update(
                obs_next=np.array([obs_next]) if self.risk_data.obs_next is None else np.concatenate([self.risk_data.obs_next, np.array([obs_next])], axis=0),
                cost=np.array([cost]) if self.risk_data.cost is None else np.concatenate([self.risk_data.cost, np.array([cost])], axis=0),
            )
            if render:
                self.env.render()

            # add data into the buffer
            ptr, ep_rew, ep_len, ep_idx = self.buffer.add(
                self.data, buffer_ids=ready_env_ids
            )

            # collect statistics
            step_count += len(ready_env_ids)

            if np.any(done):
                env_ind_local = np.where(done)[0]
                env_ind_global = ready_env_ids[env_ind_local]
                episode_count += len(env_ind_local)
                episode_lens.append(ep_len[env_ind_local])
                episode_rews.append(ep_rew[env_ind_local])
                episode_start_indices.append(ep_idx[env_ind_local])
                termination_count += np.sum(terminated)
                truncation_count += np.sum(truncated)
                # now we copy obs_next to obs, but since there might be finished
                # episodes, we have to reset finished envs first.
                self._reset_env_with_ids(env_ind_local, env_ind_global, gym_reset_kwargs)
                for i in env_ind_local:
                    self._reset_state(i)

                # remove surplus env id from ready_env_ids to avoid bias in selecting
                # environments
                if n_episode:
                    surplus_env_num = len(ready_env_ids) - (n_episode - episode_count)
                    if surplus_env_num > 0:
                        mask = np.ones_like(ready_env_ids, dtype=bool)
                        mask[env_ind_local[:surplus_env_num]] = False
                        ready_env_ids = ready_env_ids[mask]
                        self.data = self.data[mask]

            self.data.obs = self.data.obs_next

            if n_episode and episode_count >= n_episode:
                break

        # generate statistics
        self.collect_step += step_count
        self.collect_episode += episode_count
        self.collect_time += max(time.time() - start_time, 1e-9)

        if n_episode:
            self.data = Batch(
                obs={},
                act={},
                rew={},
                cost={},
                terminated={},
                truncated={},
                done={},
                obs_next={},
                info={},
                policy={}
            )
            self.reset_env()

        if episode_count > 0:
            rews, lens = list(map(np.concatenate, [episode_rews, episode_lens]))
            rew_mean = rews.mean()
            len_mean = lens.mean()
        else:
            rew_mean = len_mean = 0

        done_count = termination_count + truncation_count

        return {
            "n/ep": episode_count,
            "n/st": step_count,
            "rew": rew_mean,
            "len": len_mean,
            "total_cost": total_cost,
            "cost": total_cost / episode_count,
            "truncated": truncation_count / done_count,
            "terminated": termination_count / done_count,
            "risk_loss": None if self.risk_loss is None else self.risk_loss.item()
        }

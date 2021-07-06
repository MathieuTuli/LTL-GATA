from typing import Optional, List, Tuple

import logging
import pdb

from gutils import FixedSizeList

import numpy as np

from segment_tree import MinSegmentTree, SumSegmentTree
from utils import LinearSchedule
from components import Sample

logger = logging.getLogger()


class PrioritizedExperienceReplay:
    def __init__(self,
                 beta: float,
                 batch_size: int,
                 multi_step: int,
                 max_episode: int,
                 seed: int = None,
                 alpha: float = 0.,
                 capacity: int = 100_000,
                 discount_gamma_game_reward: float = 1.,
                 accumulate_reward_from_final: bool = False,
                 recurrent_memory: bool = False,
                 sample_update_from: int = 0,
                 sample_history_length: int = 1) -> None:
        self._max_priority = 1.
        self.multi_step = multi_step
        self.batch_size = batch_size
        self.recurrent_memory = recurrent_memory
        self.sample_update_from = sample_update_from
        self.sample_history_length = sample_history_length
        self.beta_scheduler = LinearSchedule(
            schedule_timesteps=max_episode,
            initial_p=beta, final_p=1.0)

        self._buffer = FixedSizeList(capacity=capacity)
        self._rng = np.random.RandomState(seed)

        self.capacity = capacity
        it_capacity = 2 ** int(np.ceil(np.log2(capacity)))
        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        assert np.greater_equal(alpha, 0.)
        self._alpha = alpha
        self._discount_gamma_game_reward = discount_gamma_game_reward
        self._accumulate_reward_from_final = accumulate_reward_from_final

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def at_capacity(self) -> bool:
        return len(self._buffer) == self.capacity

    @property
    def buffer(self):
        return self._buffer

    def update_priorities(self, indices: List[int],
                          priorities: List[float]) -> bool:
        for idx, priority in zip(indices, priorities):
            if np.greater(priority, 0.):
                assert 0 <= idx < len(self)
                self._it_sum[idx] = priority ** self._alpha
                self._it_min[idx] = priority ** self._alpha
                self._max_priority = max(self._max_priority, priority)
            else:
                logger.error(f"Something wrong with priority: {priority}")
                return False
        return True

    # TODO improve efficiency
    def avg_rewards(self):
        if len(self) == 0:
            return 0.
        return np.mean([sample.reward for sample in self._buffer
                        if sample is not None])

    def add(self, sample: Sample) -> None:
        self._buffer.append(sample)
        idx = len(self) - 1
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def get_next_final_idx(self, idx: int) -> Optional[int]:
        for i, sample in enumerate(self._buffer[idx:]):
            if sample.is_final:
                return i + idx
        return None

    def _sample_proportional(self) -> List[int]:
        return [self._it_sum.find_prefixsum_idx(
            self._rng.random() * self._it_sum.sum(
                0, len(self) - 1)) for _ in range(self.batch_size)]

    def get_samples_and_stepped(self, idx: int, n: int,
                                recurrent_memory: bool) -> List[Sample]:
        assert n > 0
        # if n == 1:
        #     if self._buffer[idx].is_final:
        #         return tuple([None for _ in range(3)])
        # else:
        #     if np.any([item.is_final for item in self._buffer[idx: idx + n]]):
        #         return tuple([None for _ in range(3)])

        next_final_idx = self.get_next_final_idx(idx)
        if next_final_idx is None or idx + n > next_final_idx:
            # n = idx - next_final_idx
            return tuple([None for _ in range(3)])
        samples, stepped_samples, rewards = list(), list(), list()
        iteration_count = 1
        if recurrent_memory:
            iteration_count = n
            n = 1
        for j in range(iteration_count):
            # n + 1 or just n?
            length = next_final_idx - (idx + j) + 1 if \
                self._accumulate_reward_from_final else n if not \
                recurrent_memory else 1
            sample = self._buffer[idx + j]
            stepped_sample = self._buffer[idx + n + j]
            _rewards = [self._discount_gamma_game_reward ** i *
                        self._buffer[idx + j + i].reward for
                        i in range(length)]
            reward = np.sum(_rewards)
            samples.append(sample)
            stepped_samples.append(stepped_sample)
            rewards.append(reward)
        return samples, stepped_samples, rewards

    def get_samples(self, episode_no: int,
                    recurrent_memory: bool = False
                    ) -> Tuple[List[Sample],
                               List[Sample],
                               List[float],
                               List[Tuple[int, int]],
                               List[float]]:
        logger.debug("Getting samples from ER")
        if len(self) < self.batch_size:
            return tuple([None for _ in range(5)])
        beta = self.beta_scheduler.value(episode_no)
        assert np.greater(beta, 0.)
        idxs = self._sample_proportional()
        ns = self._rng.randint(1, self.multi_step + 1, size=self.batch_size)
        all_samples, all_stepped_samples, all_rewards, weights = \
            [[list() for _ in range(self.sample_history_length if
                                    self.recurrent_memory else 1)]
             for i in range(4)]
        indices = list()
        for idx, n in zip(idxs, ns):
            samples, stepped_samples, rewards = \
                self.get_samples_and_stepped(
                    idx, self.sample_history_length if self.recurrent_memory
                    else n,
                    recurrent_memory=self.recurrent_memory)
            if samples is None:
                continue
            if self.recurrent_memory:
                indices.append((idx, self.sample_history_length))
            else:
                indices.append((idx, n))
            for step in range(self.sample_history_length if
                              self.recurrent_memory else 1):
                all_rewards[step].append(rewards[step])
                all_samples[step].append(samples[step])
                all_stepped_samples[step].append(stepped_samples[step])

        if len(indices) == 0:
            return tuple([None for _ in range(5)])
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self)) ** (-beta)
        for step in range(self.sample_history_length if
                          self.recurrent_memory else 1):
            for (idx, n) in indices:
                p_sample = self._it_sum[idx + step] / self._it_sum.sum()
                weight = (p_sample * len(self)) ** (-beta)
                weights[step].append(weight / max_weight)
        return all_samples, all_stepped_samples, all_rewards, indices, weights

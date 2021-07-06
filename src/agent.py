from typing import Tuple, List, Dict, Any
from argparse import Namespace
from pathlib import Path

import logging
import copy
import json
import pdb

import torch.nn.functional as F
import numpy as np
import torch

from textworld import EnvInfos

from experience_replay import PrioritizedExperienceReplay
from components import AgentModes, Actions, Vocabulary
from graph_updater import GraphUpdater
from belief_graph import BeliefGraph
from utils import LinearSchedule
from optim import get_optimizer
from state import BatchedStates
from model import PolicyNet
from ltl import LTL

logger = logging.getLogger()


class Agent:
    def __init__(self, config: Namespace, word_vocab: Vocabulary,
                 ltl_vocab: Vocabulary, relation_vocab: Vocabulary,
                 node_vocab: Vocabulary,
                 action_vocab: Vocabulary,
                 pretrain: bool = False) -> None:
        self.ltl = config.ltl
        self.training = config.training
        self.evaluate = config.evaluate
        self.test_config = config.test
        with (config.io.data_dir / 'uuid_mapping.json').open('r') as f:
            self.uuid_mapping = json.load(f)
        self.set_random_seed(self.training.random_seed)
        self._states: BatchedStates = None
        self._eval_states: BatchedStates = None
        self._word_vocab = word_vocab
        self._ltl_vocab = ltl_vocab
        self._relation_vocab = relation_vocab
        self._node_vocab = node_vocab
        self._action_vocab = action_vocab
        self.graph_updater = GraphUpdater(
            checkpoint=config.graph_updater.checkpoint,
            vocab_path=config.io.vocab_dir,
            word_vocab=self._word_vocab,
            pretrained_embedding_path=(
                config.io.pretrained_embedding_path),
            node_vocab=self._node_vocab,
            relation_vocab=self._relation_vocab,
            config=config.graph_updater)
        # self.graph_updater.eval()
        # for param in self.graph_updater.parameters():
        #     param.requires_grad = Tr
        self.policy_net = PolicyNet(
            config=config, word_vocab=self._word_vocab,
            ltl_vocab=self._ltl_vocab,
            action_vocab=self._action_vocab,
            pretrain=pretrain,
            graph_updater=self.graph_updater,
            context_length=self.training.context_length)
        self.target_net = PolicyNet(
            config=config, word_vocab=self._word_vocab,
            ltl_vocab=self._ltl_vocab,
            action_vocab=self._action_vocab,
            pretrain=pretrain,
            graph_updater=self.graph_updater,
            context_length=self.training.context_length)
        self.target_net.train()
        self.update_target_net()
        for param in self.target_net.parameters():
            param.requires_grad = False
        self.replay = ep = config.training.experience_replay
        self.use_belief_graph = config.model.use_belief_graph or \
            config.model.use_ltl
        self.use_ltl = config.model.use_ltl
        self.recurrent_memory = config.model.recurrent_memory
        self.experience = PrioritizedExperienceReplay(
            beta=ep['beta'],
            batch_size=ep['batch_size'],
            multi_step=ep['multi_step'],
            max_episode=config.training.max_episode,
            seed=config.training.random_seed,
            alpha=ep['alpha'],
            capacity=ep['capacity'],
            discount_gamma_game_reward=ep['discount_gamma_game_reward'],
            accumulate_reward_from_final=ep['accumulate_reward_from_final'],
            recurrent_memory=self.recurrent_memory,
            sample_update_from=ep['sample_update_from'],
            sample_history_length=ep['sample_history_length']
        )
        self.epsilon = self.training.epsilon_greedy['anneal_from']
        self.epsilon_scheduler = LinearSchedule(
            schedule_timesteps=self.training.epsilon_greedy['episodes'],
            initial_p=self.epsilon,
            final_p=self.training.epsilon_greedy['anneal_to'])
        self.optimizer, self.scheduler = get_optimizer(
            self.policy_net, config.training.optimizer)
        if self.training.cuda:
            self.cuda()

    def save_model(self, episode_no: int, path: Path,
                   best_train: int, best_eval: int) -> None:
        data = {
            'policy_net': self.policy_net.state_dict(),
            'policy_net_word_vocab': self.policy_net.word_vocab,
            'policy_net_ltl_vocab': self.policy_net.ltl_vocab,
            'policy_net_action_vocab': self.policy_net.action_vocab,
            'episode_no': episode_no,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.optimizer.state_dict(),
            'best_train': best_train,
            'best_eval': best_eval,
        }
        torch.save(data, path)

    def load_model(self, path: Path) -> None:
        data = torch.load(path)
        if 'policy_net' not in data:
            self.policy_net.load_state_dict(data)
            return 0
        self.policy_net.load_vocab(data['policy_net_word_vocab'],
                                   data['policy_net_ltl_vocab'],
                                   data['policy_net_action_vocab'])
        self.policy_net.load_state_dict(data['policy_net'])
        self.target_net.load_vocab(data['policy_net_word_vocab'],
                                   data['policy_net_ltl_vocab'],
                                   data['policy_net_action_vocab'])
        self.update_target_net()
        self.optimizer.load_state_dict(data['optimizer'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(data['scheduler'])
        if self.training.cuda:
            self.cuda()
        return data['episode_no'], data['best_train'], data['best_eval']

    def reset_eval_states(self,
                          obs: List[str],
                          actions: List[str],
                          infos: Dict[str, List[Any]]) -> BatchedStates:
        config = self.evaluate if self.mode == AgentModes.eval else \
            self.test_config
        if config.difficulty_level in {'r', 'mixed'}:
            diffs = [self.uuid_mapping[game.metadata['uuid']]
                     if game.metadata['uuid'] in self.uuid_mapping else 11
                     for game in infos['game']]
        else:
            diffs = [config.difficulty_level for _ in range(
                config.batch_size)]
        self.graph_updater.prev_graph_hidden_state = \
            torch.zeros(
                len(obs), self.graph_updater.config.block_hidden_dim).cuda()
        belief_graphs = [BeliefGraph(
            o, node_vocab=self._node_vocab,
            relation_vocab=self._relation_vocab,
            ground_truth=self.graph_updater.use_ground_truth_graph,
            seperator='_') for
            o in obs]
        facts = self.graph_updater(
            [f"{o} <sep> {a}" for o, a in zip(obs, actions)],
            [bg.facts_as_triplets for bg in belief_graphs],
            actions=actions,
            infos=infos)
        for i, facts_ in enumerate(facts):
            belief_graphs[i].update(facts_)
            belief_graphs[i].update_memory()
        if self.use_ltl or (
                self.training.graph_reward_lambda > 0
                and self.training.graph_reward_filtered):
            ltl_formulas = [LTL(
                facts=bg.facts,
                win_facts=win_facts,
                fail_facts=fail_facts,
                use_ground_truth=self.ltl.use_ground_truth,
                reward_scale=self.ltl.reward_scale,
                first_obs=first_obs,
                as_bonus=self.ltl.as_bonus,
                next_constrained=self.ltl.next_constrained,
                difficulty=diff,
                incomplete_cookbook=self.ltl.incomplete_cookbook,
                single_reward=self.ltl.single_reward,
                single_token_prop=self.ltl.single_token_prop,
                reward_per_progression=self.training.reward_per_ltl_progression,
                no_cookbook=self.ltl.no_cookbook,
                negative_for_fail=self.ltl.negative_for_fail,
                dont_progress=self.ltl.dont_progress)
                for first_obs, bg, win_facts, fail_facts, diff in
                zip(obs, belief_graphs, infos['win_facts'],
                    infos['fail_facts'],
                    diffs)]
        else:
            ltl_formulas = [None for _ in obs]
        self._eval_states = BatchedStates(observations=obs,
                                          actions=actions,
                                          ltl_formulas=ltl_formulas,
                                          belief_graphs=belief_graphs)

    @ property
    def eval_states(self) -> BatchedStates:
        return self._eval_states

    def update_eval_states(self, observations: List[str],
                           current_actions: List[str],
                           dones: List[bool],
                           infos: Dict[str, Any] = None) -> None:
        if self.use_belief_graph:
            facts = self.graph_updater(
                [f"{obs} <sep> {a}" for obs, a in zip(
                    observations, current_actions)],
                [bg.facts_as_triplets for bg in
                    self._eval_states.belief_graphs],
                actions=current_actions,
                infos=infos)
        else:
            facts = [None for _ in observations]
        return self._eval_states.update(observations, current_actions,
                                        facts, dones)

    def reset_states(self,
                     obs: List[str],
                     actions: List[str],
                     infos: Dict[str, List[Any]]) -> BatchedStates:
        if self.training.difficulty_level in {'r', 'mixed'}:
            diffs = [self.uuid_mapping[game.metadata['uuid']]
                     if game.metadata['uuid'] in self.uuid_mapping else 11
                     for game in infos['game']]
        else:
            diffs = [self.training.difficulty_level for _ in range(
                self.training.batch_size)]
        self.graph_updater.prev_graph_hidden_state = \
            torch.zeros(
                len(obs), self.graph_updater.config.block_hidden_dim).cuda()
        belief_graphs = [BeliefGraph(
            o, node_vocab=self._node_vocab,
            relation_vocab=self._relation_vocab,
            ground_truth=self.graph_updater.use_ground_truth_graph,
            seperator='_') for
            o in obs]
        facts = self.graph_updater(
            [f"{o} <sep> {a}" for o, a in zip(obs, actions)],
            [bg.facts_as_triplets for bg in belief_graphs],
            actions=actions,
            infos=infos)
        for i, facts_ in enumerate(facts):
            belief_graphs[i].update(facts_)
            belief_graphs[i].update_memory()
        if self.use_ltl or (
                self.training.graph_reward_lambda > 0
                and self.training.graph_reward_filtered):
            ltl_formulas = [LTL(
                facts=bg.facts,
                win_facts=win_facts,
                fail_facts=fail_facts,
                use_ground_truth=self.ltl.use_ground_truth,
                reward_scale=self.ltl.reward_scale,
                as_bonus=self.ltl.as_bonus,
                first_obs=first_obs,
                next_constrained=self.ltl.next_constrained,
                difficulty=diff,
                incomplete_cookbook=self.ltl.incomplete_cookbook,
                single_reward=self.ltl.single_reward,
                single_token_prop=self.ltl.single_token_prop,
                reward_per_progression=self.training.reward_per_ltl_progression,
                no_cookbook=self.ltl.no_cookbook,
                negative_for_fail=self.ltl.negative_for_fail,
                dont_progress=self.ltl.dont_progress)
                for first_obs, bg, win_facts, fail_facts, diff in
                zip(obs, belief_graphs, infos['win_facts'],
                    infos['fail_facts'],
                    diffs)]
        else:
            ltl_formulas = [None for _ in obs]
        self._states = BatchedStates(observations=obs,
                                     actions=actions,
                                     ltl_formulas=ltl_formulas,
                                     belief_graphs=belief_graphs)

    @ property
    def states(self) -> BatchedStates:
        return self._states

    def update_states(self, observations: List[str],
                      current_actions: List[str],
                      dones: List[bool],
                      infos: Dict[str, Any] = None) -> None:
        if self.use_belief_graph:
            facts = self.graph_updater(
                [f"{obs} <sep> {a}" for obs, a in zip(
                    observations, current_actions)],
                [bg.facts_as_triplets for bg in self._states.belief_graphs],
                actions=current_actions,
                infos=infos)
        else:
            facts = [None for _ in observations]
        return self._states.update(observations, current_actions, facts, dones)

    def cuda(self) -> None:
        if torch.cuda.is_available():
            # torch.set_default_tensor_type(torch.cuda.FloatTensor)
            self.policy_net.cuda()
            self.target_net.cuda()
            self.graph_updater.cuda()
            # torch.backends.cudnn.deterministic = True
        else:
            logger.critical("CUDA set but no CUDA device available")
            raise RuntimeError("CUDA set but no CUDA device available")

    def set_random_seed(self, seed: int) -> None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.training.cuda:
            torch.cuda.manual_seed(seed)

    def update_target_net(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def eval(self) -> None:
        self.policy_net.eval()
        self.mode = AgentModes.eval

    def test(self) -> None:
        self.policy_net.eval()
        self.mode = AgentModes.test

    def train(self) -> None:
        self.policy_net.train()
        self.mode = AgentModes.train

    def games_max_scores(self, infos: Dict[str, Any]) -> List[float]:
        scores = [game.max_score for game in infos["game"]]
        original_scores = copy.copy(scores)
        score_offset = np.zeros_like(scores)
        # if self.training.use_negative_reward:
        #     score_offset = 1
        if self.training.reward_ltl and self.policy_net.use_ltl:
            optimal_path_lengths = np.array([np.sum([len(tuples) > 0 for
                                                     tuples in quest])
                                             for quest in infos['win_facts']])
            # if self.training.reward_per_ltl_progression:
            #     # +1 for cookbook
            #     score_offset = 1 + optimal_path_lengths
            # else:
            if self.training.reward_per_ltl_progression:
                score_offset += (1 + optimal_path_lengths) \
                    * self.ltl.reward_scale
            elif self.ltl.single_reward:
                score_offset = np.ones_like(scores)
            else:
                states = self.states if self.mode == AgentModes.train else \
                    self.eval_states
                _len = [len(s.ltl.translator.formulas) for s in states]
                score_offset = _len * self.ltl.reward_scale
            if self.training.reward_ltl_only:
                return (original_scores,
                        (np.ones_like(scores) * score_offset).tolist())
        if self.training.penalize_path_length > 0:
            score_offset -= self.training.penalize_path_length * \
                optimal_path_lengths
        return (original_scores,
                (np.array(scores) + score_offset).tolist())

    # TODO maybe a better way to do this?
    def adjust_scores(self, scores: List[int],
                      step_no: int,
                      state_update_rewards: List[float],
                      state_update_dones: List[int],
                      wons: List[bool],
                      dones: List[bool]) -> List[float]:
        original_scores = copy.copy(scores)
        reward_ltl = self.training.reward_ltl and self.policy_net.use_ltl
        if reward_ltl:
            # if not self.training.reward_per_ltl_progression:
            #     state_update_rewards = state_update_dones
            pass
            # if self.training.reward_ltl_only:
            #     return original_scores, state_update_rewards
        else:
            state_update_rewards = [0] * len(scores)
        score_offsets = np.array(state_update_rewards)
        if self.training.reward_ltl_positive_only and reward_ltl:
            score_offsets[np.where(score_offsets < 0)] = 0
        if reward_ltl and self.training.reward_ltl_only:
            return original_scores, score_offsets
        if self.training.penalize_path_length > 0:
            score_offsets -= self.training.penalize_path_length * (step_no + 1)
        scores = np.array(scores)
        # if self.training.use_negative_reward:
        #     neg_offset = (np.array(dones, dtype=float) * -1) + \
        #         (np.array(wons, dtype=float))
        #     scores[np.where(np.array(neg_offset) < 0)] = 0
        #     score_offsets + neg_offset
        adjusted_scores = (scores + score_offsets).tolist()
        if self.training.reward_ltl and self.policy_net.use_ltl and \
                self.training.persistent_negative_reward:
            for i, ltl_score in enumerate(state_update_rewards):
                if ltl_score < 0:
                    adjusted_scores[i] = ltl_score
                if dones[i] and not wons[i]:
                    adjusted_scores[i] = -1
        if self.training.graph_reward_lambda > 0:
            adjusted_scores += self.training.graph_reward_lambda * \
                self.get_graph_rewards()
        return original_scores, adjusted_scores

    def get_graph_rewards(self) -> np.ndarray:
        states = self.states if self.mode == AgentModes.train else \
            self.eval_states
        return np.array([0 if len(state.past) < 2 else
                         state.belief_graph.graph_rewards(
            prev_facts=state.past[-1].belief_graph._facts_as_triplets,
            entities=state.ltl.entities if state.ltl is not None else None,
            filtered=self.training.graph_reward_filtered)
            for state in states])

    @staticmethod
    def env_infos(light: bool, win_formulas: bool) -> EnvInfos:
        request_infos = EnvInfos()
        request_infos.game = True
        request_infos.facts = not light
        request_infos.win_facts = win_formulas
        request_infos.fail_facts = win_formulas
        request_infos.description = not light
        request_infos.won = True
        request_infos.lost = True
        request_infos.admissible_commands = True
        return request_infos

    def choose_actions_indices(
            self, actions_scores: torch.Tensor,
            actions_mask: torch.Tensor, admissible_actions: List[Actions],
            greedy: bool, random: bool, eps: float) -> torch.Tensor:
        if greedy and random:
            logging.critical(
                "Asked to act greedily and randomly which is not possible")
            raise ValueError(
                "Asked to act greedily and randomly which is not possible")
        elif greedy:
            return self.choose_maxQ_actions(actions_scores, actions_mask).cpu()
        elif random:
            return self.choose_random_actions(admissible_actions)
        batch_size = len(actions_scores)
        maxQ_filtered_actions = self.choose_maxQ_actions(
            actions_scores, actions_mask).cpu()
        random_filtered_actions = self.choose_random_actions(
            admissible_actions)
        r = np.random.uniform(low=0., high=1.,
                              size=batch_size)
        # less than selects random
        if eps is None:
            eps = self.epsilon
        less_than_e = torch.tensor(
            r <= eps, dtype=torch.int64).reshape((batch_size, 1))
        # stack both options, gather based on less than epsilon
        return torch.gather(torch.stack((maxQ_filtered_actions,
                                         random_filtered_actions), dim=1),
                            dim=-1, index=less_than_e)

    def choose_maxQ_actions(
            self, actions_scores: torch.Tensor,
            actions_mask: torch.Tensor) -> torch.Tensor:
        actions_scores += -torch.min(
            actions_scores, -1, keepdim=True)[0] + 1e-2
        actions_scores *= actions_mask
        if self.mode == AgentModes.test and self.test_config.softmax:
            ret = torch.tensor([])
            for b, mask in zip(actions_scores, actions_mask):
                tmp = torch.functional.F.softmax(
                    b / self.test_config.softmax_temperature).detach().cpu()
                tmp *= mask.detach().cpu()
                options = torch.nonzero(torch.isclose(
                    tmp, tmp.max(), atol=1e-4)).flatten()
                ret = torch.cat(
                    (ret, torch.tensor([np.random.choice(options)])))
            return torch.tensor(ret).int()
        else:
            return torch.argmax(actions_scores.detach().cpu(), -1,)

    def choose_random_actions(
            self, admissible_actions: List[Actions]) -> torch.Tensor:
        return torch.tensor([
            np.random.choice(
                len(candidates)) for candidates in admissible_actions],
            dtype=torch.int32)

    def act(self,
            states: BatchedStates,
            admissible_actions: List[Actions],
            greedy: bool = False,
            random: bool = False,
            previous_hidden: torch.Tensor = None,
            previous_cell: torch.Tensor = None,
            eps: float = None) -> Actions:
        actions_scores, actions_mask = None, None
        if not random:
            actions_scores, actions_mask, previous_hidden, previous_cell = \
                self.policy_net(
                    states, admissible_actions, previous_hidden, previous_cell)
        with torch.no_grad():
            next_actions_indices = self.choose_actions_indices(
                actions_scores=actions_scores,
                actions_mask=actions_mask,
                admissible_actions=admissible_actions,
                greedy=greedy,
                random=random,
                eps=eps)
        return next_actions_indices.squeeze(), previous_hidden, previous_cell

    def get_loss(self, episode_no: int,) -> Tuple[float, float]:
        # pdb.set_trace()
        _samples, _stepped_samples, _rewards, sample_indices, _weights = \
            self.experience.get_samples(
                episode_no, self.recurrent_memory)
        if _samples is None:
            return None, None
        # losses, q_values = list(), list()
        sample_indices = np.array(sample_indices)
        all_q_values, td_errors, dones = list(), list(), list()
        losses, mlm_losses = list(), list()
        previous_hidden, previous_cell = None, None
        for step_no, (samples, stepped_samples, rewards, weights) in \
                enumerate(zip(_samples, _stepped_samples, _rewards, _weights)):
            stepped_states, stepped_admissible_actions = list(), list()
            states, admissible_actions, indices = list(), list(), list()
            for sample, stepped_sample in zip(samples, stepped_samples):
                states.append(sample.state)
                admissible_actions.append(sample.admissible_actions)
                indices.append(sample.action)
                stepped_states.append(stepped_sample.state)
                stepped_admissible_actions.append(
                    stepped_sample.admissible_actions)
                dones.append(sample.done)
            states = BatchedStates(states=states)
            stepped_states = BatchedStates(states=stepped_states)
            not_dones = 1 - torch.tensor(
                dones, device=self.policy_net.device, dtype=torch.int64)
            actions_scores, actions_mask, previous_hidden, previous_cell = \
                self.policy_net(
                    states, admissible_actions, previous_hidden, previous_cell)
            q_values = torch.gather(
                actions_scores, 1, torch.tensor(
                    indices, dtype=torch.int64,
                    device=self.policy_net.device).reshape((-1, 1)))
            if self.recurrent_memory and \
                    step_no < self.replay['sample_update_from']:
                continue
            with torch.no_grad():
                stepped_actions_scores, stepped_actions_mask, _, _ = \
                    self.policy_net(
                        stepped_states, stepped_admissible_actions,
                        previous_hidden, previous_cell)
                stepped_indices = self.choose_maxQ_actions(
                    stepped_actions_scores, stepped_actions_mask
                ).to(self.policy_net.device)
                stepped_indices = stepped_indices.reshape((-1, 1))
                stepped_actions_scores_tgt, stepped_actions_tgt_mask, _, _ = \
                    self.target_net(
                        stepped_states, stepped_admissible_actions,
                        previous_hidden, previous_cell)
                # stepped_actions_scores_tgt *= not_dones.unsqueeze(1)
                stepped_q_values = torch.gather(
                    stepped_actions_scores_tgt, 1, stepped_indices)
                discount = torch.tensor(
                    (np.ones((stepped_indices.shape[0])) *
                     self.replay['discount_gamma_game_reward']) **
                    sample_indices[:, 1],
                    device=self.policy_net.device, dtype=torch.float64)
                # dones = torch.tensor(
                #     [s.is_final for s in samples], dtype=torch.float64,
                #     device=self.policy_net.device)
                # discount *= (1 - dones)
            # pdb.set_trace()
            rewards = torch.tensor(rewards, device=self.policy_net.device) + \
                stepped_q_values.squeeze() * (discount * not_dones)
            # ** self.replay['multi_step'])
            rewards = rewards.type(torch.float32)
            loss = F.smooth_l1_loss(
                q_values.squeeze(), rewards, reduction='none')
            loss *= torch.tensor(
                weights, device=self.policy_net.device, dtype=torch.float64)
            losses.append(loss)
            loss = self.policy_net.mlm_loss(states.observations)
            mlm_losses.append(loss)
            all_q_values.append(q_values)

            # q_values has shape [*mod-replay-batch-size*, 1]
            # rewards has shape [*mod-replay-batch-size*,
            # *mod-replay-batch-size*]
            abs_td_err = torch.abs(q_values.squeeze() - rewards)
            td_errors.append(abs_td_err)

        _range = 1
        if self.recurrent_memory:
            _range = self.replay['sample_history_length'] - \
                self.replay['sample_update_from']
        for i in range(_range):
            abs_td_err = td_errors[i]
            td_errors.append(abs_td_err)
            new_priorities = abs_td_err + self.replay['eps']
            self.experience.update_priorities(
                sample_indices[:, 0] + i + (self.replay['sample_update_from']
                                            if self.recurrent_memory else 0),
                new_priorities.detach().cpu())

        loss = torch.stack(losses).mean()
        q_values = torch.stack(all_q_values).mean()
        if any(mlm_losses):
            loss = loss + self.training.mlm_alpha * \
                torch.stack(mlm_losses).mean()
        return loss, q_values

    def update_dqn(self, episode_no: int) -> Tuple[float, float]:
        loss, q_values = self.get_loss(episode_no)
        if loss is None:
            return None, None

        self.policy_net.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        if np.greater(self.training.optimizer['clip_grad_norm'], 0.):
            torch.nn.utils.clip_grad_norm_(
                self.policy_net.parameters(),
                self.training.optimizer['clip_grad_norm'])
        self.optimizer.step()
        return torch.mean(loss), torch.mean(q_values)

    def finalize_episode(self, episode_no: int) -> None:
        if (episode_no + self.training.batch_size) % \
                self.training.target_net_update_frequency <= \
                episode_no % self.training.target_net_update_frequency:
            self.update_target_net()
        if episode_no < self.training.learn_from_this_episode:
            return
        if episode_no < self.training.epsilon_greedy['episodes'] + \
                self.training.learn_from_this_episode:
            self.epsilon = self.epsilon_scheduler.value(
                episode_no - self.training.learn_from_this_episode)
            self.epsilon = max(self.epsilon, 0.)
        if self.scheduler is not None and episode_no > \
                self.training.learn_from_this_episode:
            for _ in range(self.training.batch_size):
                self.scheduler.step()

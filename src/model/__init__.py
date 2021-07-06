from typing import List, Tuple
from argparse import Namespace

import logging
import pdb

import numpy as np
import torch

from utils import max_len, to_pt, pad_sequences
from components import Actions, Vocabulary
from model.features import TextEncoder
from model.layers import LSTMCell
from state import BatchedStates


logger = logging.getLogger()


class ActionNet(torch.nn.Module):
    def __init__(self, hidden: int, num_inputs: int, **kwargs) -> None:
        super(ActionNet, self).__init__(**kwargs)
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(hidden * num_inputs, hidden),
            torch.nn.ReLU(),
        )
        self.final = torch.nn.Linear(hidden, 1)

    def forward(self, inputs: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        out = self.layers(inputs) * mask.unsqueeze(-1)
        return self.final(out).squeeze(-1) * mask


class PolicyNet(torch.nn.Module):
    def __init__(self, config: Namespace,
                 word_vocab: Vocabulary,
                 ltl_vocab: Vocabulary,
                 action_vocab: Vocabulary,
                 pretrain: bool,
                 graph_updater=None,
                 context_length: int = 1,
                 **kwargs) -> None:
        super(PolicyNet, self).__init__(**kwargs)
        self._dummy = torch.nn.Parameter(torch.empty(0))
        # TODO clean this up
        self.config = config
        for k, v in vars(config.model).items():
            setattr(self, k, v)

        self.pretrain = pretrain
        self.context_length = context_length
        self.build(config, word_vocab=word_vocab,
                   ltl_vocab=ltl_vocab,
                   action_vocab=action_vocab,
                   pretrain=pretrain,
                   graph_updater=graph_updater)

    @property
    def device(self) -> str:
        return self._dummy.device

    def load_vocab(self, word_vocab: Vocabulary,
                   ltl_vocab: Vocabulary,
                   action_vocab: Vocabulary) -> None:
        assert isinstance(word_vocab, Vocabulary)
        assert isinstance(ltl_vocab, Vocabulary)
        assert isinstance(action_vocab, Vocabulary)
        if self.concat_strings:
            self.word_vocab = word_vocab
            self.ltl_vocab = ltl_vocab
            self.action_vocab = action_vocab
            concat_vocab = action_vocab
            concat_vocab.name = 'concat-vocab'
            concat_vocab += ['[ACTION]']
            if self.use_observations:
                concat_vocab += word_vocab
                concat_vocab += ['[OBS]']
            if self.use_ltl:
                concat_vocab += ltl_vocab
                concat_vocab += ['[LTL]']
            self.text_encoder = TextEncoder(config=self.config.text_encoder,
                                            vocab=concat_vocab)
        else:
            self.ltl_vocab = ltl_vocab
            self.word_vocab = word_vocab
            self.action_vocab = action_vocab

            diff_ltl = self.use_ltl and not self.same_ltl_text_encoder
            if diff_ltl and self.ltl_encoder is not None:
                if len(ltl_vocab) != len(self.ltl_encoder.vocab):
                    self.ltl_encoder = TextEncoder(
                        config=self.config.ltl_encoder, vocab=ltl_vocab)
                else:
                    self.ltl_encoder.load_vocab(ltl_vocab)
            if self.text_encoder is not None:
                if len(word_vocab) != len(self.text_encoder.vocab):
                    self.text_encoder = TextEncoder(
                        config=self.config.text_encoder, vocab=word_vocab,)
                else:
                    self.text_encoder.load_vocab(word_vocab)

    def build(self, config: Namespace, word_vocab: Vocabulary,
              ltl_vocab: Vocabulary, action_vocab: Vocabulary,
              pretrain: bool, graph_updater) -> None:
        # assert self.action_net_hidden_size == 768 if \
        #     self.use_pretrained_lm_for_text else True, \
        #     "Action net hidden size must match BERT output size of 768"
        self.text_encoder, self.ltl_encoder, self.graph_encoder, \
            self.actions_encoder = None, None, None, None
        if self.concat_strings:
            self.word_vocab = word_vocab
            self.ltl_vocab = ltl_vocab
            self.action_vocab = action_vocab
            concat_vocab = action_vocab
            concat_vocab.name = 'concat-vocab'
            concat_vocab += ['[ACTION]']
            if self.use_observations:
                concat_vocab += word_vocab
                concat_vocab += ['[OBS]']
            if self.use_ltl:
                concat_vocab += ltl_vocab
                concat_vocab += ['[LTL]']
            self.text_encoder = TextEncoder(config=config.text_encoder,
                                            vocab=concat_vocab)
            # 1 for the encoded admissible actions
            self.action_network = ActionNet(hidden=self.action_net_hidden_size,
                                            num_inputs=1)
            if self.recurrent_memory:
                num_inputs = 1
                self.recurrent_memory_unit = LSTMCell(
                    self.action_net_hidden_size * num_inputs,
                    self.action_net_hidden_size, use_bias=True)
        else:
            if pretrain:
                self.ltl_encoder = TextEncoder(config=config.ltl_encoder,
                                               vocab=ltl_vocab,)
            elif self.use_ltl:
                # assert not bool(self.same_ltl_text_encoder *
                #                 self.ltl_text_string_concat), \
                #     "Config violation: 'same_ltl_text_encoder' and " + \
                #     "'ltl_text_string_concat' can't both be True"
                if self.same_ltl_text_encoder:
                    word_vocab += ltl_vocab
                    ltl_vocab += word_vocab
                else:
                    # ltl_vocab += word_vocab
                    self.ltl_encoder = TextEncoder(config=config.ltl_encoder,
                                                   vocab=ltl_vocab,)

            if self.use_observations:
                if pretrain and config.pretrain.text or not pretrain:
                    self.text_encoder = TextEncoder(config=config.text_encoder,
                                                    vocab=word_vocab,)
            if self.use_ltl and self.same_ltl_text_encoder and \
                    not pretrain:
                self.ltl_encoder = self.text_encoder
            if self.use_belief_graph:
                self.graph_encoder = graph_updater
            # self.load_vocab(word_vocab, ltl_vocab,)

            self.ltl_vocab = ltl_vocab
            self.word_vocab = word_vocab
            if self.use_independent_actions_encoder:
                self.action_vocab = action_vocab
                self.actions_encoder = TextEncoder(
                    config=config.actions_encoder, vocab=action_vocab,)
            else:
                self.action_vocab = self.word_vocab
                self.actions_encoder = self.text_encoder

            # 1 for the encoded admissible actions
            num_inputs = 1 + np.sum(np.array([
                self.use_observations,
                self.use_belief_graph,
                self.use_ltl]))
            self.action_network = ActionNet(hidden=self.action_net_hidden_size,
                                            num_inputs=num_inputs)
            if self.recurrent_memory:
                num_inputs -= 1
                self.recurrent_memory_unit = LSTMCell(
                    self.action_net_hidden_size * num_inputs,
                    self.action_net_hidden_size, use_bias=True)

    def encode_actions(self, actions: List[List[str]]) -> torch.Tensor:
        """
        # actions come out as NumActionsxLengthxEmbed
        # stacking gives BatchxNumActionsxLengthxEmbed
        @
        returns: torch Tensor [batch-size, num-actions, embed-size]
        """
        # we first sum over the length (dim=1) then pad the num actions
        # since num actions per batch may not be the same size
        if self.use_pretrained_lm_for_actions:
            actions_mask = list()
            unwrapped_actions = list()
            batch_size = len(actions)
            max_num_action = max_len(actions)
            for _actions in actions:
                actions_len = len(_actions)
                padding_len = (max_num_action - actions_len)
                unwrapped_actions.extend(
                    _actions + [self.actions_encoder.tokenizer.pad_token] *
                    padding_len)
                actions_mask.extend([1] * len(_actions) + [0] * (padding_len))
            encoded_actions, _mask = self.actions_encoder(
                unwrapped_actions)
            max_word_num = _mask.shape[1]
            actions_mask = torch.tensor(actions_mask, device=self.device)
            encoded_actions = encoded_actions.view(
                batch_size, max_num_action, max_word_num, -1)
            actions_mask = actions_mask.view(
                batch_size, max_num_action)

            # batch-size x max-num-action
            # batch-size x max-num-action x hid
            # _mask = torch.sum(actions_mask, -1)
            _mask = actions_mask
            encoded_actions = torch.sum(encoded_actions, dim=-2)
            tmp = torch.eq(_mask, 0).float().to(self.device)
            _mask = _mask + tmp
            # batch-size x max-num-action x hid
            encoded_actions = encoded_actions / _mask.unsqueeze(-1)
        else:
            batch_size = len(actions)
            max_num_action = max_len(actions)
            input_action_candidate_list = list()
            for i in range(batch_size):
                word_list = [item.split() for item in actions[i]]
                word_id_list = [[self.action_vocab[tok] for tok in tokens]
                                for tokens in word_list]
                input_word = pad_sequences(
                    word_id_list, maxlen=max_len(word_id_list)).astype('int32')
                word_level = to_pt(input_word, True)
                input_action_candidate_list.append(word_level)
            max_word_num = max([item.size(1)
                                for item in input_action_candidate_list])

            inputs = torch.zeros(
                (batch_size, max_num_action, max_word_num),
                device=self.device, dtype=torch.long)
            for i in range(batch_size):
                j, k = input_action_candidate_list[i].shape
                assert j == input_action_candidate_list[i].size(0)
                assert k == input_action_candidate_list[i].size(1)
                inputs[i, :j, :k] = input_action_candidate_list[i]

            inputs = inputs.view(batch_size * max_num_action, max_word_num)

            encoded_actions, actions_mask = self.actions_encoder(
                inputs, compute_word_ids=False)
            if self.actions_encoder.lstm_backbone:
                encoded_actions = encoded_actions.view(
                    batch_size, max_num_action, -1)
            elif self.actions_encoder.mlp_backbone:
                encoded_actions = encoded_actions.view(
                    batch_size, max_num_action, -1)
            else:
                encoded_actions = encoded_actions.view(
                    batch_size, max_num_action, max_word_num, -1)
                encoded_actions = torch.sum(encoded_actions, dim=-2)
            actions_mask = actions_mask.view(
                batch_size, max_num_action, max_word_num)
            # batch-size x max-num-action
            _mask = torch.sum(actions_mask, -1)
            # batch-size x max-num-action x hid
            tmp = torch.eq(_mask, 0).float().to(self.device)
            _mask = _mask + tmp
            # batch-size x max-num-action x hid
            encoded_actions = encoded_actions / _mask.unsqueeze(-1)
            actions_mask = actions_mask.byte().any(-1).float()
        return encoded_actions, actions_mask

    def combine_features(
            self, num_actions: int, batch_size: int,
            encoded_obs: torch.Tensor, obs_mask: torch.Tensor,
            encoded_bg: torch.Tensor, bg_mask: torch.Tensor,
            encoded_ltl: torch.Tensor, ltl_mask: torch.Tensor,
            previous_hidden: torch.Tensor = None,
            previous_cell: torch.Tensor = None) -> torch.Tensor:
        if self.concat_features:
            encoded_features = None
            for name, feature, mask in [
                ('obs', encoded_obs, obs_mask),
                ('ltl', encoded_ltl, ltl_mask),
                ('bg', encoded_bg, bg_mask),
            ]:
                if feature is None:
                    continue
                if name == 'obs':
                    sumit = self.text_encoder.lstm_backbone or \
                        self.text_encoder.mlp_backbone if\
                        self.text_encoder else None
                elif name == 'ltl':
                    sumit = self.ltl_encoder.lstm_backbone or \
                        self.ltl_encoder.mlp_backbone if\
                        self.ltl_encoder else None
                elif name == 'bg':
                    sumit = False
                # masked mean
                # if name == 'obs' and not self.use_pretrained_lm_for_text or \
                #         name == 'ltl' and not self.use_pretrained_lm_for_ltl:
                _mask = torch.sum(mask, -1)  # batch
                if not sumit:
                    feature = torch.sum(feature, dim=1)  # batch x hid
                tmp = torch.eq(_mask, 0).float().to(self.device)
                _mask = _mask + tmp
                feature = feature / \
                    _mask.unsqueeze(-1)  # batch x hid
                # TODO check this for pretraining
                # if num_actions > 1:
                # feature = torch.stack([feature] * num_actions, dim=1)
                if encoded_features is None:
                    encoded_features = feature
                else:
                    encoded_features = torch.cat([encoded_features, feature],
                                                 dim=-1)
        else:
            logger.critical(
                "Concat features is disable but no other " +
                "aggregation mechanism exists")
            raise RuntimeError(
                "Concat features is disable but no other " +
                "aggregation mechanism exists")
        if self.recurrent_memory:
            previous_hidden, previous_cell = \
                self.recurrent_memory_unit(encoded_features,
                                           h_0=previous_hidden,
                                           c_0=previous_cell)
        return torch.stack([encoded_features] * num_actions, dim=1), \
            previous_hidden, previous_cell

    def encode(self, states: BatchedStates,
               admissible_actions: List[Actions],
               previous_hidden: torch.Tensor,
               previous_cell: torch.Tensor) -> torch.Tensor:
        """
        @returns:
          encoded_features: torch Tensor of size
            [batch-size, num-action-candidates, embedding-size]
          mask: torch Tensor of size [batch-size, num-action-candidates]
            used to mask the padded actions from the action network
        """
        encoded_obs, obs_mask, encoded_bg, bg_mask, encoded_ltl, ltl_mask = \
            tuple([None] * 6)
        if self.concat_strings:
            # encoded_obs, obs_mask = self.text_encoder([
            obs = None
            ltl = None
            batch_size = len(admissible_actions)
            if self.use_observations:
                obs = [
                    ' '.join(['[OBS]', obs]) for
                    obs in states.observations]
            if self.use_ltl:
                ltl = [
                    ' '.join(['[LTL]', ltl.tokenize()]) for
                    ltl in states.ltl_formulas]
            max_num_action = max_len(admissible_actions)
            inputs = list()
            final_mask = list()
            for i, actions in enumerate(admissible_actions):
                pad = [self.text_encoder.vocab.pad_token for _ in range(
                    max_num_action - len(actions))]
                final_mask.extend([1] * len(actions) + [0] * len(pad))
                actions += pad
                for action in actions:
                    if obs and ltl:
                        inputs.append(
                            ' '.join([obs[i], ltl[i], '[ACTION]', action]))
                    elif obs:
                        inputs.append(
                            ' '.join([obs[i], '[ACTION]', action]))
                    elif ltl:
                        inputs.append(
                            ' '.join([ltl[i], '[ACTION]', action]))
            encodings, mask = self.text_encoder(inputs)
            _mask = torch.sum(mask, -1)
            if not self.text_encoder.lstm_backbone:
                encodings = torch.sum(encodings, dim=1)  # batch x hid
            tmp = torch.eq(_mask, 0).float().to(self.device)
            _mask = _mask + tmp
            encodings = encodings / \
                _mask.unsqueeze(-1)  # batch x hid
            encodings = encodings.reshape((batch_size, max_num_action,
                                           self.action_net_hidden_size))
            final_mask = torch.tensor(final_mask, device=self.device)
            final_mask = final_mask.view(batch_size, max_num_action)

            if self.recurrent_memory:
                previous_hidden, previous_cell = \
                    self.recurrent_memory_unit(encodings,
                                               h_0=previous_hidden,
                                               c_0=previous_cell)
            return encodings, final_mask, previous_hidden, previous_cell

        else:
            obs = list()
            ltls = list()
            if self.context_length == 1:
                if self.use_ltl:
                    ltls = [ltl.tokenize() for ltl in states.ltl_formulas]
                obs = states.observations
            else:
                for state in states:
                    obs.append('<obs> ' + ' <obs> '.join(
                        [s.observation for s in
                            state.past[-(self.context_length - 1):] + [state]]
                    ))
                    ltls.append('<ltl> ' + ' <ltl> '.join(
                        [s.ltl.tokenize() for s in
                            state.past[-(self.context_length - 1):] + [state]]
                    ))
            if self.use_observations:
                encoded_obs, obs_mask = self.text_encoder(obs)
            if self.use_ltl:
                encoded_ltl, ltl_mask = self.ltl_encoder(ltls)
            if self.use_belief_graph:
                if self.graph_encoder.real_valued_graph:
                    encoded_bg, bg_mask = self.graph_encoder.encode_graph(
                        torch.stack([bg._facts if bg is not None else [] for
                                     bg in states.belief_graphs]))
                else:
                    encoded_bg, bg_mask = self.graph_encoder.encode_graph(
                        [bg.facts_as_triplets if bg is not None else [] for
                            bg in states.belief_graphs])

            encoded_actions, actions_mask = \
                self.encode_actions(admissible_actions)
            batch_size, num_actions, _ = encoded_actions.shape
            encoded_features, previous_hidden, previous_cell = \
                self.combine_features(
                    num_actions, batch_size,
                    encoded_obs, obs_mask, encoded_bg, bg_mask,
                    encoded_ltl, ltl_mask,
                    previous_hidden, previous_cell)

            return (torch.cat([encoded_actions, encoded_features], dim=-1),
                    actions_mask, previous_hidden, previous_cell)

    def compute_inverse_dynamics_loss(self, states: BatchedStates):
        obs = states.observations
        if len(obs) <= 1:
            return None
        loss = self.text_encoder.compute_inverse_dynamics_loss(
            obs[:-1], obs[1:], states.actions)
        return loss

    def mlm_loss(self, observations: List[str]) -> torch.Tensor:
        loss = None
        if self.text_encoder is not None:
            loss = self.text_encoder.compute_mlm_loss(observations)
        # if self.ltl_encoder is not None:
        #     loss += self.ltl_encoder.mlm_loss(observations)
        return loss

    def forward(self, states: BatchedStates, admissible_actions: List[Actions],
                previous_hidden: torch.Tensor = None,
                previous_cell: torch.Tensor = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs, mask, previous_hidden, previous_cell = self.encode(
            states, admissible_actions, previous_hidden, previous_cell)
        return self.action_network(inputs, mask), mask, \
            previous_hidden, previous_cell

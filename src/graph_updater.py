from typing import Tuple, List, Dict, Any
from pathlib import Path

import copy

import torch.nn.functional as F
import numpy as np
import torch

from utils import to_pt, max_len, pad_sequences, to_np
from model.layers import (
    CQAttention, PointerSoftmax,
    DecoderBlock, EncoderBlock, Embedding,
    masked_softmax, SelfAttention, masked_mean,
    DecoderBlockForObsGen, ObservationDiscriminator
)
from components import Vocabulary


class RelationalGraphConvolution(torch.nn.Module):
    """
    Simple R-GCN layer, modified from theano/keras implementation from
        https://github.com/tkipf/relational-gcn
    We also consider relation representation here (relation labels matter)
    """

    def __init__(self, entity_input_dim, relation_input_dim,
                 num_relations, out_dim, bias=True, num_bases=0):
        super(RelationalGraphConvolution, self).__init__()
        self.entity_input_dim = entity_input_dim
        self.relation_input_dim = relation_input_dim
        self.out_dim = out_dim
        self.num_relations = num_relations
        self.num_bases = num_bases

        if self.num_bases > 0:
            self.bottleneck_layer = torch.nn.Linear(
                (self.entity_input_dim + self.relation_input_dim) *
                self.num_relations, self.num_bases, bias=False)
            self.weight = torch.nn.Linear(
                self.num_bases, self.out_dim, bias=False)
        else:
            self.weight = torch.nn.Linear(
                (self.entity_input_dim + self.relation_input_dim) *
                self.num_relations, self.out_dim, bias=False)
        if bias:
            self.bias = torch.nn.Parameter(torch.FloatTensor(self.out_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight.weight.data)
        if self.bias is not None:
            self.bias.data.fill_(0)

    def forward(self, node_features, relation_features, adj):
        # node_features: batch x num_entity x in_dim
        # relation_features: batch x num_relation x in_dim
        # adj:   batch x num_relations x num_entity x num_entity
        supports = []
        for relation_idx in range(self.num_relations):
            # batch x 1 x in_dim
            _r_features = relation_features[:, relation_idx: relation_idx + 1]
            _r_features = _r_features.repeat(
                1, node_features.size(1), 1)  # batch x num_entity x in_dim
            # batch x num_entity x in_dim+in_dim
            supports.append(torch.bmm(adj[:, relation_idx], torch.cat(
                [node_features, _r_features], dim=-1)))
        # batch x num_entity x (in_dim+in_dim)*num_relations
        supports = torch.cat(supports, dim=-1)
        if self.num_bases > 0:
            supports = self.bottleneck_layer(supports)
        output = self.weight(supports)  # batch x num_entity x out_dim

        if self.bias is not None:
            return output + self.bias
        else:
            return output


class StackedRelationalGraphConvolution(torch.nn.Module):
    '''
    input:  entity features:    batch x num_entity x input_dim
            relation features:  batch x num_relations x input_dim
            adjacency matrix:   batch x num_relations x num_entity x num_entity
    '''

    def __init__(self, entity_input_dim, relation_input_dim,
                 num_relations, hidden_dims, num_bases,
                 use_highway_connections=False, dropout_rate=0.0,
                 real_valued_graph=False):
        super(StackedRelationalGraphConvolution, self).__init__()
        self.entity_input_dim = entity_input_dim
        self.relation_input_dim = relation_input_dim
        self.hidden_dims = hidden_dims
        self.num_relations = num_relations
        self.dropout_rate = dropout_rate
        self.num_bases = num_bases
        self.real_valued_graph = real_valued_graph
        self.nlayers = len(self.hidden_dims)
        self.stack_gcns()
        self.use_highway_connections = use_highway_connections
        if self.use_highway_connections:
            self.stack_highway_connections()

    def stack_highway_connections(self):
        highways = [torch.nn.Linear(
            self.hidden_dims[i], self.hidden_dims[i]) for i in range(
                self.nlayers)]
        self.highways = torch.nn.ModuleList(highways)
        self.input_linear = torch.nn.Linear(
            self.entity_input_dim, self.hidden_dims[0])

    def stack_gcns(self):
        gcns = [RelationalGraphConvolution(
            self.entity_input_dim if i == 0 else self.hidden_dims[i - 1],
            self.relation_input_dim,
            self.num_relations, self.hidden_dims[i],
            num_bases=self.num_bases)
            for i in range(self.nlayers)]
        self.gcns = torch.nn.ModuleList(gcns)

    def forward(self, node_features, relation_features, adj):
        x = node_features
        for i in range(self.nlayers):
            if self.use_highway_connections:
                if i == 0:
                    prev = self.input_linear(x)
                else:
                    prev = x.clone()
            # batch x num_nodes x hid
            x = self.gcns[i](x, relation_features, adj)
            if self.real_valued_graph:
                x = torch.sigmoid(x)
            else:
                x = F.relu(x)
            x = F.dropout(x, self.dropout_rate, training=self.training)
            if self.use_highway_connections:
                gate = torch.sigmoid(self.highways[i](x))
                x = gate * x + (1 - gate) * prev
        return x


class GraphUpdater(torch.nn.Module):
    def __init__(self, checkpoint: Path,
                 vocab_path: Path,
                 config: Dict[str, Any],
                 word_vocab: Vocabulary,
                 pretrained_embedding_path: Path = None,
                 relation_vocab: Vocabulary = None,
                 node_vocab: Vocabulary = None,
                 **kwargs) -> None:
        super(GraphUpdater, self).__init__(**kwargs)
        self.config = config
        self._dummy = torch.nn.Parameter(torch.empty(0))
        self._facts: List[Tuple[str, str, str]] = list()
        self.use_ground_truth_graph = config.use_ground_truth_graph
        self._word_vocab = Vocabulary(word_vocab.original_tokens,
                                      name='GraphWordVocab',
                                      original_only=True)
        self._node_vocab = node_vocab
        self._relation_vocab = relation_vocab
        self.origin_relation_number = int((
            len(self._relation_vocab) - 1) / 2)
        self.word_embedding = Embedding(
            embedding_size=config.word_embedding_size,
            vocab_size=len(self._word_vocab),
            id2word=self._word_vocab,
            dropout_rate=config.embedding_dropout,
            load_pretrained=True,
            trainable=False,
            embedding_oov_init="random",
            pretrained_embedding_path=str(pretrained_embedding_path))
        self.word_embedding_prj = torch.nn.Linear(
            config.word_embedding_size, config.block_hidden_dim,
            bias=False)
        self.node_embeddings, self.relation_embeddings = None, None
        self.node_embedding = Embedding(
            embedding_size=config.node_embedding_size,
            vocab_size=len(node_vocab),
            trainable=True,
            dropout_rate=config.embedding_dropout)
        self.relation_embedding = Embedding(
            embedding_size=config.relation_embedding_size,
            vocab_size=len(relation_vocab),
            trainable=True,
            dropout_rate=config.embedding_dropout)
        self.rgcns = StackedRelationalGraphConvolution(
            entity_input_dim=config.node_embedding_size +
            config.block_hidden_dim,
            relation_input_dim=config.relation_embedding_size +
            config.block_hidden_dim,
            num_relations=len(self._relation_vocab),
            hidden_dims=config.gcn_hidden_dims,
            num_bases=config.gcn_num_bases,
            use_highway_connections=config.gcn_highway_connections,
            dropout_rate=config.gcn_dropout,
            real_valued_graph=config.use_ground_truth_graph or
            config.real_valued)
        self.real_valued_graph = config.real_valued
        self.self_attention = None
        if config.use_self_attention:
            self.self_attention = SelfAttention(
                config.block_hidden_dim, config.n_heads, 0.)
        if not config.use_ground_truth_graph:
            if self.real_valued_graph:
                # TODO CHANGE THIS TO 50 = batch_size
                self.prev_graph_hidden_state = None
                self.curr_mat = None
                self.obs_gen_attention = CQAttention(
                    block_hidden_dim=config.block_hidden_dim,
                    dropout=config.gcn_dropout)
                self.obs_gen_attention_prj = torch.nn.Linear(
                    config.block_hidden_dim * 4, config.block_hidden_dim,
                    bias=False)
                self.obs_gen_decoder = torch.nn.ModuleList([
                    DecoderBlockForObsGen(
                        ch_num=config.block_hidden_dim, k=5,
                        block_hidden_dim=config.block_hidden_dim,
                        n_head=config.n_heads,
                        dropout=config.block_dropout)
                    for _ in range(config.decoder_layers)])
                self.obs_gen_tgt_word_prj = torch.nn.Linear(
                    config.block_hidden_dim, len(self._word_vocab), bias=False)
                self.obs_gen_linear_1 = torch.nn.Linear(
                    config.block_hidden_dim, config.block_hidden_dim)
                self.obs_gen_linear_2 = torch.nn.Linear(
                    config.block_hidden_dim, int(
                        len(self._relation_vocab) / 2) *
                    len(self._node_vocab) * len(self._node_vocab))
                self.obs_gen_attention_to_rnn_input = torch.nn.Linear(
                    config.block_hidden_dim * 4, config.block_hidden_dim)
                self.obs_gen_graph_rnncell = torch.nn.GRUCell(
                    config.block_hidden_dim, config.block_hidden_dim)
                self.observation_discriminator = ObservationDiscriminator(
                    config.block_hidden_dim)
            self.max_target_length = config.max_target_length
            # Accounts for adding "self" and duplicate "_reverse"
            # see agents.py:79-82
            self.cmd_gen_attention = CQAttention(
                block_hidden_dim=config.block_hidden_dim,
                dropout=config.attention_dropout)
            self.cmd_gen_attention_prj = torch.nn.Linear(
                config.block_hidden_dim * 4,
                config.block_hidden_dim, bias=False)
            self.pointer_softmax = PointerSoftmax(
                input_dim=config.block_hidden_dim,
                hidden_dim=config.block_hidden_dim)
            self.tgt_word_prj = torch.nn.Linear(
                config.block_hidden_dim,
                len(self._word_vocab),
                bias=False)
            self.decoder = torch.nn.ModuleList([
                DecoderBlock(ch_num=config.block_hidden_dim, k=5,
                             block_hidden_dim=config.block_hidden_dim,
                             n_head=config.n_heads,
                             dropout=config.block_dropout)
                for _ in range(config.decoder_layers)])
            self.encoder_for_pretraining_tasks = torch.nn.ModuleList([
                EncoderBlock(conv_num=config.encoder_conv_num,
                             ch_num=config.block_hidden_dim,
                             k=5, block_hidden_dim=config.block_hidden_dim,
                             n_head=config.n_heads,
                             dropout=config.block_dropout)
                for _ in range(config.encoder_layers)])
            self.encoder_conv_num = config.encoder_conv_num
            if config.from_pretrained:
                self.load_checkpoint(checkpoint)

    def load_checkpoint(self, checkpoint: Path) -> None:
        pretrained_dict = torch.load(checkpoint)
        model_dict = self.state_dict()
        del model_dict['_dummy']
        for k, v in model_dict.items():
            if k not in pretrained_dict:
                import pdb
                pdb.set_trace()
            assert k in pretrained_dict
        model_dict = {k: v for k, v in pretrained_dict.items() if
                      k in model_dict}
        self.load_state_dict(model_dict, strict=False)

    @property
    def device(self) -> str:
        return self._dummy.device

    def tokenize(self, inputs: List[str]) -> torch.Tensor:
        word_list = [item.split() for item in inputs]

        word_id_list = [[self._word_vocab[tok] for tok in tokens]
                        for tokens in word_list]
        input_word = pad_sequences(
            word_id_list, maxlen=max_len(word_id_list)).astype('int32')
        input_word = to_pt(input_word, self.device != 'cpu')
        return input_word

    def encode_text(self,
                    inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embeddings, mask = self.embed(inputs)  # batch x seq_len x emb
        # batch x seq_len x seq_len
        squared_mask = torch.bmm(mask.unsqueeze(-1), mask.unsqueeze(1))
        encoding_sequence = embeddings
        for i, encoder in enumerate(self.encoder_for_pretraining_tasks):
            # batch x time x enc
            encoding_sequence = encoder(
                encoding_sequence, squared_mask, i * (
                    self.encoder_conv_num + 2) + 1,
                len(self.encoder_for_pretraining_tasks))
        return encoding_sequence, mask

    def embed(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        word_embeddings, mask = self.word_embedding(
            inputs)  # batch x time x emb
        word_embeddings = self.word_embedding_prj(word_embeddings)
        word_embeddings = word_embeddings * \
            mask.unsqueeze(-1)  # batch x time x hid
        return word_embeddings, mask

    def get_subsequent_mask(self, seq: torch.Tensor) -> torch.Tensor:
        ''' For masking out the subsequent info. '''
        _, length = seq.size()
        subsequent_mask = torch.triu(torch.ones(
            (length, length)), diagonal=1).float()
        subsequent_mask = 1.0 - subsequent_mask
        if seq.is_cuda:
            subsequent_mask = subsequent_mask.cuda()
        subsequent_mask = subsequent_mask.unsqueeze(0)  # 1 x time x time
        return subsequent_mask

    def decode(self, inputs: torch.Tensor,
               h_og: torch.Tensor,
               obs_mask: torch.Tensor,
               h_go: torch.Tensor,
               node_mask: torch.Tensor,
               input_obs: torch.Tensor) -> torch.Tensor:
        trg_embeddings, trg_mask = self.embed(
            inputs)  # batch x target_len x emb

        # batch x target_len x target_len
        trg_mask_square = torch.bmm(
            trg_mask.unsqueeze(-1), trg_mask.unsqueeze(1))
        trg_mask_square = trg_mask_square * \
            self.get_subsequent_mask(
                inputs)  # batch x target_len x target_len

        # batch x target_len x obs_len
        obs_mask_square = torch.bmm(
            trg_mask.unsqueeze(-1), obs_mask.unsqueeze(1))
        # batch x target_len x node_len
        node_mask_square = torch.bmm(
            trg_mask.unsqueeze(-1), node_mask.unsqueeze(1))

        trg_decoder_output = trg_embeddings
        for i, decoder in enumerate(self.decoder):
            trg_decoder_output, target_target_representations, \
                target_source_representations, target_source_attention = \
                decoder(
                    trg_decoder_output, trg_mask, trg_mask_square,
                    h_og, obs_mask_square, h_go, node_mask_square, i * 3 + 1,
                    len(self.decoder))  # batch x time x hid

        trg_decoder_output = self.tgt_word_prj(trg_decoder_output)
        trg_decoder_output = masked_softmax(
            trg_decoder_output, m=trg_mask.unsqueeze(-1), axis=-1)
        output = self.pointer_softmax(
            target_target_representations, target_source_representations,
            trg_decoder_output, trg_mask, target_source_attention,
            obs_mask, input_obs)

        return output

    def get_word_input(self, input_strings):
        word_list = [item.split() for item in input_strings]
        word_id_list = [[self._word_vocab[tok] for tok in tokens]
                        for tokens in word_list]
        input_word = pad_sequences(
            word_id_list, maxlen=max_len(word_id_list)).astype('int32')
        input_word = to_pt(input_word, self.device != 'cpu')
        return input_word

    def get_graph_node_name_input(self):
        res = copy.copy(self._node_vocab)
        input_node_name = self.get_word_input(res)  # num_node x words
        return input_node_name

    def get_graph_relation_name_input(self):
        res = copy.copy(self._relation_vocab)
        res = [item.replace("_", " ") for item in res]
        input_relation_name = self.get_word_input(res)  # num_node x words
        return input_relation_name

    def get_graph_relation_representations(self, relation_names_word_ids):
        # relation_names_word_ids: num_relation x num_word
        relation_name_embeddings, _mask = self.embed(
            relation_names_word_ids)  # num_relation x num_word x emb
        _mask = torch.sum(_mask, -1)  # num_relation
        relation_name_embeddings = torch.sum(
            relation_name_embeddings, 1)  # num_relation x hid
        tmp = torch.eq(_mask, 0).float()
        if relation_name_embeddings.is_cuda:
            tmp = tmp.cuda()
        _mask = _mask + tmp
        relation_name_embeddings = relation_name_embeddings / \
            _mask.unsqueeze(-1)
        relation_name_embeddings = relation_name_embeddings.unsqueeze(
            0)  # 1 x num_relation x emb

        relation_ids = np.arange(len(self._relation_vocab))  # num_relation
        relation_ids = to_pt(relation_ids,
                             cuda=relation_names_word_ids.is_cuda,
                             type='long').unsqueeze(0)  # 1 x num_relation
        relation_embeddings, _ = self.relation_embedding(
            relation_ids)  # 1 x num_relation x emb
        # 1 x num_relation x emb+emb
        relation_embeddings = torch.cat(
            [relation_name_embeddings, relation_embeddings], dim=-1)
        return relation_embeddings

    def get_graph_node_representations(self, node_names_word_ids):
        # node_names_word_ids: num_node x num_word
        node_name_embeddings, _mask = self.embed(
            node_names_word_ids)  # num_node x num_word x emb
        _mask = torch.sum(_mask, -1)  # num_node
        node_name_embeddings = torch.sum(
            node_name_embeddings, 1)  # num_node x hid
        tmp = torch.eq(_mask, 0).float()
        if node_name_embeddings.is_cuda:
            tmp = tmp.cuda()
        _mask = _mask + tmp
        node_name_embeddings = node_name_embeddings / _mask.unsqueeze(-1)
        node_name_embeddings = node_name_embeddings.unsqueeze(
            0)  # 1 x num_node x emb

        node_ids = np.arange(len(self._node_vocab))  # num_node
        node_ids = to_pt(node_ids,
                         cuda=node_names_word_ids.is_cuda,
                         type='long').unsqueeze(0)  # 1 x num_node
        node_embeddings, _ = self.node_embedding(
            node_ids)  # 1 x num_node x emb
        # 1 x num_node x emb+emb
        node_embeddings = torch.cat(
            [node_name_embeddings, node_embeddings], dim=-1)
        return node_embeddings

    def get_graph_adjacency_matrix(self, triplets):
        adj = np.zeros((len(triplets), len(self._relation_vocab), len(
            self._node_vocab), len(self._node_vocab)), dtype="float32")
        for b in range(len(triplets)):
            node_exists = set()
            for t in triplets[b]:
                node1, node2, relation = t
                assert node1 in self._node_vocab, \
                    node1 + " is not in node vocab"
                assert node2 in self._node_vocab, \
                    node2 + " is not in node vocab"
                assert relation in self._relation_vocab, \
                    relation + " is not in relation vocab"
                node1_id, node2_id, relation_id = \
                    self._node_vocab[node1], self._node_vocab[node2], \
                    self._relation_vocab[relation]
                adj[b][relation_id][node1_id][node2_id] = 1.0
                adj[b][relation_id + self.origin_relation_number][
                    node2_id][node1_id] = 1.0
                node_exists.add(node1_id)
                node_exists.add(node2_id)
            # self relation
            for node_id in list(node_exists):
                adj[b, -1, node_id, node_id] = 1.0
        adj = to_pt(adj, self.device != 'cpu', type='float')
        return adj

    def encode_graph(self, graph_input):
        # batch x num_node x emb+emb
        if self.node_embeddings is None:
            node_names_word_ids = self.get_graph_node_name_input()
            self.node_embeddings = self.get_graph_node_representations(
                node_names_word_ids)  # 1 x num_node x emb+emb

        if self.relation_embeddings is None:
            relation_names_word_ids = self.get_graph_relation_name_input()
            self.relation_embeddings = self.get_graph_relation_representations(
                relation_names_word_ids)  # 1 x num_node x emb+emb
        if isinstance(graph_input, list):
            input_adjacency_matrices = self.get_graph_adjacency_matrix(
                graph_input)
        elif isinstance(graph_input, torch.Tensor):
            input_adjacency_matrices = graph_input
        else:
            raise NotImplementedError
        input_adjacency_matrices = input_adjacency_matrices.to(self.device)
        node_embeddings = self.node_embeddings.repeat(
            input_adjacency_matrices.size(0), 1, 1)
        # batch x num_relation x emb+emb
        relation_embeddings = self.relation_embeddings.repeat(
            input_adjacency_matrices.size(0), 1, 1)
        # batch x num_node x enc
        node_encoding_sequence = self.rgcns(
            node_embeddings, relation_embeddings, input_adjacency_matrices)

        if self.use_ground_truth_graph:
            node_mask = torch.ones(node_encoding_sequence.size(
                0), node_encoding_sequence.size(1))  # batch x num_node
            if node_encoding_sequence.is_cuda:
                node_mask = node_mask.cuda()
        else:
            # batch x num_node x num_node
            node_mask = torch.sum(input_adjacency_matrices[:, :-1, :, :], 1)
            node_mask = torch.sum(node_mask, -1) + \
                torch.sum(node_mask, -2)  # batch x num_node
            node_mask = torch.gt(node_mask, 0).float()
            node_encoding_sequence = node_encoding_sequence * \
                node_mask.unsqueeze(-1)

        if self.self_attention is not None:
            mask_squared = torch.bmm(
                node_mask.unsqueeze(-1), node_mask.unsqueeze(1))
            node_encoding_sequence, _ = self.self_attention(
                node_encoding_sequence, mask_squared, node_encoding_sequence,
                node_encoding_sequence)

        return node_encoding_sequence, node_mask

    def hidden_to_adjacency_matrix(self, hidden, batch_size):
        num_node = len(self._node_vocab)
        num_relation = len(self._relation_vocab)
        if hidden is None:
            adjacency_matrix = torch.zeros(
                batch_size, num_relation, num_node, num_node)
            adjacency_matrix = adjacency_matrix.cuda()
        else:
            adjacency_matrix = torch.tanh(self.obs_gen_linear_2(
                F.relu(self.obs_gen_linear_1(
                    hidden)))).view(batch_size,
                                    int(num_relation / 2), num_node, num_node)
            adjacency_matrix = adjacency_matrix.repeat(1, 2, 1, 1)
            for i in range(int(num_relation / 2)):
                adjacency_matrix[:, int(
                    num_relation / 2) + i] = \
                    adjacency_matrix[:, i].permute(0, 2, 1)
        return adjacency_matrix

    @torch.no_grad()
    def forward(
            self, observations: List[str],
            graph: List[Tuple[str, str, str]],
            actions: List[str] = None,
            infos: Dict[str, Any] = None) -> List[Tuple[str, str, str]]:
        if self.use_ground_truth_graph:
            if infos is None:
                raise ValueError(
                    "Can't have 'None' infos for ground truth graph")
            if 'facts' not in infos.keys():
                raise ValueError(
                    "Must have 'facts' as infos key. Set EnvInfos(facts=True)")
            return infos['facts']
        elif self.real_valued_graph:
            if self.prev_graph_hidden_state is not None:
                prev_graph_hidden_state = self.prev_graph_hidden_state.detach()
            # TE-encode
            input_obs = self.get_word_input(observations)
            prev_action_word_ids = self.get_word_input(actions)
            prev_action_encoding_sequence, prev_action_mask = self.encode_text(
                prev_action_word_ids)
            obs_encoding_sequence, obs_mask = self.encode_text(input_obs)
            prev_adjacency_matrix = self.hidden_to_adjacency_matrix(
                prev_graph_hidden_state, batch_size=len(
                    observations))
            node_encoding_sequence, node_mask = self.encode_graph(
                prev_adjacency_matrix)

            h_ag = self.obs_gen_attention(
                prev_action_encoding_sequence,
                node_encoding_sequence, prev_action_mask, node_mask)
            h_ga = self.obs_gen_attention(
                node_encoding_sequence, prev_action_encoding_sequence,
                node_mask, prev_action_mask)

            h_ag = self.obs_gen_attention_prj(
                h_ag)
            h_ga = self.obs_gen_attention_prj(
                h_ga)

            h_og = self.obs_gen_attention(
                obs_encoding_sequence, node_encoding_sequence, obs_mask,
                node_mask)
            h_go = self.obs_gen_attention(
                node_encoding_sequence, obs_encoding_sequence, node_mask,
                obs_mask)

            h_og = self.obs_gen_attention_prj(
                h_og)  # bs X len X block_hidden_dim
            h_go = self.obs_gen_attention_prj(
                h_go)  # bs X len X block_hidden_dim

            ave_h_go = masked_mean(h_go, m=node_mask, dim=1)
            ave_h_og = masked_mean(h_og, m=obs_mask, dim=1)
            ave_h_ga = masked_mean(h_ga, m=node_mask, dim=1)
            ave_h_ag = masked_mean(h_ag, m=prev_action_mask, dim=1)

            rnn_input = self.obs_gen_attention_to_rnn_input(
                torch.cat([ave_h_go, ave_h_og, ave_h_ga, ave_h_ag], dim=1))
            rnn_input = torch.tanh(rnn_input)  # batch x block_hidden_dim
            h_t = self.obs_gen_graph_rnncell(
                rnn_input, prev_graph_hidden_state) if \
                prev_graph_hidden_state is not None else \
                self.obs_gen_graph_rnncell(rnn_input)
            current_adjacency_matrix = self.hidden_to_adjacency_matrix(
                h_t, batch_size=len(
                    observations))
            del self.prev_graph_hidden_state
            self.prev_graph_hidden_state = h_t.detach()
            self.curr_mat = current_adjacency_matrix.detach().cpu()
            return self.curr_mat
        else:
            batch_size = len(observations)
            # encode
            input_obs = self.get_word_input(observations)
            obs_encoding_sequence, obs_mask = self.encode_text(input_obs)
            node_encoding_sequence, node_mask = self.encode_graph(
                graph)

            h_og = self.cmd_gen_attention(
                obs_encoding_sequence, node_encoding_sequence, obs_mask,
                node_mask)
            h_go = self.cmd_gen_attention(
                node_encoding_sequence, obs_encoding_sequence, node_mask,
                obs_mask)
            h_og = self.cmd_gen_attention_prj(h_og)
            h_go = self.cmd_gen_attention_prj(h_go)

            # step 2, greedy generation
            # decode
            input_target_token_list = [["<bos>"] for i in range(batch_size)]
            eos = np.zeros(batch_size)
            for _ in range(self.max_target_length):

                input_target = self.tokenize(
                    [" ".join(item) for item in input_target_token_list])
                # batch x time x vocab
                pred = self.decode(input_target, h_og,
                                   obs_mask, h_go, node_mask, input_obs)
                # pointer softmax
                pred = to_np(pred[:, -1])  # batch x vocab
                pred = np.argmax(pred, -1)  # batch
                for b in range(batch_size):
                    new_stuff = [self._word_vocab[int(pred[b])]
                                 ] if eos[b] == 0 else list()
                    input_target_token_list[b] = input_target_token_list[
                        b] + new_stuff
                    if pred[b] == self._word_vocab["<eos>"]:
                        eos[b] = 1
                if np.sum(eos) == batch_size:
                    break
            return [" ".join(item[1:]) for item in input_target_token_list]

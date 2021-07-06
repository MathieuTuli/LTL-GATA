from typing import List, Tuple
from argparse import Namespace

import pdb

from torch.autograd import Variable

import torch

from model.layers import Embedding, EncoderBlock, SelfAttention
from utils import max_len, to_pt, pad_sequences


class SimpleMLP(torch.nn.Module):
    def __init__(self,
                 word_embedding_size: int,
                 action_net_hidden_size: int, **kwargs):
        super(SimpleMLP, self).__init__(**kwargs)
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(word_embedding_size, word_embedding_size),
            torch.nn.ReLU(),
            torch.nn.Linear(word_embedding_size, action_net_hidden_size),
        )

    def forward(self, inputs: torch.Tensor,) -> torch.Tensor:
        out = self.layers(inputs)
        return out


class SimpleLSTM(torch.nn.Module):
    def __init__(self,
                 word_embedding_size: int,
                 hidden_size: int,
                 num_layers: int,
                 action_net_hidden_size: int,
                 **kwargs):
        super(SimpleLSTM, self).__init__(**kwargs)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = torch.nn.LSTM(
            input_size=word_embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True)
        self.head = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, action_net_hidden_size)
        )
        self._dummy = torch.nn.Parameter(torch.empty(0))

    @property
    def device(self) -> str:
        return self._dummy.device

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, length, hidden = inputs.shape
        h_0 = Variable(torch.zeros(self.num_layers, batch_size,
                       self.hidden_size)).to(device=self.device)
        c_0 = Variable(torch.zeros(self.num_layers, batch_size,
                       self.hidden_size)).to(device=self.device)
        output, (hn, cn) = self.lstm(inputs, (h_0, c_0))
        hn = hn.view(-1, self.hidden_size)
        out = self.head(hn)
        return out


class TextEncoder(torch.nn.Module):
    def __init__(self,
                 config: Namespace,
                 vocab: List[str],
                 **kwargs) -> None:
        super(TextEncoder, self).__init__(**kwargs)
        self._dummy = torch.nn.Parameter(torch.empty(0))
        for k, v in vars(config).items():
            setattr(self, k, v)
        self.load_vocab(vocab)
        if self.use_pretrained_lm:
            from model.pretrained_lm import get_model_tokenizer
            self.lm, self.tokenizer = get_model_tokenizer(
                self.pretrained_lm_name,
                self.pretrained_lm_checkpoint)
            # ----- Add Tokens -----
            vocab += list(self.tokenizer.special_tokens_map.values())

            # tokens = list(dict.fromkeys(tokens))
            add_tokens = [token for token in vocab.tokens if token not in
                          self.tokenizer.vocab]
            self.tokenizer.add_tokens(add_tokens)
            self.lm.resize_token_embeddings(len(self.tokenizer))
            if self.pretrained_lm_name == 'bert':
                embedding_dim = 768
            self.encoder = torch.nn.Linear(embedding_dim,
                                           self.action_net_hidden_size)
            # ----- Add Tokens -----
            # ----- Delete Unused Tokens -----
            # count = 0
            # word_embedding_idxs = list()
            # vocab = list(self.tokenizer.get_vocab().items())
            # for tok, idx in vocab:
            #     if tok not in tokens:
            #         del self.tokenizer.vocab[tok]
            #     else:
            #         self.tokenizer.vocab[tok] = count
            #         word_embedding_idxs.append(idx)
            #         count += 1
            # self.tokenizer.added_tokens_encoder.clear()
            # self.tokenizer.added_tokens_decoder.clear()
            # assert len(self.tokenizer) == len(tokens)
            # word_embeddings = self.lm.embeddings.word_embeddings.weight[
            #     word_embedding_idxs]
            # self.lm.resize_token_embeddings(len(self.tokenizer))
            # self.lm.embeddings.word_embeddings.weight.data = word_embeddings
            # ----- Delete Unused Tokens -----

        else:
            self.word_embedding = Embedding(
                embedding_size=self.word_embedding_size,
                vocab_size=len(vocab),
                id2word=vocab.tokens,
                dropout_rate=0.,
                load_pretrained=True,
                trainable=False,
                embedding_oov_init="random" if not
                self.one_hot_encoding else 'onehot',
                pretrained_embedding_path=self.pretrained_embedding_path)
            self.word_embedding_prj = torch.nn.Linear(
                self.word_embedding_size, self.action_net_hidden_size,
                bias=False)
            if self.self_attention:
                self.self_attention = SelfAttention(
                    self.action_net_hidden_size, self.n_heads, 0.)
            else:
                self.self_attention = None
            if config.lstm_backbone:
                self.encoder = SimpleLSTM(
                    word_embedding_size=self.word_embedding_size,
                    hidden_size=128,
                    num_layers=self.num_encoders,
                    action_net_hidden_size=self.action_net_hidden_size)
            elif config.mlp_backbone:
                self.encoder = SimpleMLP(
                    word_embedding_size=self.word_embedding_size,
                    action_net_hidden_size=self.action_net_hidden_size)
            else:
                self.encoder = torch.nn.ModuleList([
                    EncoderBlock(
                        conv_num=self.encoder_conv_num,
                        ch_num=self.action_net_hidden_size, k=5,
                        block_hidden_dim=self.action_net_hidden_size,
                        n_head=self.n_heads, dropout=0.,)
                    for _ in range(self.num_encoders)])
                self.num_encoders = self.num_encoders
            if not self.trainable:
                for param in self.parameters():
                    param.requires_grad = False
            if self.mlm_loss:
                self.mlm_head = torch.nn.Sequential(
                    torch.nn.Linear(self.action_net_hidden_size,
                                    self.action_net_hidden_size, bias=True),
                    torch.nn.LayerNorm(self.action_net_hidden_size,
                                       eps=1e-12, elementwise_affine=True),
                    torch.nn.Linear(self.action_net_hidden_size,
                                    len(self.vocab), bias=True),
                )
        if False:
            self.inverse_dynamics = torch.nn.Sequential(
                torch.nn.Linear(self.action_net_hidden_size * 2,
                                self.action_net_hidden_size * 2),
                torch.nn.ReLU(),
                torch.nn.Linear(self.action_net_hidden_size * 2,
                                self.action_net_hidden_size)
            )

    def load_vocab(self, vocab) -> None:
        self.vocab = vocab

    @property
    def device(self) -> str:
        return self._dummy.device

    def compute_inverse_dynamics_loss(self, obs: List[str],
                                      next_obs: List[str],
                                      actions: List[str],
                                      hat: bool = True):
        encoded_obs, obs_mask = self.forward(obs)
        encoded_next_obs, next_obs_mask = self.forward(next_obs)
        encoded_obs = torch.sum(encoded_obs, dim=1)
        encoded_next_obs = torch.sum(encoded_next_obs, dim=1)
        actions_inv = self.inverse_dynamics(torch.cat((
            encoded_obs, encoded_next_obs - encoded_obs), dim=1))
        return None

    def compute_mlm_loss(self, text: List[str]):
        if not self.mlm_loss:
            return None
        inputs = self.tokenize(text)
        labels = torch.clone(inputs)
        rand = torch.rand(inputs.shape, device=self.device)
        masking_mask = (rand < 0.15) * (inputs != self.vocab.pad_token_id)
        inputs[torch.where(masking_mask)] = self.vocab.mask_token_id
        sequence_output, _ = self.forward(inputs, compute_word_ids=False)
        predictions = self.mlm_head(sequence_output)
        # print([self.vocab.tokens[idx]
        #       for idx in inputs[0] if idx != self.vocab.pad_token_id])
        # print([self.vocab.tokens[idx]
        #       for idx in torch.argmax(predictions[0], dim=1)
        #       if idx != self.vocab.pad_token_id])
        loss_fcn = torch.nn.CrossEntropyLoss()
        labels[torch.where(labels == self.vocab.pad_token_id)] = -100
        return loss_fcn(predictions.view((-1, len(self.vocab))),
                        labels.view(-1))

    def tokenize(self, text: List[str]) -> torch.Tensor:
        word_list = [item.split() for item in text]
        word_id_list = [[self.vocab[tok] for tok in tokens]
                        for tokens in word_list]
        input_word = pad_sequences(
            word_id_list, maxlen=max_len(word_id_list)).astype('int32')
        input_word_ids = to_pt(input_word, True)
        return input_word_ids

    def forward(self, text: List[str],
                compute_word_ids: bool = True,
                position_encoding_method: str = 'cossine',
                trees=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        @arguments:
          text: list of strings of shape [batch-size]
          pad: number of additional padding items to append
            to text list of strings for batch reasons

        @returns:
          encodings: encoded text, Tensor, size
            [batch-size, sequence-length, embed-size]
          mask: mask of size
            [batch-size, sequence-length]
        """
        if self.use_pretrained_lm:
            inputs = self.tokenizer.batch_encode_plus(
                text, padding=True, add_special_tokens=True,
                return_tensors='pt')
            for k, v in inputs.items():
                inputs[k] = v.to(self.device)
            outputs = self.lm(**inputs)
            outputs = outputs.last_hidden_state
            outputs = self.encoder(outputs)
            return outputs, inputs.attention_mask
        else:
            if compute_word_ids:
                input_word_ids = self.tokenize(text)
            else:
                input_word_ids = text
            embeddings, mask = self.word_embedding(
                input_word_ids)  # batch x time x emb
            squared_mask = torch.bmm(mask.unsqueeze(-1), mask.unsqueeze(1))
            position_encoding = None
            if position_encoding_method == 'tree':
                raise NotImplementedError

            if self.lstm_backbone:
                encoded_text = embeddings
                encoded_text = self.encoder(encoded_text)
            elif self.mlp_backbone:
                encoded_text = torch.sum(embeddings, dim=1)
                encoded_text = self.encoder(encoded_text)
            else:
                embeddings = self.word_embedding_prj(embeddings)
                embeddings = embeddings * \
                    mask.unsqueeze(-1)  # batch x time x hid
                encoded_text = embeddings
                for i in range(self.num_encoders):
                    # batch x time x enc
                    encoded_text = self.encoder[i](
                        encoded_text, squared_mask, i * (
                            self.encoder_conv_num + 2) + 1, self.num_encoders,
                        position_encoding_method=position_encoding_method,
                        position_encoding=position_encoding)
            if self.self_attention is not None:
                mask_squared = torch.bmm(mask.unsqueeze(dim=-1),
                                         mask.unsqueeze(dim=1))
                encoded_text, _ = self.self_attention(
                    encoded_text, mask_squared, encoded_text, encoded_text)
            return encoded_text, mask


class BagOfWords(torch.nn.Module):
    def __init__(self,
                 config: Namespace,
                 vocab: List[str],
                 **kwargs) -> None:
        super(BagOfWords, self).__init__(**kwargs)
        self._dummy = torch.nn.Parameter(torch.empty(0))
        for k, v in vars(config).items():
            setattr(self, k, v)
        self.load_vocab(vocab)

    def load_vocab(self, vocab) -> None:
        self.vocab = vocab

    @property
    def device(self) -> str:
        return self._dummy.device

    def tokenize(self, text: List[str]) -> torch.Tensor:
        word_list = [item.split() for item in text]
        word_id_list = [[self.vocab[tok] for tok in tokens]
                        for tokens in word_list]
        input_word = pad_sequences(
            word_id_list, maxlen=max_len(word_id_list)).astype('int32')
        input_word_ids = to_pt(input_word, True)
        return input_word_ids

    def forward(self, text: List[str],
                compute_word_ids: bool = True,
                position_encoding_method: str = 'cossine',
                trees=None) -> Tuple[torch.Tensor, torch.Tensor]:
        ...

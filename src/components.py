from __future__ import annotations

from typing import NamedTuple, List, Dict, Union
from dataclasses import dataclass
from enum import Enum
from copy import copy

import logging

import numpy as np

from state import State

Actions = List[str]

logger = logging.getLogger()


@dataclass
class ResultsCSVField:
    time: str = None
    episode_no: int = None
    dqn_loss: float = None
    train_game_points: float = None
    train_normalized_game_points: float = None
    train_rewards: float = None
    train_normalized_rewards: float = None
    train_game_rewards: float = None
    train_steps: float = None
    train_success: float = None
    eval_game_points: float = None
    eval_normalized_game_points: float = None
    eval_rewards: float = None
    eval_normalized_rewards: float = None
    eval_steps: float = None
    eval_success: float = None

    def keys(self):
        return list(vars(self).keys())

    def values(self):
        return list(vars(self).values())


class Vocabulary:
    def __init__(self, vocab: List[str], name: str = 'Vocabulary',
                 original_only: bool = False) -> None:
        self.trash = set()
        self.original_tokens = copy(vocab)
        if not original_only:
            if '<unk>' not in vocab and '[UNK]' not in vocab:
                vocab += ['<unk>']
            if '<mask>' in vocab:
                self.mask_token = '<mask>'
            elif '[MASK]' in vocab:
                self.mask_token = '[MASK]'
            else:
                vocab += ['<mask>']
                self.mask_token = '<mask>'
            if '<pad>' in vocab:
                self.pad_token = '<pad>'
            elif '[PAD]' in vocab:
                self.pad_token = '[PAD]'
            else:
                vocab += ['<pad>']
                self.pad_token = '<pad>'
        self.name = name
        self.tokens = vocab
        self.tokens = list(dict.fromkeys(self.tokens))
        self.map = self.build_map(vocab)

    @property
    def mask_token_id(self) -> int:
        return self.map[self.mask_token]

    @property
    def pad_token_id(self) -> int:
        return self.map[self.pad_token]

    def build_map(self, vocab: List[str]) -> Dict[str, int]:
        return {tok: i for i, tok in enumerate(vocab)}

    def __add__(self, other: Union[Vocabulary, List]) -> Vocabulary:
        if isinstance(other, list):
            self.tokens += other
        elif isinstance(other, Vocabulary):
            self.tokens += other.tokens
        else:
            raise ValueError("Other must be of type Vocabulary or List")
        self.tokens = list(dict.fromkeys(self.tokens))
        self.map = self.build_map(self.tokens)
        return self

    def __eq__(self, other) -> bool:
        return self.tokens == self.tokens and self.map == self.map

    def __str__(self) -> str:
        return self.name

    def __len__(self) -> int:
        return len(self.map)

    def __iter__(self):
        return iter(self.map)

    def __contains__(self, tok) -> bool:
        return tok in self.map

    def __getitem__(self, tok) -> int:
        if isinstance(tok, int) or isinstance(tok, np.int64):
            return self.tokens[tok]
        if tok not in self.map:
            if tok not in self.trash:
                logger.warning(f"Token '{tok}' not found in vocab: {self}")
                self.trash.update([tok])
            if '<unk>' not in self.map:
                return self.map['-']
            else:
                return self.map['<unk>']
        return self.map[tok]


class Sample(NamedTuple):
    step: int
    action: int
    done: float
    state: State
    reward: float
    is_final: bool
    admissible_actions: Actions


class SampleMetadata(NamedTuple):
    index: int
    weight: float
    priority: float
    probability: float


class AgentModes(Enum):
    eval = 0
    train = 1
    test = 2

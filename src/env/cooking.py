from typing import Tuple, List, Dict, Any, Union
from copy import copy, deepcopy
from pathlib import PosixPath

import logging
import random
import glob
import pdb
import re
import os

from textworld.gym.envs.textworld_batch import TextworldBatchGymEnv as TWEnv
from textworld.logic import Proposition, Variable
from textworld import EnvInfos

from logic import Proposition as LProposition, prune_actions
from ltl import LTL

import textworld.gym
import numpy as np
import codecs
import spacy
import gym

from components import Actions, Vocabulary
logger = logging.getLogger()


def preprocess_facts(facts, mapping, tokenizer=None) -> str:
    if isinstance(facts, Proposition):
        if facts.name in mapping:
            facts.name = mapping[facts.name]
        for arg in facts.arguments:
            if arg.name in mapping:
                arg.name = mapping[arg.name]
        return copy(facts)
    for i, prop in enumerate(facts):
        if prop[0].name in mapping:
            prop[0].name = mapping[prop[0].name]
        for arg in prop[0].arguments:
            if arg.name in mapping:
                arg.name = mapping[arg.name]
        facts = (copy(prop),)
    return facts


def preprocess_string(string: str, mapping, tokenizer=None) -> str:
    if string is None:
        return "nothing"
    string = string.replace("\n", ' ')
    if "$$$$$$$" in string:
        string = string.split("$$$$$$$")[-1]
    string = re.sub(' +', ' ', string).strip()
    if len(string) == 0:
        return "nothing"
    if tokenizer is not None:
        string = " ".join([t.text for t in tokenizer(string)])
        if 'frosted - glass' in string:
            string = string.replace('frosted - glass', 'frosted-glass')
    string = string.lower()
    for tok, tokto in mapping.items():
        string = string.replace(tok, tokto)

    return string


class RecipeWrappedEnv:
    """
    This is bad practice! Fake Env wrapper
    """

    def __init__(self, env: TWEnv, vocab_dir: PosixPath,
                 real_valued_graph: bool,
                 randomized_nouns_verbs: bool,
                 train: bool,
                 prune: bool = False,
                 strip_instructions: bool = False,
                 eleven: bool = False) -> None:
        # super(ModifiedEnv, self).__init__(**kwargs)
        self.env = env
        self.tokenizer = spacy.load('en_core_web_sm', disable=[
                                    'ner', 'parser', 'tagger'])
        self.real_valued_graph = real_valued_graph
        self.word_vocab, self.ltl_vocab, self.relation_vocab, \
            self.node_vocab,  self.action_vocab = self.load_vocabs(vocab_dir,
                                                                   eleven)
        self.prev_adm = None
        self.mapping = dict()
        self.tokens = dict()
        self.tok_idxs = dict()
        self.prune = prune
        self.strip_instructions = strip_instructions
        if randomized_nouns_verbs and train:
            self.tokens = {
                'ingredients': (
                    'banana',
                    'block of cheese',
                    'carrot',
                    'orange bell pepper',
                    'pork chop',
                    'purple potato',
                    'red apple',
                    'red hot pepper',
                    'red onion',
                    'red potato',
                    'white onion',
                    'yellow bell pepper',
                    'yellow potato',
                ),
                'cooking_methods': (
                    # cook I with C
                    'stove',
                    'oven',
                ),
                'cooking_methods_facts': (
                    'fried',
                    'roasted',
                ),
                'preparation_methods': (
                    # P I with knife
                    'chop',
                    'dice',
                    'slice'
                ),
                'preparation_methods_facts': (
                    'chopped',
                    'diced',
                    'sliced'
                ),
            }
        self.randomized_nouns_verbs = randomized_nouns_verbs and train

    def load_vocabs(self, vocab_dir: PosixPath,
                    eleven: bool) -> Tuple[
            Vocabulary, Vocabulary, Vocabulary, Vocabulary]:
        word_vocab = list()
        with codecs.open(str(vocab_dir / 'word_vocab.txt'),
                         mode='r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if self.real_valued_graph and line.strip() == 'examined':
                    continue
                word_vocab.append(line.strip())
        if eleven and False:
            ings = ['green_apple', 'green_bell_pepper', 'green_hot_pepper',
                    'red_tuna', 'white_tuna',
                    'lettuce', 'tomato', 'yellow_apple', 'yellow_onion']
            ltl_vocab = ['_'.join([ing, i]) for ing in ings for i in [
                'in_player', 'is_fried', 'is_roasted', 'is_diced',
                'is_sliced', 'is_chopped']]
        else:
            ltl_vocab = list()
        with codecs.open(str(vocab_dir / 'ltl_vocab.txt'),
                         mode='r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                ltl_vocab.append(line.strip())
        relation_vocab = list()
        with codecs.open(str(vocab_dir / "relation_vocab.txt"),
                         mode='r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                relation_vocab.append(line.strip().lower())
        # add reverse relations
        for i in range(len(relation_vocab)):
            relation_vocab.append(relation_vocab[i] + "_reverse")
        # if not use_ground_truth_graph:
        if not self.real_valued_graph:
            relation_vocab.append('self')
        node_vocab = list()
        with codecs.open(str(vocab_dir / 'node_vocab.txt'),
                         mode='r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if self.real_valued_graph and line.strip() == 'examined':
                    continue
                node_vocab.append(line.strip().lower())
        if eleven and False:
            action_vocab = ['tomato', 'green', 'lettuce', 'tuna']
        else:
            action_vocab = list()
        with codecs.open(str(vocab_dir / 'action_vocab.txt'),
                         mode='r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                action_vocab.append(line.strip().lower())
        return (Vocabulary(word_vocab, 'word-vocab'),
                Vocabulary(ltl_vocab, 'ltl-vocab'),
                Vocabulary(relation_vocab, 'relation-vocab',
                           original_only=True),
                Vocabulary(node_vocab, 'node-vocab', original_only=True),
                Vocabulary(action_vocab, 'action-vocab', original_only=True))

    def process_obs_infos(self, obs: List[str], infos: Dict[str, List[Any]],
                          ltl_formulas: List[LTL],
                          ) -> Tuple[List[str], Dict[str, List[Any]]]:
        for commands in infos['admissible_commands']:
            cmds = copy(commands)
            for i, cmd in enumerate(cmds):
                if cmd != 'examine cookbook' and cmd.split()[0] in {
                        'examine', 'look', 'inventory'}:
                    commands.remove(cmd)
        if self.prune:
            infos['admissible_commands'] = \
                [prune_actions(
                    ltl if isinstance(ltl, str) else
                    ltl.tokenize(), actions) for ltl, actions in
                 zip(ltl_formulas, infos['admissible_commands'])]
            # try:
            #     commands.remove('take cookbook from counter')
            # except Exception:
            #     pass
        # TODO this is inefficient clean up
        self.real_adm = deepcopy(infos['admissible_commands'])
        obs = [preprocess_string(
            o, self.mapping, self.tokenizer) for o in obs]

        if self.strip_instructions:
            for i, o in enumerate(obs):
                if 'you are hungry !' in o:
                    obs[i] = o.replace(
                        "you are hungry ! let 's cook a delicious meal . check the cookbook in the kitchen for the recipe . once done , enjoy your meal !", "")
                elif 'you open the copy' in o:
                    obs[i] = 'you open the copy of " cooking : a modern approach ( 3rd ed . ) "'
                elif 'you need to take the knife first' in o:
                    obs[i] = "you can ' t do that"

        infos['admissible_commands'] = [[
            preprocess_string(a, self.mapping, self.tokenizer) for a in
            commands] for commands in infos['admissible_commands']]
        if self.randomized_nouns_verbs:
            infos['win_facts'] = [[
                preprocess_facts(f, self.mapping) for f in
                deepcopy(facts)] for facts in infos['win_facts']]
            infos['facts'] = [[
                preprocess_facts(f, self.mapping) for f in
                deepcopy(facts)] for facts in infos['facts']]
        return obs, infos

    def seed(self, seed: int) -> None:
        self.env.seed(seed)

    def step(self, actions: List[Union[str, int]],
             ltl_formulas: List[LTL],
             ) -> Tuple[List[str],
                        List[float],
                        List[bool],
                        Dict[str, List[Any]]]:
        if not isinstance(actions[0], str):
            str_actions = [cmds[i] for cmds, i in zip(self.real_adm, actions)]
            actions = str_actions
        obs, dones, scores, infos = self.env.step(actions)
        obs, infos = self.process_obs_infos(obs, infos, ltl_formulas)
        if True:
            infos = self.update_infos(actions, infos)
        return obs, dones, scores, infos

    def update_infos(self, actions: Actions,
                     infos: Dict[str, Any]) -> Dict[str, Any]:
        for i, action in enumerate(actions):
            if action == 'examine cookbook':
                infos['facts'][i].append(
                    LProposition(name='examined',
                                 arguments=[Variable('cookbook', type='o')],
                                 seperator='_'))
        return infos

    def reset(self) -> Tuple[List[str], Dict[str, List[Any]]]:
        obs, infos = self.env.reset()
        if self.randomized_nouns_verbs:
            self.mapping = dict()
            idxs = None
            for k, toks in self.tokens.items():
                # this will make it use the previous idxs
                # to match
                if 'facts' not in k:
                    idxs = random.sample(
                        np.arange(len(toks)).tolist(), len(toks))
                for i, tok in enumerate(toks):
                    self.mapping[tok] = toks[idxs[i]]
            self.prev_adm = deepcopy(infos['admissible_commands'])
        return self.process_obs_infos(obs, infos, [''] * len(obs))


def get_cooking_game_env(data_dir: PosixPath,
                         vocab_dir: PosixPath,
                         difficulty_level: int,
                         requested_infos: EnvInfos,
                         max_episode_steps: int,
                         batch_size: int,
                         real_valued_graph: bool,
                         randomized_nouns_verbs: bool,
                         all_games: bool = False,
                         split: str = 'train',
                         training_size: int = 20,
                         game_limit: int = -1,
                         prune: bool = False,
                         strip_instructions: bool = False) -> Tuple[None, int]:
    splits = {'train', 'valid', 'test'}
    assert difficulty_level in {1, 2, 3, 4, 5,
                                6, 7, 8, 9, 10, 11, 12, 13, 99,
                                'r', 'mixed'}
    assert split in splits
    assert training_size in {1, 20, 100}
    if all_games:
        assert split == 'train'
    logger.info(f'{split} : Batch Size : {batch_size}')
    logger.info(f'{split} : Training Size : {training_size}')
    if game_limit > 0:
        logger.info(f'{split} : Game Limit : {game_limit}')
    else:
        logger.info(f'{split} : Game Limit : {training_size}')
    logger.info(f'{split} : Difficulty level : {difficulty_level}')

    # training games
    if not all_games:
        splits = [split]
    game_file_names = []
    for split in splits:
        if split == 'train':
            split += '_'
            tsize = training_size
        else:
            tsize = ''
        if difficulty_level == 'r':
            diffs = [3, 7, 5, 9]
            max_games = 25
        if difficulty_level == 'mixed':
            diffs = [1, 3, 7, 11]
            max_games = 25
        else:
            diffs = [difficulty_level]
            max_games = training_size
        for difficulty_level in diffs:
            game_path = f"{data_dir}/{split}" + \
                f"{tsize}/difficulty_level_{difficulty_level}"
            if os.path.isdir(game_path):
                game_file_names += glob.glob(os.path.join(game_path, "*.z8"))[
                    :max_games]
            else:
                game_file_names.append(game_path)

    if game_limit > 0:
        game_file_names = game_file_names[:game_limit]
    env_id = textworld.gym.register_games(
        sorted(game_file_names), request_infos=requested_infos,
        max_episode_steps=max_episode_steps, batch_size=batch_size,
        name="training" if split == 'train' else 'eval',
        asynchronous=False, auto_reset=False)
    env = gym.make(env_id)
    env = RecipeWrappedEnv(env, vocab_dir, real_valued_graph,
                           randomized_nouns_verbs,
                           train=split == 'train',
                           prune=prune,
                           strip_instructions=strip_instructions,
                           eleven=difficulty_level in {11, 'mixed'})
    num_game = len(game_file_names)
    return env, num_game

from __future__ import annotations

from typing import List, Dict, Any, Union, Tuple, Deque
from pathlib import Path, PosixPath
from argparse import Namespace

import pickle

from logic import Variable, Proposition

import numpy as np
import torch
import yaml

MISSING_WORDS = set()
CONSTANT_NAMES = {"P": "player", "I": "player",
                  "ingredient": None, "slot": None, "RECIPE": "cookbook"}


def proposition_to_triplet(proposition: Proposition) -> Proposition:
    if len(proposition.names) == 1:
        return (get_variable_name(proposition.names[0]),
                proposition.name, 'is')
    return (get_variable_name(proposition.names[0]),
            get_variable_name(proposition.names[1]),
            proposition.name)


def triplet_to_proposition(triplet: Tuple[str, str, str],
                           seperator: str) -> Proposition:
    if triplet[-1] == 'is':
        Proposition(name=triplet[1], arguments=[
                    Variable(triplet[0])], seperator=seperator)
    return Proposition(name=triplet[2], arguments=[
        Variable(triplet[0]), Variable(triplet[1])], seperator=seperator)


def rename_variables(
        item: Union[Variable, Proposition]) -> Union[Variable, Proposition]:
    if isinstance(item, Variable):
        item.name = get_variable_name(item.name)
        return item
    if isinstance(item, Proposition):
        new_args = list()
        for var in item.arguments:
            var.name = get_variable_name(var.name)
            new_args.append(var)
        item.arguments = tuple(new_args)
        return item
    raise ValueError(
        f"Unknown item type {type(item)}. " +
        "Must be one of {Variable, Proposition}")


def max_len(list_of_list):
    if len(list_of_list) == 0:
        return 0
    return max(map(len, list_of_list))


def pad_sequences(sequences, maxlen=None, dtype='int32', value=0.):
    '''
    Partially borrowed from Keras
    # Arguments
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        value: float, value to pad the sequences to the desired value.
    # Returns
        x: numpy array with dimensions (number_of_sequences, maxlen)
    '''
    if isinstance(sequences, np.ndarray):
        return sequences
    lengths = [len(s) for s in sequences]
    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)
    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break
    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        # pre truncating
        trunc = s[-maxlen:]
        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))
        # post padding
        x[idx, :len(trunc)] = trunc
    return x


def _word_to_id(word, word2id):
    try:
        return word2id[word]
    except KeyError:
        key = word + "_" + str(len(word2id))
        if key not in MISSING_WORDS:
            print("Warning... %s is not in vocab, vocab size is %d..." %
                  (word, len(word2id)))
            # print(word)
            # print(word2id)
            # raise
            MISSING_WORDS.add(key)
            # with open("missing_words.txt", 'a+') as outfile:
            #     outfile.write(key + '\n')
            #     outfile.flush()
        return word2id['<unk>']  # actually just 1


def _words_to_ids(words, word2id):
    return [_word_to_id(word, word2id) for word in words]


def get_variable_name(name: str) -> str:
    return CONSTANT_NAMES[name] if name in CONSTANT_NAMES else name


def load_config(config_file: Path, params: List[str]) -> Namespace:
    assert Path(config_file).exists(), \
        f"Could not find config file {config_file}"
    with open(config_file) as reader:
        config = yaml.safe_load(reader)
    # Parse overriden params.
    for param in params:
        fqn_key, value = param.split("=")
        entry_to_change = config
        keys = fqn_key.split(".")
        for k in keys[:-1]:
            entry_to_change = entry_to_change[k]
        entry_to_change[keys[-1]] = yaml.load(value)
    # print(config)
    config = Namespace(**config)
    for k, v in config.__dict__.items():
        if isinstance(v, dict):
            config.__dict__[k] = Namespace(**v)

    config.graph_updater.checkpoint = Path(
        config.graph_updater.checkpoint).expanduser()
    config.io.pretrained_embedding_path = Path(
        config.io.pretrained_embedding_path).expanduser()
    config.ltl_encoder.pretrained_embedding_path = Path(
        config.ltl_encoder.pretrained_embedding_path).expanduser()
    config.text_encoder.pretrained_embedding_path = Path(
        config.text_encoder.pretrained_embedding_path).expanduser()
    config.actions_encoder.pretrained_embedding_path = Path(
        config.actions_encoder.pretrained_embedding_path).expanduser()
    root = Path(config.io.root)
    if root == root.expanduser():
        config.io.root = Path(config_file).expanduser().parent / root
    else:
        config.io.root = root.expanduser()
    config.io.output_dir = config.io.root / config.io.output_dir
    config.io.checkpoint_dir = config.io.root / config.io.checkpoint_dir
    config.io.trajectories_dir = config.io.root / config.io.trajectories_dir
    config.io.output_dir.mkdir(exist_ok=True, parents=True)
    config.io.checkpoint_dir.mkdir(exist_ok=True, parents=True)
    config.io.trajectories_dir.mkdir(exist_ok=True, parents=True)
    config.io.data_dir = Path(config.io.data_dir).expanduser()
    config.io.vocab_dir = Path(config.io.vocab_dir).expanduser()
    if config.test.filename is None:
        if (config.io.checkpoint_dir / 'best.pt').exists():
            config.test.filename = config.io.checkpoint_dir / 'best.pt'
        else:
            config.test.filename = config.io.checkpoint_dir / 'best_eval.pt'
    else:
        test_filename = Path(config.test.filename)
        if test_filename == test_filename.expanduser():
            config.test.filename = config.io.root / test_filename
        else:
            config.test.filename = test_filename.expanduser()

    if config.io.tag is None:
        config.io.tag = config.io.root.name
    return config


def serialize_namespace(config: Namespace) -> Dict[str, Any]:
    config = vars(config)
    for k, v in config.items():
        if isinstance(v, Namespace):
            config[k] = serialize_namespace(v)
        if isinstance(v, PosixPath):
            config[k] = str(v)
    return config


def expand_trajectories(
        episode_no: int, batch_size: int,
        obs: List[str], infos: Dict[str, Any],
        states: BatchedStates,
        trajectories: Deque[Dict[str, Any]],
        **kwargs) -> List[Any]:
    for i in range(batch_size):
        idx = -(batch_size - i)
        trajectories[idx]['states'] = states.states[i]
        trajectories[idx]['observations'].append(obs[i])
        trajectories[idx]['infos'] = {k: v[i] for k, v in infos.items()}
        for k, v in kwargs.items():
            trajectories[idx][k].append(v[i])
        # trajectories[idx]['admissible_actions'].append(admissible_actions[i])
        # trajectories[idx]['actions'].append(actions[i])
        # trajectories[idx]['rewards'].append(scores[i])
        # trajectories[idx]['step_rewards'].append(step_scores[i])
        # trajectories[idx]['terminals'].append(dones[i])
    return trajectories


def save_trajectories(trajectories, trajectories_file,) -> None:
    with trajectories_file.open('wb') as f:
        for path in trajectories:
            for k, v in path.items():
                if k not in {'infos', 'admissible_actions', 'states'}:
                    path[k] = np.array(v)
        pickle.dump(trajectories, f)


def to_np(x):
    if isinstance(x, np.ndarray):
        return x
    return x.data.cpu().numpy()


def to_pt(np_matrix, cuda=False, type='long'):
    if type == 'long':
        if cuda:
            return torch.autograd.Variable(torch.from_numpy(np_matrix).type(
                torch.LongTensor).cuda())
        else:
            return torch.autograd.Variable(torch.from_numpy(np_matrix).type(
                torch.LongTensor))
    elif type == 'float':
        if cuda:
            return torch.autograd.Variable(torch.from_numpy(np_matrix).type(
                torch.FloatTensor).cuda())
        else:
            return torch.autograd.Variable(torch.from_numpy(np_matrix).type(
                torch.FloatTensor))


def graph_triplets_to_string(list_of_triples):
    list_of_triples = ["|".join(item) for item in list_of_triples]
    list_of_triples.sort()
    key = "<|>".join(list_of_triples)
    return key


class EpisodicCountingMemory:

    def __init__(self):
        self.reset()

    def push(self, stuff):
        """stuff is list of list of list of strings.
           e.g.: [[['player', 'shed', 'at'], ['workbench', 'shed', 'at']]]
        """
        assert len(stuff) > 0  # batch size should be greater than 0
        if len(self.memory) == 0:
            for _ in range(len(stuff)):
                self.memory.append(set())

        for b in range(len(stuff)):
            key = graph_triplets_to_string(stuff[b])
            self.memory[b].add(key)

    def has_not_seen(self, stuff):
        assert len(stuff) > 0  # batch size should be greater than 0
        res = []
        for b in range(len(stuff)):
            key = graph_triplets_to_string(stuff[b])
            res.append(key not in self.memory[b])
        return res

    def reset(self):
        self.memory = []

    def __len__(self):
        return len(self.memory)


class LinearSchedule:
    """
    Linear interpolation between initial_p and final_p over
    schedule_timesteps. After this many timesteps pass final_p is
    returned.
    :param schedule_timesteps: (int) Number of timesteps for which to linearly
        anneal initial_p to final_p
    :param initial_p: (float) initial output value
    :param final_p: (float) final output value
    """

    def __init__(self, schedule_timesteps: int, final_p: float,
                 initial_p: float = 1.0) -> None:
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p
        self.schedule = np.linspace(initial_p, final_p, schedule_timesteps)

    def value(self, step: int) -> float:
        if step < 0:
            return self.initial_p
        if step >= self.schedule_timesteps:
            return self.final_p
        else:
            return self.schedule[step]

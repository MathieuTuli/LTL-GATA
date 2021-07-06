from typing import List, Tuple, Set, Union

import torch

from textworld.logic import Proposition

from utils import triplet_to_proposition, proposition_to_triplet


def exists_triplet(triplets, arg1, arg2, relation):
    for i, t in enumerate(triplets):
        if arg1 in [t[0], "*"] and\
           arg2 in [t[1], "*"] and\
           relation in [t[2], "*"]:
            return i
    return None


def update_graph_triplets(triplets, commands, node_vocab, relation_vocab):
    # remove duplicate but remain the order
    tmp_commands = []
    for cmd in commands:
        if cmd not in tmp_commands:
            tmp_commands.append(cmd)
    commands = tmp_commands
    for cmd in commands:
        # get verb-arg1-arg2
        if not (cmd.startswith("add") or cmd.startswith("delete")):
            continue
        cmd = cmd.split()
        if len(cmd) <= 3:
            continue
        verb = cmd[0]
        relation = cmd[-1]
        if relation not in relation_vocab:
            continue
        nouns = " ".join(cmd[1:-1])
        arg1, arg2 = "", ""
        for n in node_vocab:
            if nouns.startswith(n):
                tmp = nouns[len(n):].strip()
                if tmp == n:
                    continue
                if tmp in node_vocab:
                    arg1 = n
                    arg2 = tmp
                    break
        if arg1 == "" or arg2 == "":
            continue
        # manipulate KG
        index = exists_triplet(triplets, arg1, arg2, relation)
        if verb == "add":
            if index is not None:
                continue
            triplets.append([arg1, arg2, relation])
        else:
            if index is None:
                continue
            triplets = triplets[:index] + triplets[index + 1:]
    return triplets


class BeliefGraph:
    def __init__(self, observation: str,
                 node_vocab: Set[str],
                 relation_vocab: Set[str],
                 ground_truth: bool,
                 seperator: str) -> None:
        self._facts = set()
        self._node_vocab = node_vocab
        self._facts_as_triplets = list()
        self._observations = observation
        self._ground_truth = ground_truth
        self._relation_vocab = relation_vocab
        self._seperator = seperator
        self.reward = 0
        self.memory = set()

    def to_str(self, facts) -> str:
        return str(['-'.join(fact) for fact in facts].sort())

    @property
    def seen(self) -> bool:
        return self.to_str(self._facts_as_triplets) in self.memory

    def update_memory(self):
        self.memory.add(self.to_str(self._facts_as_triplets))

    def graph_rewards(self, prev_facts: List[Tuple[str, str, str]],
                      entities: List[str],
                      filtered: bool) -> float:
        # if self._ground_truth:
        #     return 0
        # if self.seen:
        #     return self.reward
        if filtered:
            prev_facts = set([tuple(f)
                             for f in prev_facts if f[0] in entities])
            curr_facts = set([tuple(f) for f in self._facts_as_triplets
                              if f[0] in entities])
        else:
            prev_facts = set([tuple(f) for f in prev_facts])
            curr_facts = set([tuple(f) for f in self._facts_as_triplets])
        self.reward += len(curr_facts - prev_facts)
        self.update_memory()
        return self.reward

    @property
    def facts_as_triplets(self) -> Set[Tuple[str, str, str]]:
        if self._ground_truth:
            triplets = list()
            for prop in self._facts:
                triplet = proposition_to_triplet(prop)
                node1, node2, relation = triplet
                if node1 in self._node_vocab and node2 in \
                        self._node_vocab and \
                        relation in self._relation_vocab:
                    triplets.append(triplet)
            return triplets
        return self._facts_as_triplets

    @property
    def facts(self) -> Set[Proposition]:
        return self._facts

    def update(self, facts: Union[List[List[Tuple[Proposition]]],
                                  List[Tuple[str, str, str]]],) -> None:
        if facts is None:
            return
        if self._ground_truth:
            # self._facts = self._facts | set(facts)
            self._facts = facts
            return
        if isinstance(facts, torch.Tensor):
            self._facts = facts
            return
        # per example in a batch
        predict_cmds = facts.split("<sep>")
        if predict_cmds[-1].endswith("<eos>"):
            predict_cmds[-1] = predict_cmds[-1][:-5].strip()
        else:
            predict_cmds = predict_cmds[:-1]
        if len(predict_cmds) == 0:
            return
        predict_cmds = [" ".join(item.split()) for item in predict_cmds]
        predict_cmds = [item for item in predict_cmds if len(item) > 0]
        self._facts_as_triplets = update_graph_triplets(
            self._facts_as_triplets, predict_cmds,
            self._node_vocab, self._relation_vocab)
        new_facts = [triplet_to_proposition(triplet, self._seperator)
                     for triplet in self._facts_as_triplets]
        self._facts = new_facts

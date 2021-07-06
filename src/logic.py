from typing import List
from copy import copy

import pdb

from textworld.logic import Variable as TWVar, Proposition as TWProp


CONSTANT_NAMES = {"P": "player", "I": "player",
                  "ingredient": None, "slot": None, "RECIPE": "cookbook"}


def get_variable_name(name: str) -> str:
    return CONSTANT_NAMES[name] if name in CONSTANT_NAMES else name


class Variable(TWVar):
    def __str__(self) -> str:
        return super().__str__()


class Proposition(TWProp):
    def __init__(self, seperator: str, **kwargs) -> None:
        super(Proposition, self).__init__(**kwargs)
        self.seperator = seperator

    def __str__(self) -> str:
        obj_subj = get_variable_name(self.names[0])
        if len(self.names) == 1:
            string = self.seperator.join([
                obj_subj.replace(' ', self.seperator), 'is',
                self.name])
        else:
            obj_other = get_variable_name(self.names[1])
            string = self.seperator.join([
                obj_subj.replace(' ', self.seperator),
                self.name,
                obj_other.replace(' ', self.seperator), ])
        return string


def proposition_from_textworld_logic(proposition: TWProp,
                                     seperator: str = '_') -> Proposition:
    return Proposition(
        seperator=seperator,
        name=proposition.name,
        arguments=[Variable(x) for x in proposition.names],)


def prune_actions(formula, actions: List[str]):
    pruned = list()
    if 'is_sliced' not in formula and 'is_chopped' not in formula and \
            'is_diced' not in formula:
        return actions
    for i, action in enumerate(actions):
        ing = None
        if 'with knife' in action:
            if 'slice' in action:
                method = 'sliced'
                ing = action.split("slice")[1].split("with")[0].strip()
            if 'dice' in action:
                method = 'diced'
                ing = action.split("dice")[1].split("with")[0].strip()
            if 'chop' in action:
                method = 'chopped'
                ing = action.split("chop")[1].split("with")[0].strip()
            if ing is None or ing == '':
                pruned.append(action)
                continue
            if ing in formula and method in formula:
                pruned.append(action)
        else:
            pruned.append(action)
    return pruned

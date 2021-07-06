'''
Win facts for a single game are as:
   [[],
     [(Proposition('in', (Variable('red potato', 'f'), Variable('I', 'I'))),)],
     [],
     [(Proposition('chopped', (Variable('red potato', 'f'),)),)],
     [(Proposition('in', (Variable('meal', 'meal'), Variable('I', 'I'))),)],
     [(Proposition('consumed', (Variable('meal', 'meal'),)),)]]
'''
from typing import List, Tuple
from copy import deepcopy

import logging
import pdb
import re

from logic import Proposition, Variable, proposition_from_textworld_logic

from ltl import progression
from ltl.translator import Translator

logger = logging.getLogger()


class PadLTL:
    def __init__(self):
        self.name = 'PadLTL'

    def tokenize(self):
        return '( )'

    def tree(self) -> None:
        return None


SEPERATOR = '_'


class LTL:
    '''
    Class that converts and wraps a TextWorld game's win_facts
    as LTL on a single game level (not batched)
    '''

    def __init__(
            self,
            use_ground_truth: bool,
            facts: List[List[Tuple[Proposition, ...]]],
            win_facts: List[List[Tuple[Proposition, ...]]],
            fail_facts: List[List[Tuple[Proposition, ...]]],
            difficulty: int,
            first_obs: str,
            reward_per_progression: bool = False,
            reward_scale: int = 1,
            as_bonus: bool = True,
            single_token_prop: bool = True,
            incomplete_cookbook: bool = False,
            no_cookbook: bool = False,
            single_reward: bool = False,
            next_constrained: bool = False,
            negative_for_fail: bool = False,
            dont_progress: bool = False) -> None:
        self.reward_scale = reward_scale
        self.as_bonus = as_bonus
        self.next_constrained = next_constrained
        self.win_facts = win_facts
        self.fail_facts = fail_facts
        self.use_ground_truth = use_ground_truth
        self.dont_progress = dont_progress
        self.reward_per_progression = reward_per_progression
        self.single_reward = single_reward
        self.negative_for_fail = negative_for_fail
        global SEPERATOR
        self.single_token_prop = single_token_prop
        self.no_cookbook = no_cookbook
        self.progress = self.progress_diffall
        self._reward = 0
        self._done = False
        self._violated = False
        self.translator = Translator(level=difficulty)
        if self.use_ground_truth:
            self.translator.formulas = list()
            cookbook_prop = 'next' if not incomplete_cookbook or \
                no_cookbook else 'eventually'
            examined_cookbook = Proposition(
                name='examined',
                arguments=[Variable('cookbook', type='o')],
                seperator=SEPERATOR)
            if difficulty in {5, 9}:
                self.translator.formulas.append(
                    ('eventually', 'player_at_kitchen'),
                )
            else:
                self.translator.formulas.append((
                    cookbook_prop, str(examined_cookbook)))
                self.translator.formulas.append(
                    progression.standardize((
                        self.facts_to_ltl(self.win_facts)))
                )
        else:
            self.translator.generate_ltl(first_obs)
        self.translator_copy = deepcopy(self.translator)
        self.prev_facts = set(
            [str(proposition_from_textworld_logic(
                f, seperator=SEPERATOR)) for f in facts])

    @property
    def entities(self):
        string = self.tokenize_recursively(self.translator_copy[
            -len(self.translator.formulas)])
        string = re.sub(' +', ' ', string).strip()
        if not self.single_token_prop:
            string = string.replace('_', ' ')
        preds = [pred for pred in string.split(' ') if '_' in pred]
        return {' '.join(tok.split('_')[:-2]) for tok in preds}

    def tree(self) -> None:
        return None

    @property
    def formula(self) -> str:
        return self._formula

    def progress_empty(self, *args, **kwargs):
        return 0, False

    def progress_diffall(self, facts: List[Proposition], action: str,
                         done: bool, observation: str):
        # will only generate once, handles downstream
        if not self.use_ground_truth:
            self.translator.generate_ltl(observation)
            self.translator_copy = deepcopy(self.translator)
        if self._done:
            return self._reward, self._done
        if not self.translator.formulas:
            return self._reward, self._done
        if facts:
            facts = set([str(proposition_from_textworld_logic(
                f, seperator=SEPERATOR)) for f in facts])
        events = facts
        self.prev_facts = facts
        formula = self.translator.formulas[0]
        old_len = len(str(formula))
        formula = progression.progress_and_clean(formula, events)
        progressed = old_len > len(str(formula))
        # if formula is True or formula is False:
        #     break
        if formula == 'None':
            pdb.set_trace()
        if 'always' in str(formula) and 'eventually' not in str(formula):
            formula = 'True'
        self.translator.formulas[0] = formula
        if formula == 'True':
            del self.translator.formulas[0]
            self._done = len(self.translator.formulas) == 0
            if done and not self._done and self.negative_for_fail:
                self._reward = -1 * self.reward_scale
                self._done = True
                self._violated = True
                self.translator.violated = True
            if not self.single_reward:
                self._reward += 1 * self.reward_scale
            elif self._done:
                self._reward = 1
            return self._reward, self._done
        elif formula == 'False':
            self._violated = True
            self.translator.violated = True
            if not self.single_reward:
                # if self.as_bonus:
                #     self._reward -= 1 * self.reward_scale
                # else:
                self._reward = -1 * self.reward_scale
            else:
                self._reward = -1 * self.reward_scale
            self._done = True
            self.translator.formulas = list()
            return self._reward, self._done
        elif progressed and self.reward_per_progression:
            assert not self.single_reward
            self._reward += 1 * self.reward_scale
            if done and not self._done:
                self._reward = -1 * self.reward_scale
                self._done = True
                self._violated = True
                self.translator.violated = True
            return self._reward, self._done
        if done and not self._done:
            self._reward = -1 * self.reward_scale
            self._done = True
            self._violated = True
            self.translator.violated = True

        return self._reward, False

    def facts_to_ltl(self, facts):
        """
        Used if toggling ground truth - doesn't capture
        """
        ltl = list()
        prop = 'eventually'
        for q_count, quest in enumerate(facts):
            if len(quest) == 0:
                continue
            if len(quest) > 1:
                quest_ltl = ['or']
            for prop_tuples in quest:
                if len(prop_tuples) == 0:
                    continue
                if len(prop_tuples) > 1:
                    tuple_ltl = ['and']
                for proposition in prop_tuples:
                    proposition = str(
                        proposition_from_textworld_logic(proposition,
                                                         seperator=SEPERATOR))
                    if len(prop_tuples) > 1:
                        tuple_ltl.append((prop, proposition))
                    else:
                        tuple_ltl = (prop, proposition)
                if not isinstance(tuple_ltl, str):
                    tuple_ltl = tuple(tuple_ltl)
                if len(quest) > 1:
                    quest_ltl.append(tuple_ltl)
                else:
                    quest_ltl = tuple_ltl
            if not isinstance(quest_ltl, str):
                quest_ltl = tuple(quest_ltl)
            ltl.append(quest_ltl)
        if self.next_constrained:
            curr_form = None
            prev_form = None
            for i, prop in enumerate(reversed(ltl)):
                rel, pred = prop
                if prev_form is None:
                    curr_form = prop
                else:
                    curr_form = (
                        'eventually', ('and', pred, ('next', prev_form)))
                prev_form = curr_form
            ltl = curr_form
        else:
            curr_form = None
            prev_form = None
            for i, prop in enumerate(reversed(ltl)):
                rel = 'and' if i == len(ltl) - 1 else 'and'
                if prev_form is None:
                    curr_form = prop
                else:
                    curr_form = (rel, prop, prev_form)
                prev_form = curr_form
            ltl = curr_form
        return tuple(ltl)

    def tokenize(self):
        if self._violated:
            return 'violated'
        elif not self.translator.formulas:
            return 'success'
        if self.no_cookbook and 'cookbook' in str(self.translator.formulas[0]):
            return 'null'
        if self.dont_progress:
            string = self.tokenize_recursively(self.translator_copy.formulas[
                -len(self.translator_copy.formulas)])
        else:
            string = self.tokenize_recursively(self.translator.formulas[0])
        string = re.sub(' +', ' ', string).strip()
        if not self.single_token_prop:
            string = string.replace('_', ' ')
        return string

    def tokenize_recursively(self, ltl):
        string = ''
        for item in ltl:
            if isinstance(item, tuple):
                string += f' {self.tokenize_recursively(item)}'
            else:
                string += f' {item}'
        return string

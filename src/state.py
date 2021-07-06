from __future__ import annotations
from typing import List, Tuple
from copy import deepcopy, copy

from belief_graph import BeliefGraph
from ltl import LTL


class State:
    def __init__(self,
                 observation: str = None,
                 action: str = None,
                 ltl: LTL = None,
                 belief_graph: BeliefGraph = None,
                 past: List[State] = None) -> None:
        self._ltl = ltl
        self._past = list() if past is None else past
        self._action = action
        self._observation = observation
        self._belief_graph = belief_graph

    def update(self, observation: str,
               action: str,
               done: bool,
               facts: List[Tuple[str, str, str]]) -> None:
        self._past.append(
            State(observation=self._observation,
                  ltl=deepcopy(self._ltl),
                  action=self._action,
                  belief_graph=copy(self._belief_graph),
                  past=self.past))

        self._action = action
        self._observation = observation
        self._belief_graph.update(facts)
        if self._ltl is not None:
            ltl_reward, ltl_done = self._ltl.progress(
                self._belief_graph.facts, action, done, observation)
        else:
            ltl_reward, ltl_done = 0, False
        return ltl_reward, ltl_done

    @property
    def past(self) -> List[State]:
        return self._past

    @property
    def action(self) -> str:
        return self._action

    @property
    def observation(self) -> str:
        return self._observation

    @property
    def ltl(self) -> LTL:
        return self._ltl

    @property
    def belief_graph(self) -> BeliefGraph:
        return self._belief_graph


class BatchedStates:
    def __init__(self,
                 states: List[states] = None,
                 observations: List[str] = None,
                 actions: List[str] = None,
                 ltl_formulas: List[LTL] = None,
                 belief_graphs: List[BeliefGraph] = None,
                 action_space_size: int = None) -> None:
        if states is not None:
            self._states = states
        elif None in [observations, actions, ltl_formulas, belief_graphs]:
            raise ValueError(
                "Either states must be passed or all of " +
                "{observations, actions, ltl_formulas, belief_graphs}")
        else:
            self._states = [State(observation=obs, action=act,
                                  ltl=ltl, belief_graph=bg) for
                            obs, act, ltl, bg in
                            zip(observations, actions,
                                ltl_formulas, belief_graphs)]

    def __len__(self) -> int:
        return len(self._states)

    def update(self, observations: List[str],
               actions: List[str],
               facts: List[Tuple[str, str, str]],
               dones: List[bool]):
        state_rewards, state_dones = list(), list()
        for state, obs, act, _facts, done in zip(self._states, observations,
                                                 actions, facts, dones):
            state_reward, state_done = state.update(
                observation=obs, action=act, done=done, facts=_facts)
            state_rewards.append(state_reward)
            state_dones.append(state_done)
        return state_rewards, state_dones

    def __getitem__(self, key: int) -> State:
        return self.states[key]

    @property
    def states(self) -> BatchedStates:
        return self._states

    @property
    def observations(self) -> List[str]:
        return [state.observation for state in self._states]

    @property
    def ltl_formulas(self) -> List[LTL]:
        return [state.ltl for state in self._states]

    @property
    def belief_graphs(self) -> List[BeliefGraph]:
        return [state.belief_graph for state in self._states]

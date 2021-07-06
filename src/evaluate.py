from collections import defaultdict
from argparse import Namespace

import datetime
import logging
import pdb

import numpy as np

from env.cooking import RecipeWrappedEnv
from utils import expand_trajectories
from components import AgentModes
from agent import Agent


logger = logging.getLogger()


def run(env: RecipeWrappedEnv, agent: Agent,
        num_games: int, config: Namespace,
        test: bool = False) -> None:
    trajectories = list()
    start_time = datetime.datetime.now()
    achieved_game_points, achieved_reward, total_game_steps = \
        list(), list(), list()
    original_max_scores, games_max_scores,\
        still_running_mask = list(), list(), list()
    total_success = list()
    for episode_no in range(0, num_games, config.batch_size):
        obs, infos = env.reset()

        actions = ["restart"] * config.batch_size
        prev_step_dones = [0.] * config.batch_size
        trajectories.extend([defaultdict(list)
                            for _ in range(config.batch_size)])

        if test:
            agent.test()
        else:
            agent.eval()
        agent.reset_eval_states(obs, actions, infos)

        previous_hidden, previous_cell = None, None
        original_scores, max_scores = agent.games_max_scores(infos)
        original_max_scores.extend(original_scores)
        games_max_scores.extend(max_scores)
        trajectories = expand_trajectories(
            obs=obs, infos=infos, episode_no=episode_no,
            batch_size=config.batch_size, dones=[False]*config.batch_size,
            admissible_actions=[None] * config.batch_size,
            rewards=[0] * config.batch_size,
            step_rewards=[0] * config.batch_size,
            actions=actions, trajectories=trajectories,
            game_points=[0] * config.batch_size,
            max_game_points=original_max_scores,
            max_rewards=max_scores,
            step_game_points=[0.] * config.batch_size,
            states=agent.eval_states)

        for step_no in range(config.steps_per_episode):
            admissible_actions = infos['admissible_commands']
            logger.debug("Eval acting")
            if config.feed_cookbook_observation and step_no == 0:
                next_actions_ind = [actions.index('examine cookbook') for
                                    actions in admissible_actions]
                actions = ["examine cookbook" for
                           _ in admissible_actions]
            else:
                next_actions_ind, previous_hidden, previous_cell = \
                    agent.act(states=agent.eval_states,
                              admissible_actions=admissible_actions,
                              greedy=True if np.equal(
                                  config.eps, 0) else False,
                              previous_hidden=previous_hidden,
                              previous_cell=previous_cell,
                              eps=config.eps)
                next_actions_ind = next_actions_ind.cpu().tolist()
                actions = [candidates[i] for candidates, i in zip(
                    admissible_actions, next_actions_ind)]
            logger.debug("Eval env step")
            obs, scores, dones, infos = env.step(
                actions, agent.eval_states.ltl_formulas)
            # dones_wons = [done and won for done, won in
            #               zip(dones, infos['has_won'])]
            logger.debug("Eval state update")
            state_update_rewards, state_update_dones = \
                agent.update_eval_states(obs, actions, dones, infos)
            original_scores, scores = agent.adjust_scores(
                scores, step_no, state_update_rewards, state_update_dones,
                infos['won'], dones)

            trajectories = expand_trajectories(
                obs=obs, infos=infos, episode_no=episode_no,
                states=agent.eval_states,
                batch_size=config.batch_size, dones=dones, rewards=scores,
                admissible_actions=admissible_actions,
                actions=actions, trajectories=trajectories,
                step_rewards=[0.] * config.batch_size,
                game_points=original_scores,
                max_game_points=original_max_scores,
                max_rewards=max_scores,
                step_game_points=[0.] * config.batch_size,
            )

            still_running = [1. - float(item) for item in prev_step_dones]

            prev_step_dones = dones
            still_running_mask.append(still_running)
            if np.sum(still_running) == 0:
                break
        achieved_game_points.extend(original_scores)
        achieved_reward.extend(scores)
        total_success.extend([won for won in infos['won']])
    total_game_steps = np.sum(still_running_mask, 0).tolist()
    time_mark = datetime.datetime.now()
    achieved_game_points = np.array(achieved_game_points, dtype="float32")
    achieved_reward = np.array(achieved_reward, dtype="float32")
    normalized_game_points = achieved_game_points / original_max_scores
    normalized_rewards = achieved_reward / games_max_scores
    original_max_scores = np.array(original_max_scores, dtype="float32")
    games_max_scores = np.array(games_max_scores, dtype="float32")
    logger.info(
        f"\nEval | T: {str(time_mark - start_time).rsplit('.')[0]:s} | " +
        f"normalized game points: {np.mean(normalized_game_points):2.3f} | " +
        f"normalized reward: {np.mean(normalized_rewards):2.3f} | " +
        f"game success: {np.mean(total_success):.3f} | " +
        f"steps: {np.mean(total_game_steps):2.2f}"
    )
    return (np.mean(achieved_game_points),
            np.mean(normalized_game_points),
            np.mean(achieved_reward),
            np.mean(normalized_rewards),
            np.mean(total_game_steps),
            np.mean(total_success), trajectories)

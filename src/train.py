from argparse import ArgumentParser, Namespace
from pathlib import Path

from collections import defaultdict, deque
from copy import deepcopy

import datetime
import logging
import copy
import csv
import pdb


import numpy as np
import tqdm
import yaml

from logic import proposition_from_textworld_logic
from components import Sample, ResultsCSVField
from utils import (expand_trajectories,
                   serialize_namespace,
                   save_trajectories,
                   get_variable_name)
from evaluate import run as evaluate
from ltl import PadLTL, progression
from env import get_game_env
from args import add_args
from agent import Agent
from state import State

parser = ArgumentParser()
add_args(parser)
logger = logging.getLogger()


def train(config: Namespace, pretrain: bool = False):
    start_time = datetime.datetime.now()

    fprefix = '' if not pretrain else 'pretrain_'
    results_file = config.io.output_dir / f'{fprefix}results.csv'
    config_file = config.io.output_dir / f'{fprefix}config.yaml'
    with config_file.open('w') as f:
        yaml.dump(serialize_namespace(deepcopy(config)), f)
    if not config.checkpoint.resume:
        with results_file.open('w') as f:
            writer = csv.writer(f)
            writer.writerow(ResultsCSVField().keys())

    eval_env, num_eval_game = None, None
    if pretrain:
        raise NotImplementedError
        env, _ = get_game_env(
            game='simple_ltl',
            progression_mode='full',
            sampler=config.pretain.ltl_sampler,
            batch_size=config.pretrain.batch_size,
            real_valued_graph=config.graph_updater.real_valued,
            randomized_nouns_verbs=config.training.randomized_nouns_verbs,)
        agent = Agent(config=config,
                      word_vocab=env.word_vocab,
                      ltl_vocab=env.ltl_vocab,
                      action_vocab=env.action_vocab,
                      pretrain=True,)
    else:
        # make game environments
        gr = config.training.graph_reward_lambda > 0 \
            and config.training.graph_reward_filtered
        requested_infos = Agent.env_infos(
            light=config.training.light_env_infos,
            win_formulas=config.model.use_ltl or gr)

        # training game env
        env, _ = get_game_env(
            game=config.training.game,
            data_dir=config.io.data_dir,
            vocab_dir=config.io.vocab_dir,
            difficulty_level=config.training.difficulty_level,
            requested_infos=requested_infos,
            max_episode_steps=config.training.steps_per_episode,
            batch_size=config.training.batch_size,
            split='train',
            all_games=config.training.all_games,
            training_size=config.training.training_size,
            game_limit=config.training.game_limit,
            real_valued_graph=config.graph_updater.real_valued,
            randomized_nouns_verbs=config.training.randomized_nouns_verbs,
            prune=config.training.prune_actions,
            strip_instructions=config.training.strip_instructions)

        if config.evaluate.run:
            # training game env
            eval_env, num_eval_game = get_game_env(
                game=config.evaluate.game,
                data_dir=config.io.data_dir,
                vocab_dir=config.io.vocab_dir,
                difficulty_level=config.evaluate.difficulty_level,
                requested_infos=requested_infos,
                max_episode_steps=config.evaluate.steps_per_episode,
                batch_size=config.evaluate.batch_size,
                split='valid',
                real_valued_graph=config.graph_updater.real_valued,
                randomized_nouns_verbs=False,
                prune=config.training.prune_actions,
                strip_instructions=config.training.strip_instructions)

        agent = Agent(config=config,
                      word_vocab=env.word_vocab,
                      ltl_vocab=env.ltl_vocab,
                      node_vocab=env.node_vocab,
                      relation_vocab=env.relation_vocab,
                      action_vocab=env.action_vocab,
                      pretrain=False,)
    if config.checkpoint.resume:
        fname = config.io.checkpoint_dir / (
            fprefix + 'latest.pt')
    else:
        fname = config.io.checkpoint_dir / (
            fprefix + config.checkpoint.filename)
    if (config.checkpoint.load or config.checkpoint.resume) and fname.exists():
        logging.info(f"Loading from checkpoint : {fname}")
        start_episode_no, best_train, best_eval = agent.load_model(fname)
        logger.info(f"Loaded model from {fname}")
    else:
        start_episode_no, best_train, best_eval = 0, 0., 0.
    if not config.checkpoint.resume:
        start_episode_no, best_train, best_eval = 0, 0., 0.

    trajectories = list()
    cache_dqn_loss = deque(maxlen=config.io.report_history_length)
    cache_game_steps = deque(maxlen=config.io.report_history_length)
    cache_game_points = deque(maxlen=config.io.report_history_length)
    cache_game_rewards = deque(maxlen=config.io.report_history_length)
    cache_game_points_normalized = deque(
        maxlen=config.io.report_history_length)
    cache_game_rewards_normalized = deque(
        maxlen=config.io.report_history_length)
    cache_success = deque(
        maxlen=config.io.report_history_length)
    patience, total_steps, perfect_training = 0, 0, 0
    prev_performance = 0.
    eval_game_step, eval_success = 0., 0.
    eval_game_points, eval_game_points_normalized, \
        eval_normalized_rewards, eval_rewards = 0., 0., 0., 0.

    batch_size = config.training.batch_size
    episodes = tqdm.tqdm(range(start_episode_no,
                               config.training.max_episode,
                               batch_size))
    pad_state = State(observation=agent._word_vocab.pad_token,
                      action=agent._word_vocab.pad_token,
                      ltl=PadLTL())
    # import cProfile
    # import pstats
    # import io
    # from pstats import SortKey
    # pr = cProfile.Profile()
    first_log = True
    for episode_no in episodes:
        # pr.enable()
        # if episode_no > 0:
        #     s = io.StringIO()
        #     sortby = SortKey.CUMULATIVE
        #     ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        #     ps.print_stats()
        #     print(s.getvalue())
        #     raise
        env.seed(episode_no)
        # np.random.seed(episode_no)

        obs, infos = env.reset()

        actions = ["restart"] * batch_size
        agent.train()
        agent.reset_states(obs, actions, infos)

        previous_hidden, previous_cell = None, None
        prev_rewards = [0. for _ in range(batch_size)]
        prev_step_dones = [0. for _ in range(batch_size)]
        prev_original_scores = [0. for _ in range(batch_size)]
        actions_cache = list()
        act_randomly = episode_no < config.training.learn_from_this_episode
        still_running_mask, game_rewards, game_points = list(), list(), list()
        dones_list = list()
        original_max_scores, games_max_scores = agent.games_max_scores(infos)
        if first_log:
            logger.info(f"Max Scores : {games_max_scores[0]}")
            first_log = False
        if config.io.save_trajectories_frequency > 0:
            trajectories.extend([defaultdict(list) for _ in range(batch_size)])
            trajectories = expand_trajectories(
                obs=obs, infos=infos, episode_no=episode_no - start_episode_no,
                trajectories=trajectories, states=agent.states,
                batch_size=batch_size,
                dones=[False]*batch_size,
                admissible_actions=[None] * batch_size,
                rewards=[0] * batch_size,
                step_rewards=[0] * batch_size,
                game_points=[0] * batch_size,
                max_game_points=original_max_scores,
                max_rewards=games_max_scores,
                step_game_points=[0] * batch_size,
                actions=actions,)
        for step_no in range(config.training.steps_per_episode):
            admissible_actions = infos['admissible_commands']
            logger.debug("Acting")
            if config.training.feed_cookbook_observation and step_no == 0:
                next_actions_ind = [actions.index('examine cookbook') for
                                    actions in admissible_actions]
                actions = ["examine cookbook" for
                           _ in admissible_actions]
            else:
                next_actions_ind, previous_hidden, previous_cell = agent.act(
                    states=agent.states,
                    admissible_actions=admissible_actions,
                    random=act_randomly,
                    previous_hidden=previous_hidden,
                    previous_cell=previous_cell)
                next_actions_ind = next_actions_ind.cpu().tolist()
                actions = [candidates[i] for candidates, i in zip(
                    admissible_actions, next_actions_ind)]
            actions_cache.append((next_actions_ind, admissible_actions))
            if episode_no == 100000:
                pass
                # pdb.set_trace()
            logger.debug("Doing env step")
            obs, scores, dones, infos = env.step(
                next_actions_ind if config.training.randomized_nouns_verbs
                else actions,
                agent.states.ltl_formulas)
            # dones_wons = [done and won for done, won in
            #               zip(dones, infos['won'])]
            logger.debug("Updating states")
            state_update_rewards, state_update_dones = agent.update_states(
                obs, actions, dones, infos)
            original_scores, scores = agent.adjust_scores(
                scores, step_no, state_update_rewards, state_update_dones,
                infos['won'], dones)
            if config.model.use_ltl and config.training.end_on_ltl_violation:
                dones = list(dones)
                for i, ltl_done in enumerate(state_update_dones):
                    valid = True if not still_running_mask else \
                        still_running_mask[-1][i]
                    if agent.states[i].ltl._violated and \
                            config.training.backwards_ltl and \
                            valid:
                        prev_facts = \
                            agent.states[i].past[-1].belief_graph._facts
                        curr_facts = \
                            agent.states[i].belief_graph._facts
                        entities = agent.states[i].ltl.entities
                        prev_facts = set([f
                                         for f in prev_facts if
                                         get_variable_name(f.names[0]) in
                                         entities])
                        curr_facts = set([f for f in curr_facts
                                          if get_variable_name(f.names[0])
                                          in entities])
                        diff = [x for x in curr_facts -
                                prev_facts if x in curr_facts]
                        nots = [
                            ('always',
                             ('not',
                              str(proposition_from_textworld_logic(
                                  proposition))))
                            for proposition in diff]
                        if len(nots) > 0:
                            not_form = nots[0]
                            for pred in nots[1:]:
                                not_form = progression.standardize(
                                    ('and', not_form, pred))
                            for j, state in enumerate(agent.states[i].past):
                                if not state.ltl._formulas:
                                    continue
                                try:
                                    form = state.ltl._formulas[0]
                                    if form[0] == 'next':
                                        new_form = \
                                            ('and', form, not_form)
                                    else:
                                        new_form = \
                                            progression.standardize(
                                                ('and', form, not_form))
                                    agent.states[i].past[j].ltl._formulas[0] =  \
                                        new_form
                                except:
                                    pdb.set_trace()

                    if dones[i] and not ltl_done and \
                            step_no != config.training.steps_per_episode - 1:
                        # pdb.set_trace()
                        # raise ValueError("The game is done but not the LTL")
                        pass
                    dones[i] = dones[i] or ltl_done
                dones = tuple(dones)
            for i in range(config.training.batch_size):
                if scores[i] > games_max_scores[i]:
                    # pdb.set_trace()
                    # raise ValueError("Can't have a reward > max")
                    pass

            if episode_no >= config.training.learn_from_this_episode and \
                    total_steps % config.training.update_per_k_game_steps == 0:
                logger.debug("Updating DQN")
                loss, q_values = agent.update_dqn(episode_no)
                if loss is not None:
                    cache_dqn_loss.append(loss.detach().cpu())

            if step_no == config.training.steps_per_episode - 1:
                # terminate the game because DQN requires one extra step
                dones = [True] * len(dones)
            dones_list.append(dones)

            total_steps += 1
            prev_step_dones = dones
            still_running = [1. - float(item)
                             for item in prev_step_dones]
            step_rewards = [float(curr) - float(prev) for curr,
                            prev in zip(scores, prev_rewards)]  # list of float
            # if config.training.reward_ltl and config.model.use_ltl and \
            #         not config.training.reward_ltl_positive_only:
            #     for i, (done, won) in enumerate(zip(dones, infos['won'])):
            #         if done and not won:
            #             step_rewards[i] = -1
            # if config.training.persistent_negative_reward:
            #     for i, r in enumerate(scores):
            #         if r < 0 or prev_rewards[i] < 0:
            #             step_rewards[i] = r
            step_game_points = [float(curr) - float(prev) for curr,
                                prev in zip(original_scores,
                                            prev_original_scores)]
            game_points.append(copy.copy(step_game_points))
            game_rewards.append(copy.copy(step_rewards))
            prev_rewards = scores
            prev_original_scores = original_scores
            still_running_mask.append(still_running)

            if config.io.save_trajectories_frequency > 0:
                trajectories = expand_trajectories(
                    obs=obs, infos=infos,
                    episode_no=episode_no - start_episode_no,
                    batch_size=batch_size, dones=dones,
                    states=agent.states, trajectories=trajectories,
                    admissible_actions=admissible_actions,
                    rewards=scores,
                    step_rewards=step_rewards,
                    game_points=original_scores,
                    max_game_points=original_max_scores,
                    max_rewards=games_max_scores,
                    step_game_points=step_game_points,
                    actions=actions,
                )
            # if all ended, break
            if np.sum(still_running) == 0:
                logger.debug('All games ended, breaking')
                break

        logger.debug("Done Episode")
        mod_still_running_mask = np.array(
            [[1] * config.training.batch_size] + still_running_mask[:-1])
        still_running_mask = np.array(still_running_mask)
        # if config.training.persistent_negative_reward:
        #     game_points = np.array(game_points)
        #     game_rewards = np.array(game_rewards)
        # else:
        game_points = np.array(game_points) * \
            mod_still_running_mask  # step x batch
        game_rewards = np.array(game_rewards) * \
            mod_still_running_mask  # step x batch

        avg_rewards_in_buffer = agent.experience.avg_rewards()
        for b in range(batch_size):
            # if still_running_mask[0][b] == 0:
            #     continue
            # if (still_running_mask.shape[0] ==
            #         config.training.steps_per_episode and
            #         still_running_mask[-1][b] != 0):
            # if (still_running_mask.shape[0] ==
            #         config.training.steps_per_episode and
            #         still_running_mask[-1][b] != 0):
            #     # need to pad one transition
            #     _need_pad = True
            #     tmp_game_rewards = game_rewards[:, b].tolist() + [0.]
            # else:
            #     _need_pad = False
            #     tmp_game_rewards = game_rewards[:, b]
            # if np.mean(tmp_game_rewards) < avg_rewards_in_buffer * \
            #         config.training.experience_replay[
            #             'buffer_reward_threshold'] and \
            #         agent.experience.at_capacity:
            #     continue
            # TODO TOGGLE THIS
            # past_index = -min(config.training.steps_per_episode,
            #                   len(agent.states[0].past))
            past_index = 0
            _need_pad = False
            for i in range(game_rewards.shape[0]):
                is_final = True
                if mod_still_running_mask[i][b] != 0:
                    is_final = False

                # assert actions_cache[i][1][b][actions_cache[i][0][b]] == \
                #     agent.states[b].past[past_index + i].action
                agent.experience.add(Sample(
                    step=i,
                    action=actions_cache[i][0][b],
                    state=agent.states[b].past[past_index + i],
                    reward=game_rewards[i][b],
                    admissible_actions=actions_cache[i][1][b],
                    done=dones_list[i][b],
                    is_final=is_final))
                if mod_still_running_mask[i][b] == 0:
                    break
            # _need_pad = False
            if _need_pad:
                agent.experience.add(Sample(
                    step=i+1,
                    action=agent._word_vocab.pad_token,  # 0
                    state=pad_state,  # pad_state
                    reward=0.,
                    done=True,
                    # [agent._word_vocab.pad_token],
                    admissible_actions=actions_cache[i][1][b],
                    is_final=True))

        for b in range(batch_size):
            cache_game_points.append(np.sum(game_points, 0)[b])
            cache_game_points_normalized.append(
                (np.sum(game_points, 0) / original_max_scores)[b])
            cache_game_rewards.append(np.sum(game_rewards, 0)[b])
            cache_game_rewards_normalized.append(
                (np.sum(game_rewards, 0) / games_max_scores)[b])
            cache_game_steps.append(np.sum(still_running_mask, 0)[b])
            cache_success.append(infos['won'][b])

        # finish game
        agent.finalize_episode(episode_no)

        if episode_no < config.training.learn_from_this_episode:
            continue
        time_mark = datetime.datetime.now()
        points_norm = np.mean(cache_game_points_normalized)
        rewards_norm = np.mean(cache_game_rewards_normalized)
        success = np.mean(cache_success)
        if config.io.report_frequency != 0 and episode_no > 0 and \
            (episode_no) % config.io.report_frequency <= \
                (episode_no - batch_size) % config.io.report_frequency:
            logger.info(
                f"\nTrain: {episode_no:3d} | " +
                f"Time: {str(time_mark - start_time).rsplit('.')[0]:s} | " +
                f"dqn loss: {np.mean(cache_dqn_loss):2.5f} | " +
                f"normalized game points: {points_norm:2.3f} | " +
                f"normalized rewards: {rewards_norm:2.3f} | " +
                f"game success: {success:.3f} | " +
                f"used steps: {np.mean(cache_game_steps):2.3f}"
            )

        curr_train_performance = np.mean(cache_game_rewards_normalized)
        curr_train_performance = success
        if episode_no > 0 and \
                episode_no % config.checkpoint.save_frequency == 0:
            logger.info("Saved latest model")
            agent.save_model(
                episode_no, config.io.checkpoint_dir / 'latest.pt',
                best_train, best_eval)
            if config.checkpoint.save_each:
                agent.save_model(
                    episode_no, config.io.checkpoint_dir /
                    f'episode_{episode_no}.pt',
                    best_train, best_eval)
        if config.evaluate.run and episode_no > 0 and \
                episode_no % config.evaluate.frequency == 0 \
                and eval_env is not None:
            logger.debug("Running Eval")
            eval_game_points, eval_game_points_normalized,\
                eval_rewards, \
                eval_normalized_rewards, eval_game_step, \
                eval_success, eval_trajectories = evaluate(
                    eval_env, agent, num_eval_game, config.evaluate)
            trajectories_file = config.io.trajectories_dir /\
                f'eval_trajectories_e={episode_no}.pkl'
            save_trajectories(eval_trajectories, trajectories_file)
            # TODO note this here...
            # curr_eval_performance = eval_normalized_rewards
            curr_eval_performance = eval_game_points_normalized
            curr_eval_performance = eval_success
            curr_performance = curr_eval_performance
            if curr_eval_performance > best_eval:
                best_eval = curr_eval_performance
                logger.info("Saved best model")
                agent.save_model(
                    episode_no, config.io.checkpoint_dir / 'best.pt',
                    best_train, best_eval)
                agent.save_model(
                    episode_no, config.io.checkpoint_dir / 'best_eval.pt',
                    best_train, best_eval)
            elif curr_eval_performance == best_eval:
                if curr_eval_performance > 0.:
                    logger.info("Saved best model")
                    agent.save_model(episode_no, config.io.checkpoint_dir /
                                     'best.pt',
                                     best_train, best_eval)
                    agent.save_model(episode_no, config.io.checkpoint_dir /
                                     'best_eval.pt',
                                     best_train, best_eval)
                else:
                    if curr_train_performance >= best_train:
                        logger.info("Saved best model")
                        agent.save_model(episode_no, config.io.checkpoint_dir /
                                         'best.pt',
                                         best_train, best_eval)
                        agent.save_model(episode_no, config.io.checkpoint_dir /
                                         'best_train.pt',
                                         best_train, best_eval)
        else:
            curr_eval_performance = 0.
            curr_performance = curr_train_performance
            if curr_train_performance >= best_train:
                agent.save_model(
                    episode_no, config.io.checkpoint_dir /
                    (fprefix + 'best.pt'),
                    best_train, best_eval)
                agent.save_model(
                    episode_no, config.io.checkpoint_dir /
                    (fprefix + 'best_train.pt'),
                    best_train, best_eval)
        # update best train performance
        if curr_train_performance >= best_train:
            best_train = curr_train_performance

        if prev_performance <= curr_performance:
            patience = 0
        else:
            patience += 1
        prev_performance = curr_performance

        # if patient >= patience, resume from checkpoint
        if config.training.patience > 0 and \
                patience >= config.training.patience:
            if (config.io.checkpoint_dir / 'best.pt').exists():
                patience = 0
                logger.info('Patience exceeded. ' +
                            'Reloading from a good checkpoint.')
                agent.load_model(str(config.io.checkpoint_dir /
                                     'best.pt'))

        if np.mean(points_norm) > 0.96:
            perfect_training += 1
        else:
            perfect_training = 0

        logger.debug("Writing results to file")
        with results_file.open('a') as f:
            writer = csv.writer(f)
            writer.writerow(
                ResultsCSVField(
                    time=str(time_mark - start_time).rsplit(".")[0],
                    episode_no=episode_no,
                    dqn_loss=np.mean(cache_dqn_loss),
                    train_game_points=np.mean(cache_game_points),
                    train_normalized_game_points=points_norm,
                    train_rewards=np.mean(cache_game_rewards),
                    train_normalized_rewards=np.mean(rewards_norm),
                    train_steps=np.mean(cache_game_steps),
                    train_success=success,
                    eval_game_points=eval_game_points,
                    eval_normalized_game_points=eval_game_points_normalized,
                    eval_rewards=eval_rewards,
                    eval_normalized_rewards=eval_normalized_rewards,
                    eval_steps=eval_game_step,
                    eval_success=eval_success,
                ).values())
        logger.debug("Done writing results to file")

        # if curr_performance == 1. and curr_train_performance >= 0.95:
        #     break
        if perfect_training >= 3:
            logging.info("Perfect training, done training")
            break
        if episode_no > 0 and \
                episode_no % config.io.save_trajectories_frequency == 0 and  \
                config.io.save_trajectories_frequency > 0:
            logger.info("Saving train trajectories")
            trajectories_file = config.io.trajectories_dir / \
                f'train_trajectories_e={episode_no}.pkl'
            save_trajectories(
                trajectories, trajectories_file,)
            trajectories = list()
    logger.info(
        "Train: End | " +
        f"T: {str(time_mark - start_time).rsplit('.')[0]:s} | " +
        f"dqn loss: {np.mean(cache_dqn_loss):2.5f} | " +
        f"normalized game points: {points_norm:2.3f} | " +
        f"normalized rewards: {rewards_norm:2.3f} | " +
        f"game success: {success:.3f} | " +
        f"used steps: {np.mean(cache_game_steps):2.3f}"
    )
    evaluate(eval_env, agent, num_eval_game, config.evaluate)
    agent.save_model(episode_no, config.io.checkpoint_dir / 'latest.pt',
                     best_train, best_eval)


if __name__ == '__main__':
    args = parser.parse_args()
    train(args)

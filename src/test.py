from argparse import Namespace

import logging
import csv


from utils import save_trajectories
from evaluate import run as evaluate
from env import get_game_env
from agent import Agent

logger = logging.getLogger()


def test(config: Namespace):
    # np.random.seed(config.training.random_seed)

    # make game environments
    gr = config.training.graph_reward_lambda > 0 \
        and config.training.graph_reward_filtered
    requested_infos = Agent.env_infos(light=False,
                                      win_formulas=config.model.use_ltl or gr)

    env, num_game = get_game_env(
        game=config.test.game,
        data_dir=config.io.data_dir,
        vocab_dir=config.io.vocab_dir,
        difficulty_level=config.test.difficulty_level,
        requested_infos=requested_infos,
        max_episode_steps=config.test.steps_per_episode,
        batch_size=config.test.batch_size,
        split='test',
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
    agent.load_model(config.test.filename)
    logger.info(f"Loaded model from {config.test.filename}")
    test_game_points, test_game_points_normalized, test_reward, \
        test_normalized_rewards, test_game_steps, \
        test_success, test_trajectories = evaluate(env, agent, num_game,
                                                   config.test, test=True)
    logger.info(f"Saving results to {config.io.output_dir}")
    # results_data = {'config': serialize_namespace(deepcopy(config))}
    data = {
        "test_game_points": str(test_game_points),
        "test_normalized_game_points": str(test_game_points_normalized),
        "test_rewards": str(test_reward),
        "test_normalized_rewards": str(test_normalized_rewards),
        "test_game_steps": str(test_game_steps),
        "test_success": str(test_success),
    }
    results_file = config.io.output_dir / 'test_results.csv'
    with results_file.open('w') as f:
        writer = csv.writer(f)
        writer.writerow(data.keys())
        writer.writerow(data.values())
    save_trajectories(test_trajectories,
                      config.io.trajectories_dir / 'test_trajectories.pkl',)

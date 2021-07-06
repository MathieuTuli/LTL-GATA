
from argparse import ArgumentParser, Namespace
from copy import deepcopy

from gutils import init_logger

from args import add_args
from train import train
from test import test

import utils

parser = ArgumentParser()
add_args(parser)
logger = None


def main(args: Namespace):
    config = utils.load_config(args.config, args.params)

    logger = init_logger(log_file=config.io.root / args.logs,
                         log_level=args.log_level,
                         name=config.io.tag)
    if args.pretrain:
        logger.info(f"Running pretraining for {config.io.tag}")
        config_pretrain = deepcopy(config)
        train_args = vars(config_pretrain.train)
        pretrain_args = vars(config_pretrain.pretrain)
        for elem, val in pretrain_args.items():
            if elem in train_args:
                train_args[elem] = val
        train(config=config_pretrain, pretrain=True)
    if args.train:
        logger.info("Running training")
        logger.info(f"Experiment tag: {config.io.tag}")
        logger.info(f"Output Dir: {config.io.output_dir}")
        logger.info(f"Checkpoint Dir: {config.io.checkpoint_dir}")
        logger.info(f"Trajectories Dir: {config.io.trajectories_dir}")
        train(config)
    if args.test:
        logger.info(f"Running testing for {config.io.tag}")
        test(config)
    if not args.test and not args.train:
        logger.warning("Unknown command. Use either '--test' or '--train'")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

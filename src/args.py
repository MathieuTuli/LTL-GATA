from argparse import ArgumentParser

from datetime import datetime

from gutils.components import LogLevel


def add_args(parser: ArgumentParser) -> None:
    parser.add_argument('--config', help='Config file path',
                        default='config.yaml')
    parser.add_argument(
        '--logs', help="Set output for log outputs",
        default=f"logs/tod_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.log")
    parser.add_argument('--log-level', type=LogLevel.__getitem__,
                        default=LogLevel.INFO,
                        choices=LogLevel.__members__.values(),
                        dest='log_level',
                        help="Log level.")
    parser.add_argument(
        "-p", "--params", nargs="+",
        metavar="my.setting=value", default=[],
        help="override params of the config file,"
        " e.g. -p 'training.gamma=0.95'")
    parser.add_argument("--pretrain", action='store_true', default=False,)
    parser.add_argument("--train", action='store_true', default=False,)
    parser.add_argument("--test", action='store_true', default=False,)

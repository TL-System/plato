import asyncio
import logging
import argparse

import config
from institutions import SimpleInstitution

# Setting up the parser
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--id', type=str,
                    help='Unique institution ID.')
parser.add_argument('-c', '--config', type=str, default='./config.conf',
                    help='Federated learning configuration file.')
parser.add_argument('-l', '--log', type=str, default='INFO',
                    help='Log messages level.')

args = parser.parse_args()

logging.basicConfig(
    format='[%(levelname)s][%(asctime)s]: %(message)s',
    level='INFO', datefmt='%H:%M:%S')


def main():
    """Run a federated learning institution."""
    fl_config = config.Config(args.config)

    institution = SimpleInstitution(fl_config, args.id)
    institution.configure()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(institution.start_institution())
    loop.run_forever()


if __name__ == "__main__":
    main()

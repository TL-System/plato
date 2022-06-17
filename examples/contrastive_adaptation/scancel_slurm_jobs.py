"""
Implementation of canceling multiples experiments with Slurm.


The type of the experiments should be provided by:
    - ids

Thus, the jobs containing the ids will be cancelled.

python examples/contrastive_adaptation/scancel_slum_jobs.py -i 2340 2341 2344 2346

"""

import argparse
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i',
                        '--ids',
                        nargs='+',
                        default=[],
                        help='The jobs with ids will be cancelled.')

    args = parser.parse_args()

    jobs_id = [int(input_id) for input_id in args.ids]
    print(f"Cancelling job with id: {jobs_id}")

    for id in jobs_id:
        os.system("scancel %s" % id)

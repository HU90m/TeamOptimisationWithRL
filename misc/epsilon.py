#!/usr/bin/python3
"""Script for finding the best epsilon decay
or the final value due to a given decay.
"""
from math import log, exp
from argparse import ArgumentParser


def find_decay(episodes, epsilon_start, epsilon_end):
    """Find the epsilon decay"""
    return - log(epsilon_end/epsilon_start) / episodes


def find_end(episodes, epsilon_start, epsilon_decay):
    """Find the final epsilon value"""
    return epsilon_start * exp(-epsilon_decay * episodes)


if __name__ == '__main__':
    parser = ArgumentParser(description='Calculate the desired epsilon decay.')
    parser.add_argument('--find-final', dest='find_final',
            action='store_true', default=False,
            help='If set will find the final value with a given decay.')

    parser.add_argument('-e', '--episodes', dest='episodes', type=int,
            required=True, help='number of episodes which will be run.')

    parser.add_argument('-f', '--final', dest='end', type=float, default=0.01,
            help='the final epsilon value.')
    parser.add_argument('-i', '--initial', dest='start', type=float,
            default=1., help='the initial epsilon value.')

    parser.add_argument('-d', '--decay', dest='decay', type=float,
            default=1e-4, help='the epsilon decay.')


    args = parser.parse_args()

    if args.find_final:
        print('final value:', find_end(args.episodes, args.start, args.decay))
    else:
        print('decay:', find_decay(args.episodes, args.start, args.end))

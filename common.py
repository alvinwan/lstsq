#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: common.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>
import random
import time
import threading
import multiprocessing
import numpy as np
from tqdm import tqdm
from six.moves import queue
import os
import os.path

from tensorpack import *
from tensorpack.utils.concurrency import *
from tensorpack.utils.stats import *

LAYER = 'prelu'
DAGGER = os.environ.get('DAGGER', False) == 'True'


def play_one_episode(player, func, verbose=False):
    def f(s):
        """
        Note that the state has frames from the past. I checked that the last
        three channels are the current frame, by collecting several states and
        comparing groups of channels.

        :param s: output of neural network
        :return:
        """
        spc = player.get_action_space()
        out = func([[s]])
        act = out[0][0].argmax()
        state = np.ravel(out[1][0][:, :, -3:])
        if random.random() < 0.001:
            act = spc.sample()
        if verbose:
            print(act)
        return act, state, act
    return np.mean(player.play_one_episode(f))


def play_one_dagger_episode(player, func, verbose=False, N=200, N_d=200, idx=0):
    #path = '/data/alvin/lstsq/data/spaceinvaders-precompute/%s-w-%d'
    #if DAGGER:
    #    path += ('-dagger-%d' % N_d)
    path = 'spaceinvaders-precompute/w-%d' % idx
    w = np.load(path + '.npy')
    def f(s):
        spc = player.get_action_space()
        out = func([[s]])
        state = out[1][0][:, :, -3:]
        act = state.T.dot(w).argmax()
        exp_act = out[0][0].argmax()
        if random.random() < 0.001:
            act = spc.sample()
        if verbose:
            print(act)
        return act, state, exp_act
    return np.mean(player.play_one_episode(f))


def play_model(cfg, player, num_episodes=np.inf):
    predfunc = OfflinePredictor(cfg)
    for i in range(num_episodes):
        score = play_one_episode(player, predfunc)
        print("Total:", score)


def play_dagger_model(cfg, player, num_episodes=np.inf, N=200, N_d=200, idx=0):
    predfunc = OfflinePredictor(cfg)
    scores = []
    save_path = os.path.join(player.SAVE_DIR, 'results-%d.txt' % N)
    best_mean_reward = 0
    for i in range(num_episodes):
        score = play_one_dagger_episode(player, predfunc, N=N, N_d=N_d, idx=idx)
        scores.append(score)
        print("(%d) Total:" % i, score)
        if i % 10 == 0:
            print('Average Reward:', np.mean(scores))
            if i >= 100:
                print('Last Mean Reward:', np.mean(scores[-100:]))
                best_mean_reward = max(best_mean_reward, np.mean(scores[-100:]))
                print('Best Mean Reward:', best_mean_reward)
    with open(save_path, 'w') as f:
        f.write(','.join(map(str, scores)))


def eval_with_funcs(predictors, nr_eval, get_player_fn):
    class Worker(StoppableThread, ShareSessionThread):
        def __init__(self, func, queue):
            super(Worker, self).__init__()
            self._func = func
            self.q = queue

        def func(self, *args, **kwargs):
            if self.stopped():
                raise RuntimeError("stopped!")
            return self._func(*args, **kwargs)

        def run(self):
            with self.default_sess():
                player = get_player_fn(train=False)
                while not self.stopped():
                    try:
                        score = play_one_episode(player, self.func)
                        # print("Score, ", score)
                    except RuntimeError:
                        return
                    self.queue_put_stoppable(self.q, score)

    q = queue.Queue()
    threads = [Worker(f, q) for f in predictors]

    for k in threads:
        k.start()
        time.sleep(0.1)  # avoid simulator bugs
    stat = StatCounter()
    try:
        for _ in tqdm(range(nr_eval), **get_tqdm_kwargs()):
            r = q.get()
            stat.feed(r)
        logger.info("Waiting for all the workers to finish the last run...")
        for k in threads:
            k.stop()
        for k in threads:
            k.join()
        while q.qsize():
            r = q.get()
            stat.feed(r)
    except:
        logger.exception("Eval")
    finally:
        if stat.count > 0:
            return (stat.average, stat.max)
        return (0, 0)


def eval_model_multithread(cfg, nr_eval, get_player_fn):
    func = OfflinePredictor(cfg)
    NR_PROC = min(multiprocessing.cpu_count() // 2, 8)
    mean, max = eval_with_funcs([func] * NR_PROC, nr_eval, get_player_fn)
    logger.info("Average Score: {}; Max Score: {}".format(mean, max))


class Evaluator(Triggerable):
    def __init__(self, nr_eval, input_names, output_names, get_player_fn):
        self.eval_episode = nr_eval
        self.input_names = input_names
        self.output_names = output_names
        self.get_player_fn = get_player_fn

    def _setup_graph(self):
        NR_PROC = min(multiprocessing.cpu_count() // 2, 20)
        self.pred_funcs = [self.trainer.get_predictor(
            self.input_names, self.output_names)] * NR_PROC

    def _trigger(self):
        t = time.time()
        mean, max = eval_with_funcs(
            self.pred_funcs, self.eval_episode, self.get_player_fn)
        t = time.time() - t
        if t > 10 * 60:  # eval takes too long
            self.eval_episode = int(self.eval_episode * 0.94)
        self.trainer.monitors.put_scalar('mean_score', mean)
        self.trainer.monitors.put_scalar('max_score', max)


def play_n_episodes(player, predfunc, nr):
    logger.info("Start evaluation: ")
    for k in range(nr):
        if k != 0:
            player.restart_episode()
        score = play_one_episode(player, predfunc)
        print("{}/{}, score={}".format(k, nr, score))

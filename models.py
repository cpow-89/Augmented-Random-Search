import numpy as np
import os
import helper
import glob
import copy
import gym
import pybullet_envs
from normalizer import OnlineNormalizer
import multiprocessing
import collections
from gym import wrappers
import time


class ArsWorker(object):
    def __init__(self, weights, env_name, episode_length, index):
        self.env = gym.make(env_name)
        self.weights = weights
        self.index = index
        self.normalizer = OnlineNormalizer(nb_inputs=self.env.observation_space.shape[0])
        self.episode_length = episode_length

    def _forward(self, state):
        return self.weights.dot(state)

    def work(self):
        state = self.env.reset()
        done = False
        num_plays = 0
        sum_rewards = 0
        while not done and num_plays < self.episode_length:
            state = self.normalizer.normalize(state)
            action = self._forward(state)
            state, reward, done, _ = self.env.step(action)
            reward = np.clip(reward, -1, 1)
            sum_rewards += reward
            num_plays += 1
        return self.index, sum_rewards


def run_worker(worker, return_dict):
    return_dict[worker.index] = worker.work()


class ArsAgent(object):

    def __init__(self, config):
        self.env = gym.make(config["general"]["env_name"])
        self.env = wrappers.Monitor(self.env,
                                    os.path.join(".",
                                                 *config["general"]["monitor_dir"],
                                                 config["general"]["env_name"]),
                                    force=True)
        self.weights = np.zeros((self.env.action_space.shape[0], self.env.observation_space.shape[0]))
        self.normalizer = OnlineNormalizer(nb_inputs=self.env.observation_space.shape[0])
        self.config = config

    def _sample_deltas(self):
        return [np.random.randn(*self.weights.shape) for _ in range(self.config["agent"]["nb_directions"])]

    def _update(self, rollouts, sigma_r):
        step = np.zeros(self.weights.shape)
        for r_pos, r_neg, d in rollouts:
            step += (r_pos - r_neg) * d
        self.weights += self.config["agent"]["learning_rate"] / (self.config["agent"]["nb_best_directions"] * sigma_r) * step

    def _eval_roullouts(self, rewards_pos_deltas, rewards_neg_deltas, deltas):
        scores = {k: max(r_pos, r_neg) for k, (r_pos, r_neg) in enumerate(zip(rewards_pos_deltas, rewards_neg_deltas))}
        order = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:self.config["agent"]["nb_best_directions"]]
        rollouts = [(rewards_pos_deltas[k], rewards_neg_deltas[k], deltas[k]) for k in order]
        return rollouts

    def _eval_sigma_r(self, ordered_rewards_pos_deltas, ordered_rewards_neg_deltas):
        sum_rewards = np.array(ordered_rewards_pos_deltas + ordered_rewards_neg_deltas)
        sigma_r = sum_rewards.std()
        return sigma_r

    def _forward(self, state):
        return self.weights.dot(state)

    def evaluate(self):
        state = self.env.reset()
        done = False
        num_plays = 0
        sum_rewards = 0
        while not done and num_plays < self.config["train"]["episode_length"]:
            state = self.normalizer.normalize(state)
            action = self._forward(state)
            state, reward, done, _ = self.env.step(action)
            reward = np.clip(reward, -1, 1)
            sum_rewards += reward
            num_plays += 1
        return sum_rewards

    def _run_worker_to_evaluate_deltas(self, deltas, delta_direction="positive"):
        manager = multiprocessing.Manager()
        return_dict = manager.dict()

        delta_dir_factor = 1 if delta_direction == "positive" else -1
        jobs = []
        for idx in range(self.config["agent"]["nb_directions"]):
            worker = ArsWorker(self.weights + self.config["agent"]["noise"] * delta_dir_factor * deltas[idx],
                               copy.copy(self.config["general"]["env_name"]),
                               copy.copy(self.config["train"]["episode_length"]),
                               copy.copy(idx))

            p = multiprocessing.Process(target=run_worker, args=(worker, return_dict))
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()

        return collections.OrderedDict(sorted(return_dict.values()))

    def train(self):

        start_time = time.time()

        for episode in range(self.config["train"]["nb_episodes"]):
            deltas = self._sample_deltas()

            rewards_pos_deltas = self._run_worker_to_evaluate_deltas(deltas, "positive")
            rewards_neg_deltas = self._run_worker_to_evaluate_deltas(deltas, "negative")

            # compute the standard deviation of these rewards
            sigma_r = self._eval_sigma_r(list(rewards_pos_deltas.values()), list(rewards_neg_deltas.values()))

            rollouts = self._eval_roullouts(list(rewards_pos_deltas.values()),
                                            list(rewards_neg_deltas.values()),
                                            deltas)

            # Updating agent policy
            self._update(rollouts, sigma_r)
            sum_rewards = np.array(list(rewards_pos_deltas.values()) + list(rewards_neg_deltas.values()))
            mean_worker_rewards = sum_rewards.mean()
            print("Worker Reward mean: {}".format(mean_worker_rewards))

        print("--- Total Train time %s seconds ---" % (time.time() - start_time))

    def save_model(self):
        helper.mkdir(os.path.join(".",
                                  *self.config["general"]["checkpoint_path"],
                                  self.config["general"]["env_name"]))
        current_date_time = helper.get_current_date_time()
        current_date_time = current_date_time.replace(" ", "__").replace("/", "_").replace(":", "_")
        np.save(os.path.join(".",
                             *self.config["general"]["checkpoint_path"],
                             self.config["general"]["env_name"],
                             "ckpt_" + current_date_time),
                self.weights)

    def load_model(self):
        list_of_files = glob.glob(os.path.join(".",
                                               *self.config["general"]["checkpoint_path"],
                                               self.config["general"]["env_name"],
                                               "*"))
        latest_file = max(list_of_files, key=os.path.getctime)
        self.weights = np.load(latest_file)

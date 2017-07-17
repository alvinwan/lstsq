import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import gym
from gym import wrappers
import os
import os.path as osp
from gym import spaces
import random
import time

t0 = time.time()

ENV_ID = 'SpaceInvaders-v4'

def atari_model(img_in, num_actions, scope, reuse=False):
    # as described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    with tf.variable_scope(scope, reuse=reuse):
        out = img_in
        with tf.variable_scope("convnet"):
            # original architecture
            out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
        out = layers.flatten(out)
        with tf.variable_scope("action_value"):
            fc = out = layers.fully_connected(out, num_outputs=512,         activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)

        return fc, out

def set_global_seeds(i):
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)

def _process_frame84(frame):
    img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
    img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
    resized_screen = cv2.resize(img, (84, 110),  interpolation=cv2.INTER_LINEAR)
    x_t = resized_screen[18:102, :]
    x_t = np.reshape(x_t, [84, 84, 1])
    return x_t.astype(np.uint8)

class ProcessFrame84(gym.Wrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1))

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        return _process_frame84(obs), reward, done, info

    def _reset(self):
        return _process_frame84(self.env.reset())

def get_session(config_kwargs={}):
    tf.reset_default_graph()
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1,
        **config_kwargs)
    session = tf.Session(config=tf_config)
    print("AVAILABLE GPUS: ", get_available_gpus())
    return session


def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.physical_device_desc for x in local_device_protos if x.device_type == 'GPU']


def wrap_custom(env):
    env = ProcessFrame84(env)
    return env


def get_custom_env(env_id, seed):
    env = gym.make(env_id)

    set_global_seeds(seed)
    env.seed(seed)

    expt_dir = './tmp/'
    env = wrappers.Monitor(env, osp.join(expt_dir, "gym"), force=True)
    env = wrap_custom(env)

    return env

def get_dqn(save_path='/data/alvin/deep-q-learning/checkpoints/561527/step-final.ckpt', session_config_kwargs={}):
    seed = 0
    env = get_custom_env(ENV_ID, seed)
    num_actions = env.action_space.n
    img_h, img_w, img_c = env.observation_space.shape
    frame_history_len = 4
    input_shape = (img_h, img_w, frame_history_len * img_c)

    session = get_session(session_config_kwargs)
    # set up placeholders
    # placeholder for current observation (or state)
    obs_t_ph = tf.placeholder(tf.uint8, [None] + list(input_shape))
    # casting to float on GPU ensures lower data transfer times.
    obs_t_float = tf.cast(obs_t_ph, tf.float32) / 255.0

    global_vars = tf.GraphKeys.GLOBAL_VARIABLES
    fc, out = atari_model(obs_t_float, num_actions, scope='q_func')

    saver = tf.train.Saver()
    saver.restore(session, save_path)
    return session, fc, out, obs_t_ph


def x_to_fc5(x, session, fc, out, obs_t_ph):
    return session.run([fc], {obs_t_ph: x})[0]


def x_to_action(x, session, fc, out, obs_t_ph):
    return np.argmax(session.run([out], {obs_t_ph: x})[0])

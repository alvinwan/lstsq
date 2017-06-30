from .interface import FeaturizeInterface
from path import Path

import numpy as np


# moved tensorflow imports into functions to avoid tensorflow ipmort issues


class AtariConv3(FeaturizeInterface):
    """Project into subspace."""

    def __init__(
                self,
                path: Path,
                env,
                frame_history_len: int=4):
        super(AtariConv3, self).__init__(path, env)
        self.frame_history_len = frame_history_len
        self.session = get_session()

        import tensorflow as tf

        # initialize placeholders
        self.obs_t_ph = tf.placeholder(
            tf.uint8, [None] + list(self.get_input_shape()))
        self.obs_t_float = tf.cast(self.obs_t_ph, tf.float32) / 255.0

        num_actions = self.env.action_space.n
        self.network = atari_model(self.obs_t_float, num_actions, 'q_func')

    def phi(self, X: np.ndarray, model) -> np.array:
        """Use neural network to find lower dimensional representation."""
        raise NotImplementedError()

    def get_action(self, X: np.array, network) -> int:
        curr_q_eval = self.session.run([network], {self.obs_t_ph: X})
        action = np.argmax(curr_q_eval)
        return np.asscalar(action)

    def train(self, X: np.array, _, param: str):
        raise UserWarning('Use github.com/alvinwan/deep-q-learning to train.')

    def get_input_shape(self):
        img_h, img_w, img_c = self.env.observation_space.shape
        input_shape = (img_h, img_w, self.frame_history_len * img_c)
        return input_shape

    def load_model(self, save_path: str='step-final.ckpt'):
        """Load checkpoint from model dir/. Updates self.

        Looks for save_path in data/<name>-enc/
        """
        import tensorflow as tf
        saver = tf.train.Saver()
        # path = os.path.join(self.path.encoded_dir, save_path)
        # path ='data/raw-atari-model-84/step-final.ckpt'
        path = 'data/raw-atari-model/step-2640000.ckpt'
        saver.restore(self.session, path)
        print(' * Restore from', path)
        return self.network


#############
# UTILITIES #
#############


def get_session():
    import tensorflow as tf
    tf.reset_default_graph()
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    session = tf.Session(config=tf_config)
    print("AVAILABLE GPUS: ", get_available_gpus())
    return session


def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.physical_device_desc for x in local_device_protos if x.device_type == 'GPU']


def atari_model(img_in, num_actions: int, scope: str, reuse=False):
    import tensorflow.contrib.layers as layers
    import tensorflow as tf
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
            out = layers.fully_connected(out, num_outputs=512,         activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)

        return out
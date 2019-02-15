import tensorflow as tf
import itertools
import numpy as np


class Space(object):

    def __init__(self, space_in):
        self.check_space_in(space_in)
        self.space = space_in
        self.combinations = Space.get_combinations(space_in)

    @classmethod
    def check_space_in(cls, space_in):
        assert isinstance(space_in, dict), 'Error:Input space should be a dictionary.'
        for value in space_in.values():
            assert isinstance(value, list), 'Error:Space value should be a list.'

    @classmethod
    def get_combinations(cls, space):
        space_sets   = list(map(list, space.values()))
        combinations = list(itertools.product(*space_sets))
        return combinations

    def get_para(self, index):
        values = self.combinations[index]
        keys   = list(self.space.keys())
        para   = dict()
        for i in range(len(keys)):
            para[keys[i]] = values[i]
        return para


class Network(object):

    def __init__(self, input_layer):
        self.input_layer = input_layer

    @property
    def num_layer(self):
        return self.layer_names

    @property
    def layer_names(self):
        return list(self.__dict__.keys())

    def build_layers(self, layer_dict):
        layer_names = list(layer_dict.keys())
        for name in layer_names:
            current_name = list(self.__dict__.keys())
            assert name not in current_name, 'Error: Duplicated layer names.'

            func_name    = layer_dict[name]['func_name']
            input_arg    = layer_dict[name]['input_arg']
            input_name   = current_name[-1]
            layer_para   = layer_dict[name]['layer_para']

            layer_para[input_arg] = getattr(self, input_name)
            layer_func = getattr(LayerFactory, func_name)
            setattr(self, name, layer_func()(**layer_para))

    def add_layer_duplicates(self, layer_dict, n_copy):
        num_layer     = 0
        layer_names   = list(layer_dict.keys())
        for i in range(n_copy):
            num_layer += 1
            for name in layer_names:
                current_names = list(self.__dict__.keys())
                input_name    = current_names[-1]
                new_name      = name + '_' + str(num_layer)

                assert new_name not in current_names, 'Error: Duplicated layer names.'
                new_layer_dict = {new_name: layer_dict[name]}
                new_layer_dict[new_name]['input_name'] = input_name
                self.build_layers(new_layer_dict)


class LayerFactory(object):

    @staticmethod
    def fully_connected():
        return tf.contrib.layers.fully_connected

    @staticmethod
    def dense():
        return tf.layers.dense

    @staticmethod
    def flatten():
        return tf.layers.flatten

    @staticmethod
    def dropout():
        return tf.layers.dropout

    @staticmethod
    def softmax():
        return tf.contrib.layers.softmax


class Agent(object):

    def __init__(self):
        self.network  = dict()
        self.variable = dict()

    def add_network(self, name, network):
        self.network[name] = network

    def add_variable(self, name, variable):
        self.variable[name] = variable


class Exploration(object):

    @staticmethod
    def random(n_actions):
        action_idx = np.random.randint(low=0, high=n_actions)
        return action_idx

    @staticmethod
    def greedy(session, tf_target, feed_dict):
        tf_output  = session.run(tf_target, feed_dict=feed_dict)
        action_idx = np.argmax(tf_output, axis=None)
        return action_idx

    @staticmethod
    def e_greedy(session, tf_target, feed_dict, n_actions, e=0.1):
        if np.random.rand(1) < e:
            action_idx = Exploration.random(n_actions)
        else:
            action_idx = Exploration.greedy(session, tf_target, feed_dict)
        return action_idx

    @staticmethod
    def boltzmann(session, tf_target, feed_dict, n_actions):
        tf_prob = session.run(tf_target, feed_dict=feed_dict)
        action_idx = np.random.choice(n_actions, p=tf_prob)
        return action_idx


class ExperienceBuffer(object):

    def __init__(self, buffer_size, sample_width):
        self.buffer = np.empty((0, sample_width), float)
        self.buffer_size = buffer_size

    def add_sample(self, experience, is_single_sample):
        if is_single_sample:
            total_length = self.buffer.shape[0] + 1
        else:
            total_length = self.buffer.shape[0] + len(experience)

        if total_length > self.buffer_size:
            idx_start = total_length - self.buffer_size
            self.buffer = np.vstack([self.buffer[idx_start:], experience])
        else:
            self.buffer = np.vstack([self.buffer, experience])

    def get_sample(self, size):
        return self.buffer[np.random.randint(len(self.buffer), size=size)]


class MultiBandit(object):

    def __init__(self, strategy, state_space, action_space, explore_method, replay=True, lr=0.001):
        self.strategy       = strategy
        self.state_space    = state_space
        self.action_space   = action_space
        self.explore_method = explore_method
        self.replay         = replay
        self.lr             = lr



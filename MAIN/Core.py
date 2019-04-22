import tensorflow as tf
import itertools
import random
import numpy as np


class Agent(object):

    def __init__(self, network, config):
        self.session   = None
        self.network   = network
        self.config    = config
        self.data      = dict()

    def initialize_global(self):
        init = tf.global_variables_initializer()
        self.session.run(init)

    def set_session(self, session):
        self.session = session


class Network(object):

    def __init__(self, input_layer):
        self.input_layer  = input_layer

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
            layer_func = LayerFactory.get_func(func_name)
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

    @classmethod
    def get_func(cls, method):
        return getattr(cls, method)

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


class Space(object):

    def __init__(self, space):
        self.check_space(space)
        self.space = space
        self.n_combination, self.indices, self.multipliers = Space.get_attribute(space)
        self.idx_range = range(len(self.indices))

    @classmethod
    def check_space(cls, space):
        assert isinstance(space, dict), 'Error:Input space should be a dictionary.'
        for value in space.values():
            assert isinstance(value, list), 'Error:Space value should be a list.'

    @classmethod
    def get_attribute(cls, space):
        n_element  = [len(space[key]) for key in space.keys()]
        multiplier = [1]
        for i in range(-1, -len(n_element), -1):
            prod = multiplier[-1] * n_element[i]
            multiplier.append(prod)
        multiplier.reverse()

        multiplier  = tuple(multiplier)
        space_index = tuple([list(range(n)) for n in n_element])
        n_comb      = np.product(n_element)
        return n_comb, space_index, multiplier

    def get_combinations(self):
        space_keys   = list(self.space.keys())
        space_sets   = list(map(list, self.space.values()))
        combinations = list(itertools.product(*space_sets))
        comb_list    = [dict(zip(space_keys, element)) for element in combinations]
        return comb_list

    def get_random_sample(self, method):
        indices = [random.choice(idx) for idx in self.indices]
        if   method == 'indices':
            return indices
        elif method == 'index':
            return self.__indices_to_index(indices)
        elif method == 'one_hot':
            return self.__indices_to_one_hot(indices)
        elif method == 'dict':
            return self.__indices_to_dict(indices)
        else:
            raise ValueError('Error: Method should be indices/index/one_hot/dict.')

    def convert(self, sample, method):
        method = '_%s__' % self.__class__.__name__ + method
        return getattr(self, method)(self, sample)

    def __indices_to_index(self, indices):
        index = sum([indices[i] * self.multipliers[i] for i in self.idx_range])
        return index

    def __indices_to_one_hot(self, indices):
        index  = self.__indices_to_index(indices)
        output = self.__index_to_one_hot(index)
        return output

    def __indices_to_dict(self, indices):
        output = dict()
        keys   = list(self.space.keys())
        for i in self.idx_range:
            output[keys[i]] = self.space[keys[i]][indices[i]]
        return output

    def __index_to_indices(self, index):
        mod = index
        output = list(np.zeros(self.idx_range[-1] + 1, dtype=int))
        for i in self.idx_range:
            div, mod = divmod(mod, self.multipliers[i])
            output[i] = div
            if mod == 0:
                break
        return output

    def __index_to_one_hot(self, index):
        output = np.zeros((1, self.n_combination), dtype=int)
        output[0][index] = 1
        return output

    def __index_to_dict(self, index):
        indices = self.__index_to_indices(index)
        output  = self.__indices_to_dict(indices)
        return output

    def __one_hot_to_index(self, one_hot, axis=None):
        index = np.argmax(one_hot, axis=axis)
        return index

    def __one_hot_to_indices(self, one_hot):
        index  = self.__one_hot_to_index(one_hot)
        output = self.__index_to_indices(index)
        return output

    def __one_hot_to_dict(self, one_hot):
        index  = self.__one_hot_to_index(one_hot)
        output = self.__index_to_dict(index)
        return output

    def __dict_to_indices(self, dict_in):
        output = [self.space[key].index(value) for key, value in dict_in.items()]
        return output

    def __dict_to_index(self, dict_in):
        indices = self.__dict_to_indices(dict_in)
        index   = self.__indices_to_index(indices)
        return index

    def __dict_to_one_hot(self, dict_in):
        index  = self.__dict_to_index(dict_in)
        output = self.__index_to_one_hot(index)
        return output


class StepCounter(object):

    def __init__(self, start_num, end_num, n_annealing, n_dummy=0, is_descend=True):
        self.start_num  = start_num
        self.end_num    = end_num
        self.step_size  = abs((start_num - end_num) / n_annealing)
        self.n_dummy    = n_dummy
        self.is_descend = is_descend
        self.value      = start_num
        self.n_step     = 0
        self.is_ended   = False

    def reset(self):
        self.value = self.start_num
        self.n_step     = 0

    def step(self):
        if self.is_ended is False:
            self.n_step += 1
            if (self.value != self.end_num) & (self.n_step >= self.n_dummy):
                if self.is_descend is True:
                    self.__step_down()
                elif self.is_descend is False:
                    self.__step_up()
                else:
                    raise ValueError("Error: Boolean value required for input is_descend.")

    def __step_down(self):
        self.value -= self.step_size
        if self.value < self.end_num:
            self.is_ended = True
            self.value    = self.end_num

    def __step_up(self):
        self.value += self.step_size
        if self.value > self.end_num:
            self.is_ended = True
            self.value    = self.end_num


import random
import importlib.util
import numpy as np
from MAIN.Core import Space


class StateSpace(Space):

    def __init__(self, agent):
        state = agent.config['PROCESSOR']['MachineLearning']['StateSpace']['state']
        self.agent = agent
        self.space = super(object).__init__(state)

    def process(self, method):
        if method == 'get':
            state = self.__get_state()
            self.agent.data['Network_input'] = state
        elif method == 'convert':
            state = self.__convert_state()
            self.agent.data['state'] = state
        else:
            raise ValueError("Error: method name should be get/convert.")

    def __get_state(self):
        method = self.agent.config['PROCESSOR']['MachineLearning']['StateSpace']['sampling_method']
        state  = self.space.get_random_sample(method)
        return state

    def __convert_state(self):
        method = self.agent.config['PROCESSOR']['MachineLearning']['StateSpace']['convert_method']
        state  = self.agent.data['Network_input']
        state  = self.agent.space.convert(state, method)
        return state


class ActionSpace(Space):

    def __init__(self, agent):
        action = agent.config['PROCESSOR']['MachineLearning']['ActionSpace']['action']
        self.agent = agent
        self.space = super(object).__init__(action)

    def process(self, method):
        if method == 'get':
            action = self.__get_action()
            self.agent.data['Network_action'] = action
        elif method == 'convert':
            action = self.__convert_action()
            self.agent.data['action'] = action
        else:
            raise ValueError("Error: method name should be get/convert.")

    def __get_action(self):
        method = self.agent.config['PROCESSOR']['MachineLearning']['ActionSpace']['sampling_method']
        if method == 'exploration':
            self.agent.exploration.process()
            action = self.agent.data['Exploration_action']
        else:
            action = self.space.get_random_sample(method)
        return action

    def __convert_action(self):
        method = self.agent.config['PROCESSOR']['MachineLearning']['ActionSpace']['convert_method']
        index  = self.agent.data['Exploration_action']
        action = self.space.convert(index, method)
        return action


class RewardEngine(object):

    def __init__(self, agent):
        mod_path = agent.config['PROCESSOR']['MachineLearning']['RewardEngine']['path']
        mod_name = agent.config['PROCESSOR']['MachineLearning']['RewardEngine']['module']
        engine   = agent.config['PROCESSOR']['MachineLearning']['RewardEngine']['engine']
        inputs   = agent.config['PROCESSOR']['MachineLearning']['RewardEngine']['inputs']

        import_spec   = importlib.util.spec_from_file_location(mod_name, mod_path)
        import_module = importlib.util.module_from_spec(import_spec)
        import_spec.loader.exec_module(import_module)

        self.agent  = agent
        self.engine = import_module.get_src_cls(engine)(inputs)

    def process(self):
        self.agent.data['reward'] = self.__get_reward()

    def __get_reward(self):
        state  = self.agent.data['state']
        action = self.agent.data['action']
        reward = self.engine.process(**state, **action)
        return reward


class Exploration(object):

    def __init__(self, agent):
        method = agent.config['PROCESSOR']['MachineLearning']['Exploration']['method']
        self.agent = agent
        self.func  = Exploration.get_func(method)

    def process(self):
        action_idx = self.func()
        self.agent.data['Exploration_action'] = action_idx

    @classmethod
    def get_func(cls, method):
        return getattr(cls, method)

    def __random(self):
        n_action = self.agent.action_space.n_combination
        action_idx = random.randrange(n_action)
        return action_idx

    def __greedy(self):
        q_value = self.agent.session.run(self.agent.output_layer, feed_dict=self.agent.feed_dict)
        action_idx = np.argmax(q_value, axis=1)
        return action_idx

    def __e_greedy(self):
        e = self.agent.config['PROCESSOR']['MachineLearning']['Exploration']['e']
        if random.random() < e:
            action_idx = self.__random()
        else:
            action_idx = self.__greedy()
        e.step()
        return action_idx

    def __boltzmann(self):
        prob = self.agent.session.run(self.agent.prob_layer, feed_dict=self.agent.feed_dict)
        action_idx = np.random.choice(self.agent.action_space.n_combination, p=prob)
        return action_idx


class ExperienceBuffer(object):

    def __init__(self, agent):
        buffer_size  = agent.config['PROCESSOR']['MachineLearning']['ExperienceBuffer']['buffer_size']
        sample_width = agent.config['PROCESSOR']['MachineLearning']['ExperienceBuffer']['sample_width']
        self.agent   = agent
        self.buffer  = np.empty((0, sample_width), float)
        self.buffer_size = buffer_size

    def process(self, method):
        if method == 'add':
            self.__add_sample()
        elif method == 'get':
            sample = self.__get_sample()
            self.agent.data['ExperienceBuffer_sample'] = sample
        else:
            raise ValueError("Error: method name should be add/get.")

    def __add_sample(self):
        experience = self.agent.data['sample']
        is_single_sample = True if len(experience) == 1 else False

        if is_single_sample is True:
            total_length = self.buffer.shape[0] + 1
        elif is_single_sample is False:
            total_length = self.buffer.shape[0] + len(experience)
        else:
            raise ValueError("Error: Boolean value required for input is_single_sample.")

        if total_length > self.buffer_size:
            idx_start = total_length - self.buffer_size
            self.buffer = np.vstack([self.buffer[idx_start:], experience])
        else:
            self.buffer = np.vstack([self.buffer, experience])

    def __get_sample(self):
        size   = self.agent.config['PROCESSOR']['MachineLearning']['ExperienceBuffer']['sampling_size']
        sample = self.buffer[np.random.randint(len(self.buffer), size=size)]
        return sample


class Recorder(object):

    def __init__(self, agent):
        data_field  = agent.config['PROCESSOR']['MachineLearning']['Recorder']['data_field']
        self.agent  = agent
        self.record = {key: [] for key in data_field}

    def process(self):
        for key in self.record.keys():
            self.record[key].append(self.agent.data[key])

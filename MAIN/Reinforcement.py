import tensorflow as tf
import numpy as np
import PROCESSOR.MachineLearning as ML
from MAIN.Core import Agent


class ContextualBandit(Agent):

    def __init__(self, network, config):
        super(object).__init__(network, config)

        self.exploration   = ML.Exploration(self)
        self.exp_buffer    = ML.ExperienceBuffer(self)
        self.state_space   = ML.StateSpace(self)
        self.action_space  = ML.ActionSpace(self)
        self.reward_engine = ML.RewardEngine(self)
        self.recorder      = ML.Recorder(self)

        self.output_layer  = getattr(self.network, self.config['MAIN']['Core']['Agent']['output_layer_name'])

        self.reward_holder = tf.placeholder(shape=[1], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[1], dtype=tf.int32)

        self.data['score'] = np.zeros(self.action_space.space.n_combination)

        self.weight    = tf.slice(self.output_layer, self.action_holder, [1])
        self.loss      = -(tf.log(self.weight) * self.reward_holder)
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.config['MAIN']['Core']['Agent']['learning_rate'])
        self.update    = self.optimizer.minimize(self.loss)

    def update_network(self):
        state  = self.data['state']
        action = self.data['action']
        reward = self.data['reward']

        self.feed_dict = {self.network.input_layer: state,
                          self.action_holder: action,
                          self.reward_holder: reward}

        _, self.data['output'] = self.session.run([self.update, self.output_layer], feed_dict=self.feed_dict)

    def update_score(self):
        index  = self.data['Exploration_action']
        reward = self.data['reward']
        self.data['score'][index] += reward

    def train(self, session):
        self.session  = session
        epoch_counter = self.config['MAIN']['Core']['Agent']['Trainer']['epoch_counter']
        state_method  = self.config['MAIN']['Core']['Agent']['Trainer']['state_method']
        action_method = self.config['MAIN']['Core']['Agent']['Trainer']['action_method']

        while epoch_counter.is_ended is False:
            self.state_space.process(state_method)
            self.action_space.process(action_method)
            self.reward_engine.process()
            self.update_network()
            self.update_score()
            self.recorder.process()
            epoch_counter.step()


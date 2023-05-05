# Modified version of
# DQN implementation by Tejas Kulkarni found at
# https://github.com/mrkulk/deepQN_tensorflow

import numpy as np
import tensorflow as tf

class DQN(tf.keras.Model):
    def __init__(self, params):
        super(DQN, self).__init__()
        # self.params = params
        self.lr = params['lr']
        self.discount = params['discount']
        self.load_file = params.get('load_file', None)
        self.global_step = tf.Variable(params.get('global_step', 0), dtype=tf.int64, trainable=False)
        
        self.conv1 = tf.keras.layers.Conv2D(
            filters=16, kernel_size=3, strides=1, padding='same', activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(
            filters=32, kernel_size=3, strides=1, padding='same', activation='relu')
        self.fc3 = tf.keras.layers.Dense(units=256, activation='relu')
        self.fc4 = tf.keras.layers.Dense(units=4)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

        # Initialize the global step with the value specified in params or 0
        if 'global_step' in params:
            self.global_step = tf.Variable(params['global_step'], dtype=tf.int64, trainable=False)
        else:
            self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False)
        
        # # Load pretrained weights if specified in params
        # if 'load_file' in params and params['load_file'] is not None:
        #     self.load_ckpt(params['load_file'])
    
    def call(self, inputs):
        # print('\nnetwork call\n')
        x = inputs
        x = self.conv1(x)
        x = self.conv2(x)
        x = tf.keras.layers.Flatten()(x)
        x = self.fc3(x)
        x = self.fc4(x)

        return x

    def train_step(self, bat_s, bat_a, bat_t, bat_n, bat_r):
        with tf.GradientTape() as tape:
            tf.debugging.check_numerics(bat_s, "bat_s contains NaN or infinity values.")
            tf.debugging.check_numerics(bat_a, "bat_a contains NaN or infinity values.")
            # print('bat_t type: ', type(bat_t))
            bat_t = bat_t.astype(np.float16)
            # print('bat_t: ', bat_t)
            tf.debugging.check_numerics(bat_t, "bat_t contains NaN or infinity values.")
            tf.debugging.check_numerics(bat_n, "bat_n contains NaN or infinity values.")
            tf.debugging.check_numerics(bat_r, "bat_r contains NaN or infinity values.")

            # Forward pass through the network
            q_values = self(bat_s, training=True)
            tf.debugging.check_numerics(q_values, "q_values contains NaN or infinity values.")
            # Compute the Q-values of the next states
            # q_values_next = self(bat_n, training=False)
            # tf.debugging.check_numerics(q_values_next, "q_values_next contains NaN or infinity values.")
            # q_t = tf.reduce_max(q_values_next, axis=1)
            q_t = tf.reduce_max(q_values, axis=1)
            # Compute the target Q-values
            yj = bat_r + (1.0 - bat_t) * self.discount * q_t
            tf.debugging.check_numerics(yj, "yj contains NaN or infinity values.")
            # Compute the predicted Q-values for the selected actions
            Q_pred = tf.reduce_sum(q_values * bat_a, axis=1)
            # Compute the loss
            loss = tf.reduce_sum(tf.square(yj - Q_pred))
            tf.debugging.check_numerics(loss, "loss contains NaN or infinity values.")
        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_variables)
        # Clip gradients
        gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=1.0)
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        # Increment the global step
        self.global_step.assign_add(1)
        return loss

    def save_ckpt(self, filename):
        self.save_weights(filename)

    def load_ckpt(self, filename):
        try:
            self.load_weights(filename)
            print("Successfully loaded pretrained weights from:", filename)
        except Exception as e:
            print("Error loading pretrained weights. Reason:", e)

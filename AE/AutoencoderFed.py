import sys
import datetime
import random
from pathlib import Path

import tensorflow as tf
from keras.models import Sequential, load_model, save_model
from keras.layers import Activation
from keras.layers import Dense
from keras import backend as K


from AutoencoderUtils import AutoencoderUtils


modelsBase = Path("./models/")

EPOCH = 50
COMM_ROUND = 5
NUM_CLIENTS = 10


class AutoencoderFed():

    def __init__(self, trainFilename, anomolyFilename, log):
        self.trainFilename = trainFilename
        self.anomalyFilename = anomolyFilename
        self.autoencoderUtils = AutoencoderUtils(log)
        self.modelFilename = modelsBase / "AE.h5"
        self.log = log
        self.log.info("DEBUG: AutoencoderFed init")

    def create_clients(self, data, num_clients, initial):
        # create a list of client names
        client_names = ['{}_{}'.format(initial, i + 1) for i in range(num_clients)]

        # randomize the data
        random.shuffle(data)

        # shard data and place at each client
        size = len(data) // num_clients
        shards = [data[i:i + size] for i in range(0, size * num_clients, size)]

        assert (len(shards) == len(client_names))

        return {client_names[i]: shards[i] for i in range(len(client_names))}

    def batch_data(self, data_shard, bs=32):
        dataset = tf.data.Dataset.from_tensor_slices((list(data_shard), list(data_shard)))

        return dataset.shuffle(len(list(data_shard))).batch(bs)

    def weight_scalling_factor(self, clients_trn_data, client_name):
        client_names = list(clients_trn_data.keys())

        bs = list(clients_trn_data[client_name])[0][0].shape[0]

        global_count = sum([tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy() for client_name in client_names]) * bs

        local_count = tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy() * bs

        return local_count / global_count

    def scale_model_weights(self, weight, scalar):
        return [scalar * w_i for w_i in weight]

    def sum_scaled_weights(self, scaled_weight_list):
        avg_grad = list()

        for grad_list_tuple in zip(*scaled_weight_list):
            layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
            avg_grad.append(layer_mean)

        return avg_grad

    def build(self, input_dim):
        encoding_dim = int(input_dim / 2)
        hidden_dim = int(encoding_dim / 2)

        model = Sequential()

        model.add(Dense(encoding_dim, input_shape=(input_dim,)))
        model.add(Activation("tanh"))

        model.add(Dense(hidden_dim))
        model.add(Activation("relu"))

        model.add(Dense(encoding_dim))
        model.add(Activation("relu"))

        model.add(Dense(input_dim))

        return model

    def run(self):
        x_train, x_test, x_abnormal = self.autoencoderUtils.getData(self.trainFilename, self.anomalyFilename)

        clients = self.create_clients(x_train, num_clients=NUM_CLIENTS, initial='client')

        clients_batched = dict()
        for (client_name, data) in clients.items():
            clients_batched[client_name] = self.batch_data(data)

        # create optimizer constants
        loss = 'mean_squared_error'
        metrics = ['accuracy']
        optimizer = 'adam'

        global_model = self.build(x_train.shape[1])

        # commence global training loop
        for comm_round in range(COMM_ROUND):
            global_weights = global_model.get_weights()
            scaled_local_weight_list = list()
            client_names = list(clients_batched.keys())
            random.shuffle(client_names)

            for client in client_names:
                local_model = self.build(x_train.shape[1])
                local_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

                local_model.set_weights(global_weights)

                x_val = tf.data.Dataset.from_tensor_slices((list(x_test), list(x_test))).batch(len(list(x_test)))
                self.log.info(f'Current Training= {client}')
                local_model.fit(clients_batched[client], validation_data=x_val, epochs=EPOCH, verbose=1)

                scaling_factor = self.weight_scalling_factor(clients_batched, client)
                scaled_weights = self.scale_model_weights(local_model.get_weights(), scaling_factor)
                scaled_local_weight_list.append(scaled_weights)

                K.clear_session()

            average_weights = self.sum_scaled_weights(scaled_local_weight_list)

            global_model.set_weights(average_weights)

        save_model(global_model, str(self.modelFilename))
        model = load_model(self.modelFilename)

        self.autoencoderUtils.driver(model, 'Federated', self.trainFilename, self.anomalyFilename)
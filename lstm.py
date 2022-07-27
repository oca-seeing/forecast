import tensorflow as tf


def create_lstm_one_layer(n_past,n_features_in,n_future,n_features_out,n_hidden=100):

    # E1D1
    # n_past ==> number of timesteps in input data
    # n_features_in ==> no of features at each timestep in the input data.
    # n_future ==> number of timesteps in output data
    # n_features_out ==> no of features at each timestep in the output data.

    encoder_inputs = tf.keras.layers.Input(shape=(n_past, n_features_in))
    encoder_l1 = tf.keras.layers.LSTM(n_hidden, return_state=True)
    encoder_outputs1 = encoder_l1(encoder_inputs)

    encoder_states1 = encoder_outputs1[1:]

    #
    decoder_inputs = tf.keras.layers.RepeatVector(n_future)(encoder_outputs1[0])

    #
    decoder_l1 = tf.keras.layers.LSTM(n_hidden, return_sequences=True)(decoder_inputs,initial_state = encoder_states1)
    decoder_outputs1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features_out))(decoder_l1)

    #
    model_e1d1 = tf.keras.models.Model(encoder_inputs,decoder_outputs1)

    #
    model_e1d1.summary()

    return (model_e1d1)


def create_lstm_two_layers(n_past,n_features_in,n_future,n_features_out,n_hidden=100):

    encoder_inputs = tf.keras.layers.Input(shape=(n_past, n_features_in))
    encoder_l1 = tf.keras.layers.LSTM(n_hidden,return_sequences = True, return_state=True)
    encoder_outputs1 = encoder_l1(encoder_inputs)
    encoder_states1 = encoder_outputs1[1:]
    encoder_l2 = tf.keras.layers.LSTM(n_hidden, return_state=True)
    encoder_outputs2 = encoder_l2(encoder_outputs1[0])
    encoder_states2 = encoder_outputs2[1:]
    #
    decoder_inputs = tf.keras.layers.RepeatVector(n_future)(encoder_outputs2[0])
    #
    decoder_l1 = tf.keras.layers.LSTM(n_hidden, return_sequences=True)(decoder_inputs,initial_state = encoder_states1)
    decoder_l2 = tf.keras.layers.LSTM(n_hidden, return_sequences=True)(decoder_l1,initial_state = encoder_states2)
    decoder_outputs2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features_out))(decoder_l2)
    #
    model_e2d2 = tf.keras.models.Model(encoder_inputs,decoder_outputs2)
    #
    model_e2d2.summary()

    return(model_e2d2)
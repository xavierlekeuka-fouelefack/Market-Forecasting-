from keras.layers import LSTM, GRU
from keras.layers import RepeatVector
from keras.layers import Input
from keras.layers import TimeDistributed
from keras.layers import Dense
from keras.models import Model

def encoder(n_in, latent_dim, input_dim):
    """
    creates a model of encoder & decoder
    :param n_in: dimension inside the encoder
    :param latent_dim: dimension of the latent space
    :param input_dim: input dimension
    :return: encoder & decoder models
    """
    inputs = Input(shape=(n_in, input_dim))
    encoder = LSTM(latent_dim, activation='tanh')(inputs)
    decoder = RepeatVector(n_in)(encoder)
    decoder = LSTM(latent_dim, activation='tanh', return_sequences=True)(decoder)
    decoder = TimeDistributed(Dense(input_dim))(decoder)
    encoder_model = Model(inputs=inputs, outputs=[encoder])
    model = Model(inputs=inputs, outputs=[decoder])
    model.compile(optimizer='adam', loss='mse')
    return encoder_model, decoder, model

########################## Not USED FUNCTION IN THE MODEL###########################################

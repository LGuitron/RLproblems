from keras.models import Sequential, Model
from keras.layers import Dense, Concatenate, Input, Conv2D, Flatten
from keras.models import load_model

def compile_model(board_size):
    board_input   = Input(shape=(board_size[0],board_size[1],2), name='board')
    board_conv1   = Conv2D(16, (4,4), strides=(1, 1), activation='relu', padding='same')(board_input)
    board_conv2   = Conv2D(16, (4,4), strides=(1, 1), activation='relu', padding='same')(board_conv1)
    board_conv3   = Conv2D(16, (4,4), strides=(1, 1), activation='relu', padding='same')(board_conv2)
    board_conv4   = Conv2D(32, (3,3), strides=(1, 1), activation='relu')(board_conv3)
    board_conv5   = Conv2D(64, (2,2), strides=(1, 1), activation='relu')(board_conv4)
    board_flat    = Flatten()(board_conv4)
    board_dense1   = Dense(32, activation='relu')(board_flat)
    board_dense2   = Dense(32)(board_dense1)

    # Out layer for as many columns as the board has
    out = Dense(board_size[1], activation='tanh')(board_dense2)
    model = Model(inputs = [board_input], outputs = out)
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model, "model"

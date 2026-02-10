"""ResNet + LSTM Model."""
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, ReLU, Add, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam


def resnet_block(x, filters, kernel_size):
    shortcut = x
    x = Conv1D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv1D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([shortcut, x])
    x = ReLU()(x)
    return x


def build_resnet_lstm(sequence_length, features=1, learning_rate=0.001):
    inputs = Input(shape=(sequence_length, features))
    x = Conv1D(64, kernel_size=3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = resnet_block(x, 64, 3)
    x = resnet_block(x, 64, 3)
    x = LSTM(64, return_sequences=False)(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs, name='resnet_lstm')
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    return model

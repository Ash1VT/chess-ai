import pickle

from keras.saving.save import load_model

from config import TOKENIZER_FILE_PATH, MODEL_FILE_PATH, MAX_SEQUENCE_LENGTH_FILE_PATH, DATA_INFORMATION_FILE_PATH


def load_tokenizer():
    with open(TOKENIZER_FILE_PATH, 'rb') as file:
        return pickle.load(file)


def load_trained_model():
    return load_model(MODEL_FILE_PATH)


def load_max_sequence_length():
    with open(MAX_SEQUENCE_LENGTH_FILE_PATH, 'r') as file:
        return int(file.read())


def load_data_length_information():
    with open(DATA_INFORMATION_FILE_PATH, 'r') as file:
        return int(file.readline()), int(file.readline()), int(file.readline())

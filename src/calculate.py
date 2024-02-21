import pickle

import pandas as pd
from keras.preprocessing.text import Tokenizer

from config import PREPROCESSED_DATA_FILE_PATH, TOKENIZER_FILE_PATH, MAX_SEQUENCE_LENGTH_FILE_PATH

data = pd.read_csv(PREPROCESSED_DATA_FILE_PATH)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['moves'])

max_sequence_length = max(len(seq) for seq in tokenizer.texts_to_sequences(data['moves']))


with open(TOKENIZER_FILE_PATH, 'wb+') as tokenizer_handle, open(MAX_SEQUENCE_LENGTH_FILE_PATH, 'w+') as max_sequence_length_handle:
    pickle.dump(tokenizer, tokenizer_handle, protocol=pickle.HIGHEST_PROTOCOL)
    max_sequence_length_handle.write(str(max_sequence_length))
    print('successfully calculated')

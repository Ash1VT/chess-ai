import functools
import os
from typing import List, Tuple

from keras.saving.save import load_model

from config import PREPROCESSED_DATA_FILE_PATH, NUM_EPOCHS, SPLIT_COEFFICIENT, CHUNK_SIZE
from loaders import load_tokenizer, load_max_sequence_length
from train import train


def get_checkpoint_details(checkpoint: str) -> Tuple[int, float]:
    epoch, value_loss = checkpoint.split('_')[2].split('.')[0].split('-')
    epoch = int(epoch)
    value_loss = float(value_loss)
    return epoch, value_loss


def get_best_checkpoint(checkpoints: List[str]) -> Tuple[str, int, float]:
    best_checkpoint = functools.reduce(lambda x, y: x if get_checkpoint_details(x)[1] < get_checkpoint_details(y)[1] else y,
                            checkpoints)
    epoch, value_loss = get_checkpoint_details(best_checkpoint)
    return best_checkpoint, epoch, value_loss


checkpoints = os.listdir('../models/checkpoints')

best_checkpoint, epoch, value_loss = get_best_checkpoint(checkpoints)

print(f"Found best checkpoint: {best_checkpoint}")

model = load_model(f'../models/checkpoints/{best_checkpoint}')
tokenizer = load_tokenizer()
max_sequence_length = load_max_sequence_length()

train(model, tokenizer, max_sequence_length, SPLIT_COEFFICIENT, CHUNK_SIZE, NUM_EPOCHS, epoch)

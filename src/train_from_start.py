# Define the model
from config import SPLIT_COEFFICIENT, CHUNK_SIZE, NUM_EPOCHS
from loaders import load_tokenizer, load_max_sequence_length
from model import get_model
from train import train

# Load information about data
tokenizer = load_tokenizer()
max_sequence_length = load_max_sequence_length()
model = get_model(tokenizer, max_sequence_length)


train(model, tokenizer, max_sequence_length, SPLIT_COEFFICIENT, CHUNK_SIZE, NUM_EPOCHS)

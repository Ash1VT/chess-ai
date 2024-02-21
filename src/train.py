import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split

from checkpoint import CustomModelCheckpointCallback
from config import PREPROCESSED_DATA_FILE_PATH, MODEL_FILE_PATH, NUM_EPOCHS, SPLIT_COEFFICIENT, \
    CHUNK_SIZE, CHECKPOINT_MODEL_FILE_PATH
from loaders import load_tokenizer, load_max_sequence_length
from model import get_model


# Define a generator function
def data_generator(data, labels, tokenizer, batch_size, max_sequence_length):
    num_samples = len(data)
    while True:
        for offset in range(0, num_samples, batch_size):
            batch_data = data[offset:offset + batch_size]
            batch_labels = labels[offset:offset + batch_size]

            # Convert moves to sequences
            sequences = tokenizer.texts_to_sequences(batch_data)

            # Pad sequences to make them of equal length
            padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

            yield padded_sequences, np.array(batch_labels)


def train(model, tokenizer, max_sequence_length, split_coefficient, batch_size, epochs, start_epoch=0):

    # Load the preprocessed data
    data = pd.read_csv(PREPROCESSED_DATA_FILE_PATH)
    X = data['moves']
    y = data['result']
    del data

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - split_coefficient, random_state=42)
    del X, y

    # Create generator instances
    train_generator = data_generator(X_train, y_train, tokenizer, batch_size, max_sequence_length)
    test_generator = data_generator(X_test, y_test, tokenizer, batch_size, max_sequence_length)

    # Calculate steps per epoch and validation steps
    steps_per_epoch = len(X_train) // batch_size
    validation_steps = len(X_test) // batch_size

    # Define callbacks for early stopping and model checkpointing
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(CHECKPOINT_MODEL_FILE_PATH, monitor='val_loss', verbose=1)

    # Train the model using generator
    model.fit(train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs, validation_data=test_generator,
              validation_steps=validation_steps, batch_size=batch_size, initial_epoch=start_epoch, callbacks=[early_stopping, model_checkpoint])

    # Evaluate the model
    loss, accuracy = model.evaluate(test_generator, steps=validation_steps)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

    # Save the model
    model.save(MODEL_FILE_PATH)

# Define a function to preprocess moves and predict the outcome
from keras.utils import pad_sequences


def predict_outcome(moves, tokenizer, model) -> str:
    # Convert moves to sequence
    sequences = tokenizer.texts_to_sequences([moves])

    # Pad sequence
    padded_sequence = pad_sequences(sequences)

    # Predict outcome
    prediction = model.predict(padded_sequence)[0][0]

    return f"{prediction * 100:.2f}"

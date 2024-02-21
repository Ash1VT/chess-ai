from pathlib import Path

# PANDAS
CHUNK_SIZE = 128

# DIRECTORIES
BASE_DIR = Path(__file__).absolute().parent.parent
SRC_DIR = BASE_DIR / 'src'

# MULTIPROCESSING
WORKERS_COUNT = 10

# MODEL
NUM_EPOCHS = 15
TOKENIZER_FILE_PATH = f'{BASE_DIR}/models/tokenizer.pickle'
MAX_SEQUENCE_LENGTH_FILE_PATH = f'{BASE_DIR}/models/max_sequence_length.txt'
MODEL_FILE_PATH = f'{BASE_DIR}/models/chess.h5'
CHECKPOINT_MODEL_FILE_PATH = f'{BASE_DIR}/models/checkpoints/chess_checkpoint' + '_{epoch:02d}-{val_loss:.4f}.h5'
DATA_INFORMATION_FILE_PATH = f'{BASE_DIR}/models/data_information.txt'

# FILES
RAW_DATA_FILE_PATH = f'{BASE_DIR}/data/all_with_filtered_anotations_since1998.txt'
PREPROCESSED_DATA_FILE_PATH = f'{BASE_DIR}/data/preprocessed_data.csv'
TRAINING_DATA_FILE_PATH = f"{BASE_DIR}/data/training/training_data.csv"
TESTING_DATA_FILE_PATH = f"{BASE_DIR}/data/test/testing_data.csv"

# DATA
SPLIT_COEFFICIENT = 0.8

# PYGAME
SQUARE_SIZE = 64
BOARD_SQUARES_COUNT = 8
BOARD_WIDTH = SQUARE_SIZE * (BOARD_SQUARES_COUNT + 1)
BOARD_HEIGHT = SQUARE_SIZE * (BOARD_SQUARES_COUNT + 1)

STATISTICS_WIDTH = 256
STATISTICS_HEIGHT = BOARD_HEIGHT

DISPLAY_WIDTH = STATISTICS_WIDTH + BOARD_WIDTH
DISPLAY_HEIGHT = BOARD_HEIGHT

# Colors
LIGHT_SQUARE_COLOR = (255, 206, 158)
DARK_SQUARE_COLOR = (209, 139, 71)
SELECTED_SQUARE_COLOR = (0, 255, 0)

# Figures images
FIGURES = {
    'b': 'black_bishop.svg',
    'n': 'black_knight.svg',
    'r': 'black_rook.svg',
    'q': 'black_queen.svg',
    'k': 'black_king.svg',
    'p': 'black_pawn.svg',
    'B': 'white_bishop.svg',
    'N': 'white_knight.svg',
    'R': 'white_rook.svg',
    'Q': 'white_queen.svg',
    'K': 'white_king.svg',
    'P': 'white_pawn.svg',
}
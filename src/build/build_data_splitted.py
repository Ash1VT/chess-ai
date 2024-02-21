import csv
import multiprocessing as mp
import random
import sys

from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.build.validation import process_line
from src.config import WORKERS_COUNT, RAW_DATA_FILE_PATH, TRAINING_DATA_FILE_PATH, TESTING_DATA_FILE_PATH, \
    SPLIT_COEFFICIENT, DATA_INFORMATION_FILE_PATH, PREPROCESSED_DATA_FILE_PATH, CHUNK_SIZE


def preprocess_data(input_file: str, common_output_file: str, validation_output_file: str, testing_output_file: str,
                    split_coefficient: float):
    with open(input_file, "r") as infile, \
            open(common_output_file, "w+", newline='') as common_outfile, \
            open(validation_output_file, "w+", newline='') as validation_outfile, \
            open(testing_output_file, "w+", newline='') as testing_outfile:
        common_writer = csv.writer(common_outfile)
        validation_writer = csv.writer(validation_outfile)
        testing_writer = csv.writer(testing_outfile)

        # Write headers
        common_writer.writerow(['result', 'moves'])
        validation_writer.writerow(['result', 'moves'])
        testing_writer.writerow(['result', 'moves'])

        # Define a pool of worker processes
        pool = mp.Pool(WORKERS_COUNT)

        # Read first 5 dump lines
        lines_to_read = 5
        _ = [infile.readline() for _ in range(lines_to_read)]

        # Read all lines from the input file
        lines = infile.readlines()

        # Shuffle the lines randomly
        random.shuffle(lines)

        # Get count of invalidated lines
        total_lines_count = len(lines)
        training_lines_count = int(split_coefficient * total_lines_count)

        # Split the lines between training and testing
        training_lines = lines[:training_lines_count]
        test_lines = lines[training_lines_count:]
        del lines

        # Define counters for valid lines
        valid_training_lines_count = 0
        valid_test_lines_count = 0

        # Write lines to common, validation and testing files
        for row in pool.imap_unordered(process_line, training_lines, chunksize=CHUNK_SIZE):
            if row:
                common_writer.writerow(row)
                validation_writer.writerow(row)
                valid_training_lines_count += 1

        for row in pool.imap_unordered(process_line, test_lines, chunksize=CHUNK_SIZE):
            if row:
                common_writer.writerow(row)
                testing_writer.writerow(row)
                valid_test_lines_count += 1

        total_valid_lines_count = valid_training_lines_count + valid_test_lines_count

        # Write information about valid data
        with open(DATA_INFORMATION_FILE_PATH, 'w+') as file:
            file.write(str(total_valid_lines_count) + '\n')
            file.write(str(valid_training_lines_count) + '\n')
            file.write(str(valid_test_lines_count))


if __name__ == '__main__':
    mp.freeze_support()
    preprocess_data(RAW_DATA_FILE_PATH, PREPROCESSED_DATA_FILE_PATH,
                    TRAINING_DATA_FILE_PATH, TESTING_DATA_FILE_PATH, SPLIT_COEFFICIENT)

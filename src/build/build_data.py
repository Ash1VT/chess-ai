import csv
import multiprocessing as mp
import sys

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.build.validation import process_line
from src.config import WORKERS_COUNT, RAW_DATA_FILE_PATH, PREPROCESSED_DATA_FILE_PATH


def preprocess_data(input_file, output_file):
    with open(input_file, "r") as infile, open(output_file, "w", newline='') as outfile:
        writer = csv.writer(outfile)

        # Write headers
        writer.writerow(['result', 'moves'])

        # Define a pool of worker processes
        pool = mp.Pool(WORKERS_COUNT)

        lines_to_read = 5
        _ = [infile.readline() for _ in range(lines_to_read)]

        # Process lines from the input file in parallel
        for row in pool.imap_unordered(process_line, infile, chunksize=1000):
            if not row:
                continue

            writer.writerow(row)


if __name__ == '__main__':
    mp.freeze_support()
    preprocess_data(RAW_DATA_FILE_PATH, PREPROCESSED_DATA_FILE_PATH)

import time
import os

from tqdm import tqdm

from ocr import create_dataset
from commons import DATASET_DIR


BATCH_SIZE = 32
IMG_SIZE = (320, 224) # (480, 640)  # should be dividable by 32


def main():
    dataset = create_dataset(
        DATASET_DIR.glob("train-*.tfrecord"),
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE)

    for batch in tqdm(dataset, unit="steps"):
        pass


if __name__ == "__main__":
    main()

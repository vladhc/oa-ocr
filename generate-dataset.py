import multiprocessing
import time

from ocr import write_tfrecords


def generate_dataset():
    CHUNKS = 10

    ps = []

    seed = int(time.time())

    for idx in range(10):
        p = multiprocessing.Process(
            target=write_tfrecords,
            args=(f'train-{idx}.tfrecord', 30000, (640, 480), seed + idx))
        p.start()
        ps.append(p)

    for p in ps:
        p.join()


if __name__ == "__main__":
    generate_dataset()

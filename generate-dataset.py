import multiprocessing

from ocr import write_tfrecords


def generate_dataset():
    CHUNKS = 10

    ps = []

    for idx in range(10):
        p = multiprocessing.Process(
            target=write_tfrecords,
            args=(f'eval-{idx}.tfrecord', 1000, (640, 480)))
        p.start()
        ps.append(p)

    for p in ps:
        p.join()


if __name__ == "__main__":
    generate_dataset()

import unittest

from commons import epoch_from_checkpoint


class TestCommons(unittest.TestCase):

    def test_epoch_from_checkpoint(self):
        path = "train/bool_features/model.006.hdf5"
        epoch = epoch_from_checkpoint(path)
        self.assertEqual(epoch, 6)


if __name__ == "__main__":
    unittest.main()

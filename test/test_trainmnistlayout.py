import unittest
from utils.train_mnistlayout import Trainer


class TestMnistLayout(unittest.TestCase):
    def test_generate_z(self):
        trainer = Trainer(logging=False)
        z = trainer.generate_z(size=30)
        shape = z.shape
        self.assertEqual([shape[0], shape[1], shape[2]], [
                         30, trainer.element_num, trainer.class_num + trainer.geoparam_num])

    def test_G(self):
        trainer = Trainer(logging=False)
        z = trainer.generate_z(size=30)
        pred = trainer.G(z)
        print(pred)
        self.assertEqual(z.shape, pred.shape)


if __name__ == "__main__":
    unittest.main()

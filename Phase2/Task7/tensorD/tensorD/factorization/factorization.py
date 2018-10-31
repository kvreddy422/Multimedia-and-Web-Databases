# Created by ay27 at 17/6/2
import pickle


class Model(object):
    """
    The Model class holding the
    """

    def __init__(self, before_train, in_train, after_train=None, metrics=None):
        self.before_train = before_train
        self.in_train = in_train
        self.metrics = metrics
        self.after_train = after_train

        # TODO : how to save and restore the model properly
        # def save(self, save_path):
        #     pickle.dump(self, open(save_path, 'wb'))
        #
        # @staticmethod
        # def load(save_path):
        #     return pickle.load(open(save_path, 'rb'))


class BaseFact(object):
    def build_model(self, args) -> Model:
        raise NotImplementedError

    def train(self, steps=None):
        raise NotImplementedError

    def predict(self, *key):
        raise NotImplementedError

    def full(self):
        raise NotImplementedError

    def save(self, path):
        # TODO
        pass

    @staticmethod
    def restore(path):
        # TODO
        pass

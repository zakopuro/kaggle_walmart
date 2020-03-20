import pandas as pd
from .base_create_features import Feature, get_arguments, generate_features


class test(Feature):
    def create_features(self):
        self.train['test'] = train['tt']


if __name__ == "__main__":
    args = get_arguments()

    train = pd.read_csv('input/train.csv')
    test = pd.read_csv('input/test.csv')

    generate_features(globals(),args.force)
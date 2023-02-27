import argparse
import random
import operator
import pandas as pd


class DiscreteDataCreator():

    def __init__(
            self,
            data_size: int,
            onehot: bool = False,
            correct_prob: float = 0.5,
            ):

        self._data_size = data_size
        self._onehot = onehot
        self._correct_prob = correct_prob

        # columns to create dataframe
        self._columns = ["first_number", "second_number", "operator", "answer"]
        self._onehot_columns = [
                "first_number", "second_number",
                "add", "sub", "mul",
                "answer"]

        # operator string to python operator func
        self._operators = {
                '+': operator.add,
                '-': operator.sub,
                '*': operator.mul,
                }

    # create random data
    def __rand_data(self):

        # get random integer from 1 to 10
        first_number = random.randint(1, 10)
        second_number = random.randint(1, 10)

        operator_list = list(self._operators.keys())

        # get random opeerator and create 3-tuple
        features = (
                first_number,
                second_number,
                random.choice(operator_list),
                )

        # calulate answer
        # create random wrong answer depends on self._correct_prob
        answer = self._operators[features[2]](first_number, second_number) \
            if random.random() < self._correct_prob \
            else random.randint(-9, 100)

        # if self._onehot than create operator symbol to onehot representation
        if self._onehot:
            onehot = tuple(
                    [1 if i == features[2] else 0 for i in operator_list])
            features = features[:2] + onehot

        # stack answer to features and return
        return features + (answer,)

    # create random dataframe
    def dataframe(self):
        return pd.DataFrame(
                [self.__rand_data() for i in range(self._data_size)],
                columns=self._onehot_columns
                if self._onehot
                else self._columns,
                )


class ContinuousDataCreator():

    def __init__(
            self,
            data_size: int,
            onehot: bool = False,
            correct_prob: float = 0.5,
            ):

        self._data_size = data_size
        self._onehot = onehot
        self._correct_prob = correct_prob

        # columns to create dataframe
        self._columns = ["first_number", "second_number", "operator", "answer"]
        self._onehot_columns = [
                "first_number", "second_number",
                "add", "sub", "mul", "div",
                "answer"]

        # operator string to python operator func
        self._operators = {
                '+': operator.add,
                '-': operator.sub,
                '*': operator.mul,
                '/': operator.truediv,
                }

    # create random data
    def __rand_data(self):

        # get random integer from 1 to 10
        first_number = random.uniform(1, 10)
        second_number = random.uniform(1, 10)

        operator_list = list(self._operators.keys())

        # get random opeerator and create 3-tuple
        features = (
                first_number,
                second_number,
                random.choice(operator_list),
                )

        # calulate answer
        # create random wrong answer depends on self._correct_prob
        answer = self._operators[features[2]](first_number, second_number) \
            if random.random() < self._correct_prob \
            else random.uniform(-9, 100)

        # if self._onehot than create operator symbol to onehot representation
        if self._onehot:
            onehot = tuple(
                    [1 if i == features[2] else 0 for i in operator_list])
            features = features[:2] + onehot

        # stack answer to features and return
        return features + (answer,)

    # create random dataframe
    def dataframe(self):
        return pd.DataFrame(
                [self.__rand_data() for i in range(self._data_size)],
                columns=self._onehot_columns
                if self._onehot
                else self._columns,
                )


def create_csv(
        d_type: str,
        path: str,
        data_size: int = 10000,
        correct_prob: float = 0.5,
        ):

    if d_type == "discrete":
        data_creator = DiscreteDataCreator(
                data_size=data_size,
                onehot=True,
                correct_prob=correct_prob)
    elif d_type == "continuous":
        data_creator = ContinuousDataCreator(
                data_size=data_size,
                onehot=True,
                correct_prob=correct_prob)
    else:
        raise ValueError("Use discrete or continuous as d_type parameter")

    df = data_creator.dataframe()
    df.to_csv(path, index=False)


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument("--path", required=True, type=str)
    p.add_argument("--data_size", required=True, type=int)
    p.add_argument("--d_type", required=True, type=str)
    p.add_argument("--correct_prob", required=False, type=float, default=1)

    return p.parse_args()


if __name__ == "__main__":

    config = define_argparser()

    create_csv(
            d_type=config.d_type,
            path=config.path,
            data_size=config.data_size,
            correct_prob=config.correct_prob,
            )

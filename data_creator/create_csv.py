import random
import operator
import pandas as pd


class ArithmeticDataCreator():

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


def create_csv(
        path: str,
        data_size: int = 10000,
        correct_prob: float = 0.5,
        ):

    data_creator = ArithmeticDataCreator(
            data_size=data_size,
            onehot=True,
            correct_prob=correct_prob)

    df = data_creator.dataframe()
    df.to_csv(path, index=False)


if __name__ == "__main__":
    create_csv(
            path="../data.csv",
            data_size=50000,
            correct_prob=1,
            )

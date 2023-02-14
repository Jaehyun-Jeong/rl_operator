import random
import operator
import pandas as pd


class ArithmeticDataCreator():

    def __init__(
            self,
            path: str,
            data_size: int,
            onehot: bool = False,
            ):

        self._path = path
        self._data_size = data_size
        self._onehot = onehot

        # columns to create dataframe
        self._columns = ["first_number", "oprator", "second_number", "answer"]

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
                random.choice(operator_list),
                second_number)

        if self._onehot:
            pass

        # calulate answer
        answer = self._operators[features[1]](first_number, second_number)

        # stack answer to features and return
        return features + (answer,)

    # create random dataframe
    def dataframe(self):
        return pd.DataFrame(
                [self.__rand_data() for i in range(self._data_size)],
                columns=self._columns,
                )


if __name__ == "__main__":

    data_creator = ArithmeticDataCreator("./data.csv", 100)

    print(data_creator.dataframe())

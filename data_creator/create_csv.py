import random
import operator


class ArithmeticDataCreator():

    def __init__(
            self,
            path: str,
            data_size: int,
            ):

        self._path = path
        self._data_size = data_size

        self._columns = ["first_number", "oprator", "second_number", "answer"]
        self._operators = {
                '+': operator.add,
                '-': operator.sub,
                '*': operator.mul,
                '/': operator.truediv,
                }

    def __rand_data(self):
        first_number = random.randint(1, 10)
        second_number = random.randint(1, 10)
        features = (
                first_number,
                random.choice(['+', '-', '*', '/']),
                second_number)
        answer = self._operators[features[1]](first_number, second_number)

        return features + (answer,)

    def dataframe(self):
        return self.__rand_data()


if __name__ == "__main__":

    data_creator = ArithmeticDataCreator("./data.csv", 100)

    for i in range(100):
        print(data_creator.dataframe())

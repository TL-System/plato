import numpy as np


def find_value_position(array, value):
    counter = 0
    for i in range(len(array) - 1):
        if array[i] < value and array[i + 1] > value:
            return i
        if array[i] == value:
            return i
        if array[i] > value and array[i + 1] > value:
            return i
        counter = counter + 1

    if counter == len(array) - 1:
        return len(array) - 1

    return 0

"""A module for doing bit manipulation on integers."""

from numpy.random import randint


def get_bit(number, bit_index):
    """Returns the state of the bit with the given index in a number."""
    return number >> bit_index & 1


def set_bit(number, bit_index, bit_value):
    """
    Returns the given number
    with the bit with the given index
    changed to the given value.
    """
    if bit_value:
        return number | 1 << bit_index

    return number & ~(1 << bit_index)


def find_hamming_wieght(bit_string):
    """Returns the number of bits in a bit string."""
    weight = 0
    for bit_idx in range(bit_string.bit_length()):
        if (bit_string >> bit_idx) &1:
            weight += 1
    return weight


def find_hamming_distance(bit_string1, bit_string2):
    """
    Compares 2 bit strings and returns the number of bits which are different.
    """
    return find_hamming_wieght(bit_string1 ^ bit_string2)


def flip_random_bit(num_bits, number):
    """
    Returns the given number with a random bit (up to the Nth bit) flipped.
    """
    return number ^ 1 << randint(num_bits)

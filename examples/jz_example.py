import hello
import numpy as np

hello.print_hello("Hello, world from JohnZ")

a = hello.iota(10)
n_zeros = hello.to_scalar(
    hello.count_if_zero(a))

print(f"Found {n_zeros} zeros")

n_ones = hello.to_scalar(
    hello.count_if(a, 1),
    int, np.int32
)

print(f"Found {n_ones} ones")





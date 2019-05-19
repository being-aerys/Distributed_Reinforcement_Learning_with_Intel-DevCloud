from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import ray
import time
import sys

if len(sys.argv) < 2:
    cpus = 4
else:
    cpus = int(sys.argv[1])
print('running map reduce with %d cpus' % cpus)
ray.init(num_cpus=int(cpus), include_webui=False, ignore_reinit_error=True, redis_max_memory=1000000000, object_store_memory=10000000000)

def map_serial(function, xs):
    return [function(x) for x in xs]

def map_parallel(function, xs):
    """Apply a remote function to each element of a list."""
    if not isinstance(xs, list):
        raise ValueError('The xs argument must be a list.')
    
    if not hasattr(function, 'remote'):
        raise ValueError('The function argument must be a remote function.')

    # EXERCISE: Modify the list comprehension below to invoke "function"
    # remotely on each element of "xs". This should essentially submit
    # one remote task for each element of the list and then return the
    # resulting list of ObjectIDs.
    return [function.remote(x) for x in xs]


# ***** Do not change the code below! It verifies that 
# ***** the exercise has been done correctly. *****

def increment_regular(x):
    return x + 1


@ray.remote
def increment_remote(x):
    return x + 1


xs = [1, 2, 3, 4, 5]
result_ids = map_parallel(increment_remote, xs)
assert isinstance(result_ids, list), 'The output of "map_parallel" must be a list.'
assert all([isinstance(x, ray.ObjectID) for x in result_ids]), 'The output of map_parallel must be a list of ObjectIDs.'
assert ray.get(result_ids) == map_serial(increment_regular, xs)
print('Congratulations, the test passed!')    

def sleep_regular(x):
    time.sleep(1)
    return x + 1


@ray.remote
def sleep_remote(x):
    time.sleep(1)
    return x + 1


serial_time = 0
parallel_time = 0
# Regular sleep should take 4 seconds.
# print('map_serial')
start_time = time.time()
results_serial = map_serial(sleep_regular, [1, 2, 3, 4])
serial_time += time.time() - start_time

# Initiaing the map_parallel should be instantaneous.
# print('\ncalling map_parallel:')
start_time = time.time()
result_ids = map_parallel(sleep_remote, [1, 2, 3, 4])
# Fetching the results from map_parallel should take 1 second
# (since we started Ray with num_cpus=4).
results_parallel = ray.get(result_ids)
parallel_time = time.time() - start_time

assert results_parallel == results_serial

def reduce_serial(function, xs):
    if len(xs) == 1:
        return xs[0]
    
    result = xs[0]
    for i in range(1, len(xs)):
        result = function(result, xs[i])

    return result


def add_regular(x, y):
    time.sleep(0.3)
    return x + y


assert reduce_serial(add_regular, [1, 2, 3, 4, 5, 6, 7, 8]) == 36

def reduce_parallel(function, xs):
    if not isinstance(xs, list):
        raise ValueError('The xs argument must be a list.')

    if not hasattr(function, 'remote'):
        raise ValueError('The function argument must be a remote function.')

    if len(xs) == 1:
        return xs[0]

    result = xs[0]
    for i in range(1, len(xs)):
        result = function.remote(result, xs[i])

    return result


@ray.remote
def add_remote(x, y):
    time.sleep(0.3)
    return x + y


xs = [1, 2, 3, 4, 5, 6, 7, 8]
result_id = reduce_parallel(add_remote, xs)
assert ray.get(result_id) == reduce_serial(add_regular, xs)
print('Congratulations, the test passed!')

def reduce_parallel_tree(function, xs):
    if not isinstance(xs, list):
        raise ValueError('The xs argument must be a list.')
    
    if not hasattr(function, 'remote'):
        raise ValueError('The function argument must be a remote function.')

    # The easiest way to implement this function is to simply invoke
    # "function" remotely on the first two elements of "xs" and to append
    # the result to the end of "xs". Then repeat until there is only one
    # element left in "xs" and return that element.

    # EXERCISE: Think about why that exposes more parallelism.    
    while len(xs) > 1:
        result_id = function.remote(xs[0], xs[1])
        xs = xs[2:]
        xs.append(result_id)
    return xs[0]


xs = [1, 2, 3, 4, 5, 6, 7, 8]
result_id = reduce_parallel_tree(add_remote, xs)
assert ray.get(result_id) == reduce_serial(add_regular, xs)

# Regular sleep should take 4 seconds.
# print('reduce_serial:')
start_time = time.time()
results_serial = reduce_serial(add_regular, [1, 2, 3, 4, 5, 6, 7, 8])
serial_time += time.time() - start_time

# Initiaing the map_parallel should be instantaneous.
# print('\ncalling reduce_parallel:')
result_ids = reduce_parallel(add_remote, [1, 2, 3, 4, 5, 6, 7, 8])

# Fetching the results from map_parallel should take 1 second
# (since we started Ray with num_cpus=4).
# print('\ngetting results from reduce_parallel:')
results_parallel = ray.get(result_ids)

assert results_parallel == results_serial

# Initiaing the map_parallel should be instantaneous.
print('\ncalling reduce_parallel_tree')
start_time = time.time()
result_tree_ids = reduce_parallel_tree(add_remote, [1, 2, 3, 4, 5, 6, 7, 8])
# Fetching the results from map_parallel should take 1 second
# (since we started Ray with num_cpus=4).
results_parallel_tree = ray.get(result_tree_ids)
reduce_time = time.time() - start_time
parallel_time += reduce_time 
print('reduce time:', reduce_time)
print('total time:', parallel_time)

assert results_parallel_tree == results_serial

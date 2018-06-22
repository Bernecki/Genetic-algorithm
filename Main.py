__author__ = "Paweł Bernecki"

import numpy as np
import Algorithm


def read_data(file_name):
    file = open(file_name, 'r')
    lines = file.read().splitlines()
    size = int(lines[0])
    distance = np.zeros(shape=(size, size), dtype=int)
    flow = np.zeros(shape=(size, size), dtype=int)

    current_row = 0
    for i in range(2, size + 2):
        distance[current_row, :] = (lines[i].replace('  ', ' ').split(' ')[1:])
        current_row += 1

    current_row = 0
    for i in range(size + 3, size + 3 + size):
        flow[current_row, :] = (lines[i].replace('  ', ' ').split(' ')[1:])
        current_row += 1

    file.close()
    return size, distance, flow


def main():
    file = input("Select the number of factories: 12, 14, 16, 18 or 20")
    file_name = '{0}.txt'.format(file)
    size, distance, flow = read_data(file_name)
    ga = Algorithm.GeneticAlgorithm(size, distance, flow,
                                    200, 0.5, 0.2, 50,
                                    selection='tournament',
                                    tour_size=5, caching=False)

    test_size = 1
    results_array = np.zeros(shape=(test_size, 2), dtype=int)
    for i in range(test_size):
        results = ga.run()
        results_array[i, 0] = results[0][:, 1][-1]
        results_array[i, 1] = results[0][:, 2][-1]

    ga = Algorithm.GeneticAlgorithm(size, distance, flow,
                                    50, 0.5, 0.2, 200,
                                    selection='tournament',
                                    tour_size=5, caching=False)
    results_array1 = np.zeros(shape=(test_size, 2), dtype=int)
    for i in range(test_size):
        results = ga.run()
        results_array1[i, 0] = results[0][:, 1][-1]
        results_array1[i, 1] = results[0][:, 2][-1]

    print(results_array)
    print("average of the best fitness found in 10 runs: ")
    print(np.average(results_array[:, 0], axis=0))
    print("the best fitness found in 10 runs: ")
    print(np.amin(results_array[:, 0], axis=0))
    print("average of the average fitness found in 10 runs: ")
    print(np.average(results_array[:, 1], axis=0))
    print("\n\n")
    print(results_array1)
    print("average of the best fitness found in 10 runs: ")
    print(np.average(results_array1[:, 0], axis=0))
    print("the best fitness found in 10 runs: ")
    print(np.amin(results_array1[:, 0], axis=0))
    print("average of the average fitness found in 10 runs: ")
    print(np.average(results_array1[:, 1], axis=0))


if __name__ == '__main__':
    main()

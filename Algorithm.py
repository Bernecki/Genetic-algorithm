__author__ = "Paweł Bernecki"

import numpy as np


class GeneticAlgorithm(object):
    def __init__(self, size, distance, flow, pop_size, crossover_pr, mutation_pr,
                 generations, selection="", tour_size=2, seed=-1, caching=False):
        self.size = size
        self.distance = distance
        self.flow = flow
        self.pop_size = pop_size
        self.crossover_pr = crossover_pr
        self.mutation_pr = mutation_pr
        self.generations = generations
        self.selection = selection
        self.tour_size = tour_size
        self.seed = seed
        self.caching = caching
        if self.caching:
            self.cache = {}
            self.cache_stats = np.zeros(shape=(generations + 1, 2), dtype=int)

    def compute_fitness(self, vector) -> int:
        fitness = 0
        for j in range(self.size):
            for k in range(self.size):
                fitness += self.distance[j, k] * self.flow[vector[j], vector[k]]
        return fitness

    def run(self):
        params = np.zeros(shape=(self.generations, 4), dtype=int)
        pop = self.generate_population()
        if self.caching:
            def evaluate(*args):
                return self.evaluate_cache(args[0], args[1])
        else:
            def evaluate(*args):
                return self.evaluate(args[0])

        if self.selection == "tournament":
            def selection(*args):
                return self.tour(args[0], args[1])
        elif self.selection == "roulette":
            def selection(*args):
                return self.roulette(args[0], args[1])
        else:
            def selection(*args):
                return self.ranking(args[0], args[1])

        pop_fitness = self.evaluate(pop)
        for i in range(self.generations):
            pop = selection(pop, pop_fitness)
            pop = self.crossover(pop)
            pop = self.mutate(pop)
            pop_fitness = evaluate(pop, i)
            params[i] = [i, np.amin(pop_fitness),
                         np.mean(pop_fitness), np.amax(pop_fitness)]
        return params, self.cache_stats if self.caching else 0

    def generate_population(self):
        if self.seed > -1:
            np.random.seed(self.seed)
        pop = np.zeros(shape=(self.pop_size, self.size), dtype=int)
        vector = np.arange(self.size)
        for i in range(self.pop_size):
            np.random.shuffle(vector)
            pop[i] = vector
        return pop

    def evaluate(self, pop):
        pop_fitness = np.zeros(self.pop_size, dtype=int)
        for index, value in enumerate(pop):
            pop_fitness[index] = self.compute_fitness(value)
        return pop_fitness

    def evaluate_cache(self, pop, gen_n):
        pop_fitness = np.zeros(self.pop_size, dtype=int)
        no_calculations = 0
        for index, value in enumerate(pop):
            value_string = np.array_str(value)
            if value_string in self.cache:
                pop_fitness[index] = self.cache.get(value_string)
            else:
                pop_fitness[index] = self.compute_fitness(value)
                no_calculations += 1
                self.cache[value_string] = pop_fitness[index]
        self.cache_stats[gen_n] = (gen_n, no_calculations)
        return pop_fitness

    def crossover(self, pop):
        for i in range(0, self.pop_size, 2):
            if np.random.random() < self.crossover_pr:
                crossover_point = np.random.randint(1, self.size)
                pop[i][:crossover_point], pop[i + 1][:crossover_point] = \
                    pop[i + 1][:crossover_point], pop[i][:crossover_point]
                pop[i] = self.repair(pop[i])
                pop[i + 1] = self.repair(pop[i + 1])
        return pop

    def repair(self, child):
        new_child = np.full(self.size, self.size + 1)
        for index, value in enumerate(child):
            if value not in new_child:
                new_child[index] = value
            else:
                while value in child or value in new_child:
                    value = np.random.randint(self.size)
                new_child[index] = value
        return new_child

    def mutate(self, pop):
        for i in range(self.pop_size):
            if np.random.random() < self.mutation_pr:
                rnd_index_1 = np.random.randint(1, self.size)
                rnd_index_2 = np.random.choice(
                    np.arange(self.size)[np.arange(len(pop[i])) != rnd_index_1])
                pop[i, rnd_index_1], pop[i, rnd_index_2] = \
                    pop[i, rnd_index_2], pop[i, rnd_index_1]
        return pop

    def ranking(self, pop, pop_fitness):
        half_size = self.pop_size // 2
        half_indices = np.argpartition(pop_fitness, -half_size)[:-half_size]
        new_pop = np.concatenate((pop[half_indices], pop[half_indices]))
        return new_pop

    def roulette(self, pop, pop_fitness):
        pop_weights = np.zeros(self.pop_size)
        indices = np.arange(self.pop_size)
        new_pop = np.zeros(shape=(self.pop_size, self.size), dtype=int)
        for i in range(self.pop_size):
            pop_weights[i] = 1 / (pop_fitness[i])
        fitness_sum = np.sum(pop_weights, dtype=float)
        for i in range(self.pop_size):
            pop_weights[i] /= fitness_sum
        for i in range(self.pop_size):
            new_pop[i] = pop[np.random.choice(indices, p=pop_weights)]
        return new_pop

    def tour(self, pop, pop_fitness):
        new_pop = np.zeros(shape=(self.pop_size, self.size), dtype=int)
        for i in range(self.pop_size):
            tour_pop_indices = np.random.randint(self.pop_size, size=self.tour_size)
            best_fitness = np.amin(np.take(pop_fitness, tour_pop_indices))
            best_fitness_idx = np.where(pop_fitness == best_fitness)[0][0]
            new_pop[i] = pop[best_fitness_idx]
        return new_pop

    def random_search(self):
        improvement = False
        best_fitness = 100000
        best_vector = np.zeros(shape=(1, self.size), dtype=int)
        for i in range(10000):
            current_vector = np.arange(self.size)
            np.random.shuffle(current_vector)
            current_fitness = self.compute_fitness(current_vector)
            if current_fitness < best_fitness:
                improvement = True
                best_fitness = current_fitness
                best_vector = current_vector
        return best_vector, best_fitness, improvement

    def greedy_algorithm(self):
        best_vector = np.arange(self.size)
        np.random.shuffle(best_vector)
        best_fitness = self.compute_fitness(best_vector)
        return self.greedy_check(best_vector, best_fitness)

    def greedy_check(self, best_vector, best_fitness):
        improvement = False
        current_vector = best_vector
        for i in range(self.size):
            for j in range(self.size):
                current_vector[i], current_vector[j] = current_vector[j], current_vector[i]
                if self.compute_fitness(current_vector) < best_fitness:
                    best_fitness = self.compute_fitness(current_vector)
                    best_vector = current_vector
                    improvement = True
                current_vector[i], current_vector[j] = current_vector[j], current_vector[i]
        if improvement:
            return self.greedy_check(best_vector, best_fitness)
        else:
            return best_vector, best_fitness

    def random_greedy_algorithm(self):
        best_vector = np.arange(self.size)
        np.random.shuffle(best_vector)
        best_fitness = self.compute_fitness(best_vector)
        return self.random_greedy_check(best_vector, best_fitness)

    def random_greedy_check(self, best_vector, best_fitness):
        improvement = False
        current_vector = best_vector
        i = 0  # temporary
        while not improvement and i < self.size:
            j = 0  # temporary
            while not improvement and j < self.size:
                id_1 = np.random.randint(0, self.size)
                id_2 = np.random.randint(0, self.size)
                current_vector[id_1], current_vector[id_2] = \
                    current_vector[id_2], current_vector[id_1]
                if self.compute_fitness(current_vector) < best_fitness:
                    best_fitness = self.compute_fitness(current_vector)
                    best_vector = current_vector
                    improvement = True
                current_vector[id_1], current_vector[id_2] = \
                    current_vector[id_2], current_vector[id_1]
                j += 1
            i += 1
        if improvement:
            return self.random_greedy_check(best_vector, best_fitness)
        else:
            return best_vector, best_fitness

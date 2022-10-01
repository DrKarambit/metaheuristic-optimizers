import tensorflow as tf
from .metaheuristic_optimizer_base import MetaheuristicOptimizerBase


class Genetic(MetaheuristicOptimizerBase):

    def __init__(self, model, loss_fn, x_train, y_train, batch_size=32, loss_limit=0.1, generation_limit=10, population_size=12, selection_size=4, crossover_ratio=0.2, mutation_chance=0.2, mutation_scale=0.001, name='Genetic', **kwargs):
        super(Genetic, self).__init__(model=model, loss_fn=loss_fn, x_train=x_train, y_train=y_train, batch_size=batch_size, name=name, **kwargs)
        
        self.loss_limit = tf.constant(loss_limit, dtype=tf.float32)
        self.generation_limit = tf.constant(generation_limit, dtype=tf.int32)
        self.population_size = tf.constant(population_size, dtype=tf.int32)
        self.selection_size = tf.constant(selection_size, dtype=tf.int32)
        self.crossover_ratio = tf.constant(crossover_ratio, dtype=tf.float32)
        self.mutation_chance = tf.constant(mutation_chance, dtype=tf.float32)
        self.mutation_scale = tf.constant(mutation_scale, dtype=tf.float32)
        
        self.generation = tf.Variable(0, dtype=tf.int32)
        self.best_fitness = tf.Variable(0, dtype=tf.float32)
        self.fitness_values = tf.Variable(tf.zeros(population_size, dtype=tf.float32))

    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
    def create_population(self, weights):
        return tf.repeat(tf.expand_dims(weights, axis=0), self.population_size, axis=0)

    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
    def selection(self, population):
        return tf.gather(population, tf.argsort(self.fitness_values)[:self.selection_size])

    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32), tf.TensorSpec(shape=None, dtype=tf.float32)])
    def crossover(self, population, selected_chromosomes):
        parent1 = tf.random.uniform(shape=tf.expand_dims(self.population_size, axis=0), minval=0, maxval=self.selection_size, dtype=tf.int32)
        parent2 = tf.math.mod(tf.math.add(parent1, 1), self.selection_size)
        crossover_points = tf.random.uniform(shape=tf.shape(population), minval=0, maxval=1, dtype=tf.float32) < self.crossover_ratio
        return tf.where(crossover_points, tf.gather(selected_chromosomes, parent1), tf.gather(selected_chromosomes, parent2))

    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
    def mutation(self, population):
        mutation_values = tf.random.normal(shape=tf.shape(population), mean=0.0, stddev=self.mutation_scale, dtype=tf.float32)
        mutation_points = tf.cast(tf.random.uniform(shape=tf.shape(population), minval=0, maxval=1, dtype=tf.float32) < self.mutation_chance, dtype=tf.float32)
        return tf.math.add(population, tf.math.multiply(mutation_values, mutation_points))

    @tf.function
    def update_fitness(self, weights, population):
        for i in range(self.population_size):
            weights.assign(population[i])
            self.fitness_values[i].assign(self.fitness())
        self.best_fitness.assign(tf.reduce_min(self.fitness_values))

    @tf.function
    def metaheuristic(self, weights):
        population = self.create_population(weights)
        self.update_fitness(weights, population)
        self.generation.assign(0)
        while self.generation < self.generation_limit and self.best_fitness > self.loss_limit:
            selected_chromosomes = self.selection(population)
            population = self.crossover(population, selected_chromosomes)
            population = self.mutation(population)
            self.update_fitness(weights, population)
            self.generation.assign_add(1)
        weights.assign(population[tf.argmin(self.fitness_values)])

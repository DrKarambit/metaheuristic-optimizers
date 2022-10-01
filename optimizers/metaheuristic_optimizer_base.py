import tensorflow as tf


class MetaheuristicOptimizerBase(tf.keras.optimizers.Optimizer):

    _HAS_AGGREGATE_GRAD = True

    def __init__(self, model, loss_fn, x_train, y_train, batch_size, name='MetaheuristicOptimizerBase', **kwargs):
        super(MetaheuristicOptimizerBase, self).__init__(name=name, **kwargs)

        self._model = model
        self._loss_fn = loss_fn.from_config(loss_fn.get_config())
        self._x_train = tf.constant(x_train)
        self._y_train = tf.constant(y_train)

        self._batch_size = tf.constant(batch_size, dtype=tf.int32)
        self._batch_count = tf.constant(len(y_train) // batch_size, dtype=tf.int32)
        self._batch_iteration = tf.Variable(0, dtype=tf.int32)
        self._batch_slice_start = tf.Variable(0, dtype=tf.int32)
        self._batch_slice_end = tf.Variable(batch_size, dtype=tf.int32)
        
        self._handle_count = tf.constant(len(model.trainable_weights), dtype=tf.int32)
        self._handle_iteration = tf.Variable(0, dtype=tf.int32)

    @tf.function
    def _resource_apply_dense(self, grad, handle, apply_state=None):
        self.metaheuristic(handle)
        self._handle_iteration.assign_add(1)
        if tf.math.equal(self._handle_iteration, self._handle_count):
            self._batch_handler()
            self._handle_iteration.assign(0)

    @tf.function
    def _resource_apply_sparse(self, grad, handle, indices, apply_state=None):
        raise NotImplementedError

    def _create_slots(self, var_list):
        pass

    def get_config(self):
        return super(MetaheuristicOptimizerBase, self).get_config()

    @tf.function
    def _batch_handler(self):
        self._batch_iteration.assign(tf.math.mod(tf.math.add(self._batch_iteration, 1), self._batch_count))
        self._batch_slice_start.assign(tf.math.multiply(self._batch_iteration, self._batch_size))
        self._batch_slice_end.assign(tf.math.add(self._batch_slice_start, self._batch_size))

    @tf.function
    def fitness(self):
        return self._loss_fn(
            self._y_train[self._batch_slice_start:self._batch_slice_end],
            self._model(self._x_train[self._batch_slice_start:self._batch_slice_end]))

    @tf.function
    def metaheuristic(self, weights):
        raise NotImplementedError

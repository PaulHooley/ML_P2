import numpy as np


class MomentumGradientDescent:
    def __init__(
        self,
        learning_rate=0.001,
        momentum=0.9,
        max_iters=1e4,
        epsilon=1e-8,
        batch_size=32,
        record_history=False,
    ):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.max_iters = max_iters
        self.record_history = record_history
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.prev_delta_w = None
        if record_history:
            # to store the weight history for visualization
            self.w_history = []
            
    def run(self, gradient_fn, x, y, w):
        grad = np.inf
        t = 1
        N, D = x.shape
        self.prev_delta_w = np.zeros(w.shape)
        while np.linalg.norm(grad) > self.epsilon and t < self.max_iters:
            grad = gradient_fn(x, y, w)
            delta_w = self.get_delta_w(grad)

            # weight update step
            w = w - self.learning_rate * delta_w
            if self.record_history:
                self.w_history.append(w)
            t += 1
        return w

#     def run(self, gradient_fn, x, y, w):
#         grad = np.inf
#         t = 1
#         N, D = x.shape
#         self.prev_delta_w = np.zeros(D)
#         while np.linalg.norm(grad) > self.epsilon and t < self.max_iters:
#             for i in range(0, N, self.batch_size):
#                 if x.ndim == 1:
#                     batch_x = x[i : i + self.batch_size]
#                 else:
#                     batch_x = x[i : i + self.batch_size, :]

#                 if y.ndim == 1:
#                     batch_y = y[i : i + self.batch_size]
#                 else:
#                     batch_y = y[i : i + self.batch_size, :]

#                 # compute the gradient with present weight
#                 grad = gradient_fn(batch_x, batch_y, w)
#                 delta_w = self.get_delta_w(grad)

#                 # weight update step
#                 w = w - self.learning_rate * delta_w
#                 if self.record_history:
#                     self.w_history.append(w)
#             t += 1
#         return w

    def get_delta_w(self, grad):
        beta = self.momentum
        delta_w = beta * self.prev_delta_w + (1 - beta) * grad
        self.prev_delta_w = delta_w

        return delta_w

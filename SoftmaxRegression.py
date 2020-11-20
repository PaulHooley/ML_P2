import numpy as np

class SoftmaxRegression:
    def __init__(self, add_bias=True):
        self.add_bias = add_bias

    def fit(self, x, y, optimizer):
        if x.ndim == 1:
            x = x[:, np.newaxis]
        if self.add_bias:
            N = x.shape[0]
            x = np.column_stack( (x, np.ones(N)) )
        N, D = x.shape
        C = np.max(y) + 1

        def gradient(x, y, w):
            N, D = x.shape
            C = w.shape[1]

            targets = y.reshape(-1)
            yh = np.eye(C)[targets]

            grad = np.zeros((D, C))
            for c in range(C):
                for d in range(D):
                    for n in range(N):
                        t = np.exp(np.dot(x[n,:], w[:,c]))
                        t /= np.sum(np.exp(np.dot(x, w[:,c])))
                        t -= yh[n][c]
                        t *= x[n][d]
                        grad[d][c] += t
                    grad[d][c] /= N
            return grad

        w0 = np.zeros((D, C))
        self.w = optimizer.run(gradient, x, y, w0)

    def predict(self, x):
        if self.add_bias:
            N = x.shape[0]
            x = np.column_stack( (x, np.ones(N)) )

            def softmax(z, c):
                pass
                # todo

            # print(f'{x=}')
            # print(f'{self.w=}')
        return softmax(np.dot(x, self.w))


if __name__ == '__main__':
    from sklearn.datasets import load_digits
    from sklearn.datasets import load_wine
    from GradientDescent import MomentumGradientDescent

    x, y = load_wine(return_X_y=True)
    # print(f'{x.shape=}', f'{y.shape=}', sep='\t')
    # print(y)

    optimizer = MomentumGradientDescent(record_history=True)
    model = SoftmaxRegression()
    model.fit(x, y, optimizer)
    y_predict = model.predict(x)
    # print(y_predict)

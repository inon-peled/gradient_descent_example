import random


def example1():
    # From calculation, it is expected that the local minimum occurs at x=9/4

    x_old = 0  # The value does not matter as long as abs(x_new - x_old) > precision
    x_new = 6  # The algorithm starts at x=6
    gamma = 0.01  # step size
    precision = 1E-16

    def df(x):
        y = 4 * x ** 3 - 9 * x ** 2
        return y

    while abs(x_new - x_old) > precision:
        x_old = x_new
        x_new += -gamma * df(x_old)

    print("The local minimum occurs at %f" % x_new)


def example2(x_min, y_min):
    def f(x, y):
        return 3 * (x - x_min) ** 2 + (y - y_min) ** 2

    def gradient_f(x, y):
        return 6 * (x - x_min), 2 * (y - y_min)

    def random_float():
        return random.randint(0, 100) * random.random()

    x_old, y_old, x_new, y_new = [random_float() for i in range(4)]
    # x_old, y_old, x_new, y_new = [x_min + random.random(), y_min + random.random(), x_min + random.random(), y_min + random.random()]
    gamma = 0.0001  # step size
    precision = 1E-16

    while ((x_new - x_old) ** 2 + (y_new - y_old) ** 2) ** 0.5 > precision:
        x_old, y_old = x_new, y_new
        grad = gradient_f(x_old, y_old)
        x_new -= gamma * grad[0]
        y_new -= gamma * grad[1]

    print("f_local_min = %f = f(%f, %f)" % (f(x_new, y_new), x_new, y_new))


if __name__ == '__main__':
    example2(7, 8)

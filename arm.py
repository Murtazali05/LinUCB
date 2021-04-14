import numpy as np


class linucb_disjoint_arm():

    def __init__(self, arm_index, d, alpha):
        # Инциализируем индекс ручки
        self.arm_index = arm_index

        # Значение alpha, которое контролирует exploration
        self.alpha = alpha

        # A: (d x d) матрица = D_a.T * D_a + I_d.
        # Обратная матрица A используется в ridge regression
        self.A = np.identity(d)

        # b: (d x 1) соответствующий вектор ответов.
        # Равно значению D_a.T * c_a в формулировке ridge regression
        self.b = np.zeros([d, 1])

    def calc_UCB(self, x_array):
        # Находим обратную матрицу к A для ridge regression
        A_inv = np.linalg.inv(self.A)

        # Выполняем ridge regression, чтобы получить оценку коэфициентов theta
        # theta это вектор размерности (d x 1)
        self.theta = np.dot(A_inv, self.b)

        # Меняем форму входного вектора в вектор размерности (d x 1)
        x = x_array.reshape([-1, 1])

        # Находим ucb основываясь на формулировке (mean + std_dev)
        # p это вектор размерности (1 x 1)
        p = np.dot(self.theta.T, x) + self.alpha * np.sqrt(np.dot(x.T, np.dot(A_inv, x)))

        return p

    def reward_update(self, reward, x_array):
        # Входной вектор в вектор размерности (d x 1)
        x = x_array.reshape([-1, 1])

        # Обновляем значения матрицы A размерности (d * d).
        self.A += np.dot(x, x.T)

        # Обновляем вектор b размерности (d x 1)
        # reward скалярное значение
        self.b += reward * x

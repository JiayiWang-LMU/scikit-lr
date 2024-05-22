import numpy as np
import scipy


class RankAggregation:
    def __init__(self, pair_order_matrix):
        """
        Initialize the RankAggregator with a pair order matrix.

        Parameters:
        pair_order_matrix (numpy.ndarray): A square matrix representing the pair order matrix.
        """
        if pair_order_matrix.shape[0] != pair_order_matrix.shape[1]:
            raise ValueError("Wrong shape of given matrix.")

        self.matrix = pair_order_matrix
        self.size = self.matrix.shape[0]

    def orders(self, values):
        """
        Convert ranking values to orders of ranking.

        Parameters:
        values: A list of ranking values.

        Returns:
        list: The corresponding orders of ranking.
        """
        sorted_list = sorted(((value, index) for index, value in enumerate(values)), reverse=True)
        result_list = [0] * len(values)

        current_rank = 1
        current_rank_value = sorted_list[0][0]
        for value, index in sorted_list:
            if value < current_rank_value:
                current_rank += 1
                current_rank_value = value

            result_list[index] = current_rank

        return result_list

    def mc4(self):
        """
        Perform rank aggregation using the MC4 method.

        Returns:
        list: The final ranking values.
        """
        matrix = self.matrix.copy()

        for i in range(self.size):
            for j in range(self.size):
                if i == j:
                    matrix[i][j] = 0
                elif matrix[i][j] > 0.5:
                    matrix[i][j] = 0
                else:
                    matrix[i][j] = 1 / self.size

        for i in range(self.size):
            matrix[i][i] = 1 - np.sum(matrix[i])

        transition_matrix_transp = matrix.T
        eigenvals, eigenvects = np.linalg.eig(transition_matrix_transp)

        close_to_1_idx = np.isclose(eigenvals, 1)
        target_eigenvect = eigenvects[:, close_to_1_idx]
        target_eigenvect = target_eigenvect[:, 0]

        stationary_distrib = target_eigenvect.real / np.sum(target_eigenvect.real)

        rankings = [np.round(value, 10) for value in stationary_distrib]

        return self.orders(rankings)

    def ml(self):
        """
        Perform rank aggregation using the Maximal Lotteries method.

        Returns:
        list: The final ranking values.
        """
        matrix = self.matrix.copy()
        size = self.size

        c = np.random.rand(size)
        A_ub = matrix - matrix.T
        b_ub = np.zeros(size)
        A_eq = [[1] * size]
        b_eq = [1]

        x = scipy.optimize.linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0, 1))
        rounded_x = np.round(x.x, decimals=6)

        return self.orders(rounded_x)

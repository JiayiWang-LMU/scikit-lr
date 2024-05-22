import numpy as np

class Matrix:
    def builder(self, predictions):
        total_size = 1
        class_size = predictions[0].size
        combination_size = class_size * (class_size - 1) // 2
        scores = [0 for _ in range(combination_size)]
        predictions = predictions.reshape((1, 1, -1))
        z = 0
        temp_list = predictions[0, 0]

        for j in range(class_size):
            for q in range(j + 1, class_size):
                element_j = temp_list[j]
                element_q = temp_list[q]
                if element_j == element_q:
                    scores[z] += 0.5
                elif element_j < element_q:
                    scores[z] += 1
                z += 1
        z += 1

        pair_order = [x / total_size for x in scores]

        matrix = np.eye(class_size) * 0.5
        t = 0
        for i in range(class_size):
            for j in range(i + 1, class_size):
                value = pair_order[t]
                matrix[i, j] = value
                matrix[j, i] = 1 - value
                t += 1
        t += 1
        # Return the pair order matrix
        return matrix

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from sklr.ensemble import RandomForestPartialLabelRanker
from sklr.metrics import tau_x_score
from sklr.ra import RankAggregation

if __name__ == '__main__':
    RANDOM_STATE = 42

    # Load the datasets
    file_path = './datasets/data/partial_label_ranking/ranks/iris.csv'
    dataset_info = []

    df = pd.read_csv(file_path)

    # Extract columns with names like A1, A2, ..., An for features
    feature_columns = [col for col in df.columns if col.startswith('A')]
    features = df[feature_columns].values

    # Extract columns with names like L1, L2, ..., Ln for labels
    label_columns = [col for col in df.columns if col.startswith('L')]
    labels = df[label_columns].values

    # Store features and labels for this file
    dataset_info.append({'filename': os.path.basename(file_path), 'features': features, 'labels': labels})

    mc4_results = {}
    std_results = {}

    for data in dataset_info:
        filename = data['filename']
        mc4_key = f"mc4_{filename}"
        print("Processing", filename)

        # Initialize variables to store results
        mc4_means = []
        std_devs = []

        # Specify the number of folds and repetitions
        n_splits = 10
        iterations = 5

        # Initialize lists to store results
        all_tau_mc4 = []

        # Perform 5 x 10 cross-validation
        for iteration in range(iterations):
            print("Iteration ", iteration + 1)
            iteration_tau_mc4 = []

            # Shuffle data indices before splitting
            shuffled_indices = np.random.permutation(len(data["features"]))

            # Initialize KFold cross-validator
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

            for train_index, test_index in kf.split(shuffled_indices):
                X_train, X_test = data["features"][train_index], data["features"][test_index]
                y_train, y_test = data["labels"][train_index], data["labels"][test_index]

                # Create a new model instance for each iteration
                model = RandomForestPartialLabelRanker(random_state=RANDOM_STATE)

                # Train the model
                clf = model.fit(X_train, y_train)

                # Create lists to store ranking predictions from different aggregation methods
                mc4_predictions = []

                # Predict rankings with different aggregation methods
                for instance in X_test:
                    instance = instance.reshape(1, -1)
                    # Compute the pair order matrix
                    pair_order_matrix = clf.predict(instance)
                    # Initialize the aggregator
                    rank_aggregator = RankAggregation(pair_order_matrix)
                    mc4_predictions.append(rank_aggregator.mc4())

                # Compute performance metric
                tau_mc4 = tau_x_score(np.array(mc4_predictions), y_test)
                iteration_tau_mc4.append(tau_mc4)

            # Store results for this iteration
            all_tau_mc4.append(iteration_tau_mc4)

        # Calculate the mean and standard deviation for mc4 rankings
        mc4_mean = np.mean(all_tau_mc4)
        mc4_std = np.std(all_tau_mc4)

        # Append the results
        mc4_means.append(mc4_mean)
        std_devs.append(mc4_std)

        mc4_results[mc4_key] = mc4_means
        std_results[mc4_key] = std_devs

    print(mc4_results)

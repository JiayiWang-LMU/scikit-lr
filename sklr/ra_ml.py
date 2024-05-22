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
    folder_path = './datasets/data/partial_label_ranking/ranks'
    dataset_info = []

    # Iterate through each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            csv_file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(csv_file_path)

            # Extract columns with names like A1, A2, ..., An for features
            feature_columns = [col for col in df.columns if col.startswith('A')]
            features = df[feature_columns].values

            # Extract columns with names like L1, L2, ..., Ln for labels
            label_columns = [col for col in df.columns if col.startswith('L')]
            labels = df[label_columns].values

            # Store features and labels for this file
            dataset_info.append({'filename': filename, 'features': features, 'labels': labels})

    ml_results = {}
    std_results = {}

    for data in dataset_info:
        filename = data['filename']
        ml_key = f"ml_{filename}"

        # Initialize variables to store results
        ml_means = []
        std_devs = []

        # Specify the number of folds and repetitions
        n_splits = 10
        iterations = 1

        # Initialize lists to store results
        all_tau_ml = []

        # Perform cross-validation
        for iteration in range(iterations):
            iteration_tau_ml = []
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

                # Create lists to store ranking predictions
                ml_predictions = []

                for instance in X_test:
                    instance = instance.reshape(1, -1)
                    # Compute the pair order matrix
                    pair_order_matrix = clf.predict(instance)
                    # Initialize the aggregator
                    rank_aggregator = RankAggregation(pair_order_matrix)
                    ml_predictions.append(rank_aggregator.ml())

                # Compute performance metric
                tau_ml = tau_x_score(np.array(ml_predictions), y_test)
                iteration_tau_ml.append(tau_ml)

            # Store results for this iteration
            all_tau_ml.append(iteration_tau_ml)

        # Calculate the mean and standard deviation for ml rankings
        ml_mean = np.mean(all_tau_ml)
        ml_std = np.std(all_tau_ml)

        # Append the results
        ml_means.append(ml_mean)
        std_devs.append(ml_std)

        ml_results[ml_key] = ml_means
        std_results[ml_key] = std_devs

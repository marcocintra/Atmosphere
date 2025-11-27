import numpy as np

from sklearn.metrics import (mean_absolute_error,
                             root_mean_squared_error)


def taylor_skill_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    correlation = np.corrcoef(y_true, y_pred)[0, 1]

    std_true = np.std(y_true, ddof=1)
    std_pred = np.std(y_pred, ddof=1)

    if std_true == 0 or std_pred == 0:
        return 0.0

    std_ratio = std_pred / std_true

    denominator = ((std_ratio + 1/std_ratio)**2) * (1 + 1.0)
    skill_score = (4 * (1 + correlation)) / denominator

    return skill_score


def kling_gupta_efficiency(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    r = np.corrcoef(y_true, y_pred)[0, 1]

    std_true = np.std(y_true, ddof=1)
    std_pred = np.std(y_pred, ddof=1)
    alpha = std_pred / std_true if std_true != 0 else np.inf

    mean_obs = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    beta = mean_pred / mean_obs if mean_obs != 0 else np.inf

    kge = 1 - np.sqrt(((r - 1)**2) + ((alpha - 1)**2) + ((beta - 1)**2))

    components = {'r': r, 'alpha': alpha, 'beta': beta}

    return kge, components


class GopiTecMapErrorEstimation:

    def __init__(self):
        self.datasets = []

    def add_dataset(self,
                    dataset_name,
                    ground_truth,
                    estimated):

        self.datasets.append({
            'name': dataset_name,
            'ground_truth': ground_truth,
            'estimated': estimated
        })

    def run(self):
        results = []

        for dataset in self.datasets:
            mask = np.isnan(dataset['ground_truth'])
            if np.sum(mask) > 0:
                dataset['ground_truth'][mask] = 0

            mask = np.isnan(dataset['estimated'])
            if np.sum(mask) > 0:
                dataset['estimated'][mask] = 0

            mae = mean_absolute_error(dataset['ground_truth'], dataset['estimated'])
            rmse = root_mean_squared_error(dataset['ground_truth'], dataset['estimated'])
            corr = np.corrcoef(dataset['ground_truth'], dataset['estimated'])[0, 1]
            tss = taylor_skill_score(dataset['ground_truth'], dataset['estimated'])
            kge, components = kling_gupta_efficiency(dataset['ground_truth'], dataset['estimated'])

            results.append({
                'name': dataset['name'],
                'MAE': mae,
                'RMSE': rmse,
                'CORR': corr,
                'TSS': tss,
                'KGE': kge
            })

        return results

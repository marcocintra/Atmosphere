import pandas as pd
import numpy as np

from pathlib import Path

from src.utils import create_folder


if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parent
    print(f"Project dir: {project_dir}")

    input_dir = project_dir / 'output' / 'errors'
    df = pd.read_csv(input_dir / 'test_case_06-error_estimation_by_network_and_station.csv')

    networks = df['network'].unique()

    results = []
    for network in networks:
        network_df = df[df['network'] == network].copy()
        results.append({
            'network': network,
            'MAE': np.mean(network_df['MAE']),
            'STD_MAE': np.std(network_df['MAE']),
            'RMSE': np.mean(network_df['RMSE']),
            'STD_RMSE': np.std(network_df['RMSE']),
            'CORR': np.mean(network_df['CORR']),
            'STD_CORR': np.std(network_df['CORR']),
            'TSS': np.mean(network_df['TSS']),
            'STD_TSS': np.std(network_df['TSS']),
            'KGE': np.mean(network_df['KGE']),
            'STD_KGE': np.std(network_df['KGE']),
        })
    print(results)

    output_dir = project_dir / 'output' / 'errors'
    create_folder(output_dir, clear=False)

    result_df = pd.DataFrame(results)
    result_df.to_csv(output_dir / 'test_case_06-error_estimation_by_network_and_station_stats_distribution.csv',
                     index=False)

# import numpy as np
# import pandas as pd

# from pathlib import Path

# if __name__ == '__main__':
#     datasets = {
#         'embrace': [
#             'mapas1_embrace_2022_2024_0800',
#             'mapas1_embrace_2022_2024_1600',
#             'mapas1_embrace_2022_2024_2000_2200_0000_0200_0400'
#         ],
#         'igs': [
#             'mapas1_igs_2022_2024_0800',
#             'mapas1_igs_2022_2024_1600',
#             'mapas1_igs_2022_2024_2000_2200_0000_0200_0400',
#             'mapas2_igs_2022_2024_0800',
#             'mapas2_igs_2022_2024_1600',
#             'mapas2_igs_2022_2024_2000_2200_0000_0200_0400'
#         ],
#         'maggia': [
#             'mapas1_maggia_2022_2024_0800',
#             'mapas1_maggia_2022_2024_1600',
#             'mapas1_maggia_2022_2024_2000_2200_0000_0200_0400',
#             'mapas2_maggia_2022_2024_0800',
#             'mapas2_maggia_2022_2024_1600',
#             'mapas2_maggia_2022_2024_2000_2200_0000_0200_0400'
#         ],
#         'nagoya': [
#             'mapas1_nagoya_2022_2024_0800',
#             'mapas1_nagoya_2022_2024_1600',
#             'mapas1_nagoya_2022_2024_2000_2200_0000_0200_0400',
#             'mapas2_nagoya_2022_2024_0800',
#             'mapas2_nagoya_2022_2024_1600',
#             'mapas2_nagoya_2022_2024_2000_2200_0000_0200_0400'
#         ]

#     }

#     comparisons = [
#         ['embrace', 'igs'],
#         ['embrace', 'maggia'],
#         ['embrace', 'nagoya'],
#         ['igs', 'maggia'],
#         ['igs', 'nagoya'],
#         ['maggia', 'nagoya'],
#     ]

#     base_dir = Path('.').resolve() / 'output'
#     dataset_type = 'interp'

#     result = []
#     for comparison in comparisons:
#         for i, dataset_a in enumerate(datasets[comparison[0]]):
#             dataset_b = datasets[comparison[1]][i]
#             print(dataset_a, 'x', dataset_b)

#             for file_a in sorted((base_dir / f'{dataset_a}_{dataset_type}').glob('*.npy')):
#                 epoch = np.datetime64(file_a.name[:-4].replace('.', ':'))
#                 file_b = base_dir / f'{dataset_b}_{dataset_type}' / file_a.name

#                 if not file_b.exists():
#                     continue

#                 map_a = np.load(file_a).flatten()
#                 map_a = np.nan_to_num(map_a)

#                 map_b = np.load(file_b).flatten()
#                 map_b = np.nan_to_num(map_b)

#                 corr = np.corrcoef(map_a, map_b)[0, 1]

#                 result.append({
#                     'datetime': epoch,
#                     'comparison': f'{comparison[0]} x {comparison[1]}',
#                     'dataset_a': dataset_a,
#                     'dataset_b': dataset_b,
#                     'map_a': map_a,
#                     'map_b': map_b,
#                     'corr': corr,
#                     'corr_p': corr * 100
#                 })

#     df = pd.DataFrame(result)
#     df.to_csv('result_correlation.csv', index=False)
#     df.sort_values('datetime', inplace=True)

#     for comparison in comparisons:
#         for i, dataset_a in enumerate(datasets[comparison[0]]):
#             print(comparison[0].upper(), 'x', comparison[1].upper())

#             dataset_b = datasets[comparison[1]][i]
#             print(dataset_a, 'x', dataset_b)

#             comparison_type = f'{comparison[0]} x {comparison[1]}'
#             selection = df.loc[(df['comparison'] == comparison_type)]
#             selection = selection.loc[selection['dataset_a'] == dataset_a]
#             print(len(selection))

#             map_a_concat = np.concatenate(selection['map_a'].values)
#             map_b_concat = np.concatenate(selection['map_b'].values)
#             total = np.corrcoef(map_a_concat, map_b_concat)[0, 1]


#             print(f'Total correlation: {total * 100:.2f}%')

#             for year in [2022, 2023, 2024]:
#                 mar = selection.loc[(selection['datetime'].dt.month == 3) &
#                                     (selection['datetime'].dt.year == year)]
#                 jun = selection.loc[(selection['datetime'].dt.month == 6) &
#                                     (selection['datetime'].dt.year == year)]
#                 sep = selection.loc[(selection['datetime'].dt.month == 9) &
#                                     (selection['datetime'].dt.year == year)]
#                 dec = selection.loc[(selection['datetime'].dt.month == 12) &
#                                     (selection['datetime'].dt.year == year)]
#                 if not mar.empty:
#                     map_a_concat = np.concatenate(mar['map_a'].values)
#                     map_b_concat = np.concatenate(mar['map_b'].values)
#                     mar_total = np.corrcoef(map_a_concat, map_b_concat)[0, 1]

#                     print(f'Correlation (March/{year}): {mar_total*100:.2f}%')
#                 if not jun.empty:
#                     map_a_concat = np.concatenate(jun['map_a'].values)
#                     map_b_concat = np.concatenate(jun['map_b'].values)
#                     jun_total = np.corrcoef(map_a_concat, map_b_concat)[0, 1]

#                     print(f'Correlation (June/{year}): {jun_total*100:.2f}%')
#                 if not sep.empty:
#                     map_a_concat = np.concatenate(sep['map_a'].values)
#                     map_b_concat = np.concatenate(sep['map_b'].values)
#                     sep_total = np.corrcoef(map_a_concat, map_b_concat)[0, 1]

#                     print(f'Correlation (September/{year}): {sep_total*100:.2f}%')
#                 if not dec.empty:
#                     map_a_concat = np.concatenate(dec['map_a'].values)
#                     map_b_concat = np.concatenate(dec['map_b'].values)
#                     dec_total = np.corrcoef(map_a_concat, map_b_concat)[0, 1]

#                     print(f'Correlation (December/{year}): {dec_total*100:.2f}%')

#             print()

import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Função para calcular o RMSE
def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# Função para calcular o RMSE como porcentagem da amplitude
def calculate_rmse_percent(y_true, y_pred):
    rmse = calculate_rmse(y_true, y_pred)
    # Combinamos ambos os arrays para encontrar a amplitude total
    combined = np.concatenate([y_true, y_pred])
    data_range = np.max(combined) - np.min(combined)
    # Evita divisão por zero
    if data_range == 0:
        return 0
    return (rmse / data_range) * 100

# Funções para a transformação Z de Fisher
def fisher_z_transform(r):
    """Transforma correlação r para valor z"""
    # Limita r para evitar problemas com valores extremos (-1 ou 1)
    if r >= 1.0:
        r = 0.9999
    elif r <= -1.0:
        r = -0.9999
    return 0.5 * np.log((1 + r) / (1 - r))

def fisher_z_inverse(z):
    """Transforma z de volta para correlação r"""
    return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)

# Funções para as novas métricas
def calculate_residual_error(y_true, y_pred):
    """Calcula o erro residual médio"""
    return np.mean(y_true - y_pred)

def calculate_max_residual_error(y_true, y_pred):
    """Calcula o erro residual máximo (em valor absoluto)"""
    return np.max(np.abs(y_true - y_pred))

def calculate_min_residual_error(y_true, y_pred):
    """Calcula o erro residual no 5º percentil (em valor absoluto)"""
    return np.percentile(np.abs(y_true - y_pred), 5)

def calculate_r2_score(y_true, y_pred):
    """Calcula o coeficiente de determinação (R²)"""
    return r2_score(y_true, y_pred)

def calculate_mse(y_true, y_pred):
    """Calcula o Mean Squared Error"""
    return mean_squared_error(y_true, y_pred)

def calculate_mae(y_true, y_pred):
    """Calcula o Mean Absolute Error"""
    return mean_absolute_error(y_true, y_pred)

def calculate_cosine_similarity(y_true, y_pred):
    """Calcula a similaridade de cosseno entre dois vetores"""
    dot_product = np.dot(y_true, y_pred)
    norm_y_true = np.linalg.norm(y_true)
    norm_y_pred = np.linalg.norm(y_pred)
    # Evita divisão por zero
    if norm_y_true == 0 or norm_y_pred == 0:
        return 0
    return dot_product / (norm_y_true * norm_y_pred)

def calculate_huber_loss(y_true, y_pred, delta=1.0):
    """Calcula a perda de Huber com delta padrão = 1.0"""
    errors = y_true - y_pred
    abs_errors = np.abs(errors)
    quadratic = np.minimum(abs_errors, delta)
    linear = abs_errors - quadratic
    return np.mean(0.5 * quadratic * quadratic + delta * linear)

if __name__ == '__main__':
    # Configuração do parser de argumentos
    parser = argparse.ArgumentParser(description='Calculate metrics between datasets')
    parser.add_argument('--metric', type=str, 
                        choices=['pearson', 'rmse', 'residual', 'max_residual', 'min_residual', 
                                'r2', 'mse', 'mae', 'cosine', 'huber'], 
                        default='pearson',
                        help='Metric to calculate: pearson, rmse, residual, max_residual, min_residual, r2, mse, mae, cosine, or huber')
    parser.add_argument('--huber-delta', type=float, default=1.0,
                        help='Delta parameter for Huber loss (default: 1.0)')
    parser.add_argument('--min-residual-percentile', type=float, default=5.0,
                        help='Percentile for min_residual calculation (default: 5.0)')
    args = parser.parse_args()
    
    metric_type = args.metric
    huber_delta = args.huber_delta
    min_residual_percentile = args.min_residual_percentile
    
    # Atualizar a função min_residual para usar o percentil especificado
    if metric_type == 'min_residual':
        def custom_min_residual(y_true, y_pred):
            return np.percentile(np.abs(y_true - y_pred), min_residual_percentile)
        calculate_min_residual_error = custom_min_residual
    
    # Dicionário de funções para calcular métricas
    metric_functions = {
        'pearson': lambda y_true, y_pred: np.corrcoef(y_true, y_pred)[0, 1],
        'rmse': calculate_rmse,
        'residual': calculate_residual_error,
        'max_residual': calculate_max_residual_error,
        'min_residual': calculate_min_residual_error,
        'r2': calculate_r2_score,
        'mse': calculate_mse,
        'mae': calculate_mae,
        'cosine': calculate_cosine_similarity,
        'huber': lambda y_true, y_pred: calculate_huber_loss(y_true, y_pred, huber_delta)
    }
    
    # Determinar se a métrica é do tipo "quanto maior, melhor" ou "quanto menor, melhor"
    higher_is_better = metric_type in ['pearson', 'r2', 'cosine']
    
    # Determinar se a métrica precisa de transformação Fisher para média
    needs_fisher_transform = metric_type in ['pearson', 'r2']
    
    datasets = {
        'embrace': [
            'mapas1_embrace_2022_2024_0800',
            'mapas1_embrace_2022_2024_1600',
            'mapas1_embrace_2022_2024_2000_2200_0000_0200_0400'
        ],
        'igs': [
            'mapas1_igs_2022_2024_0800',
            'mapas1_igs_2022_2024_1600',
            'mapas1_igs_2022_2024_2000_2200_0000_0200_0400',
            'mapas2_igs_2022_2024_0800',
            'mapas2_igs_2022_2024_1600',
            'mapas2_igs_2022_2024_2000_2200_0000_0200_0400'
        ],
        'maggia': [
            'mapas1_maggia_2022_2024_0800',
            'mapas1_maggia_2022_2024_1600',
            'mapas1_maggia_2022_2024_2000_2200_0000_0200_0400',
            'mapas2_maggia_2022_2024_0800',
            'mapas2_maggia_2022_2024_1600',
            'mapas2_maggia_2022_2024_2000_2200_0000_0200_0400'
        ],
        'nagoya': [
            'mapas1_nagoya_2022_2024_0800',
            'mapas1_nagoya_2022_2024_1600',
            'mapas1_nagoya_2022_2024_2000_2200_0000_0200_0400',
            'mapas2_nagoya_2022_2024_0800',
            'mapas2_nagoya_2022_2024_1600',
            'mapas2_nagoya_2022_2024_2000_2200_0000_0200_0400'
        ]
    }

    comparisons = [
        ['embrace', 'igs'],
        ['embrace', 'maggia'],
        ['embrace', 'nagoya'],
        ['igs', 'maggia'],
        ['igs', 'nagoya'],
        ['maggia', 'nagoya'],
    ]

    base_dir = Path('.').resolve() / 'output'
    dataset_type = 'interp'

    print(f"Calculating {metric_type.upper()} metrics with additional statistics...")
    
    if metric_type == 'min_residual':
        print(f"Using {min_residual_percentile}th percentile for minimum residual calculation")

    result = []
    for comparison in comparisons:
        for i, dataset_a in enumerate(datasets[comparison[0]]):
            dataset_b = datasets[comparison[1]][i]
            print(dataset_a, 'x', dataset_b)

            for file_a in sorted((base_dir / f'{dataset_a}_{dataset_type}').glob('*.npy')):
                epoch = np.datetime64(file_a.name[:-4].replace('.', ':'))
                file_b = base_dir / f'{dataset_b}_{dataset_type}' / file_a.name

                if not file_b.exists():
                    continue

                map_a = np.load(file_a).flatten()
                map_a = np.nan_to_num(map_a)

                map_b = np.load(file_b).flatten()
                map_b = np.nan_to_num(map_b)

                # Calculate statistics
                min_a = np.min(map_a)
                max_a = np.max(map_a)
                q3_a = np.percentile(map_a, 75)
                
                min_b = np.min(map_b)
                max_b = np.max(map_b)
                q3_b = np.percentile(map_b, 75)
                
                both_maps = np.concatenate([map_a, map_b])
                min_both = np.min(both_maps)
                max_both = np.max(both_maps)
                q3_both = np.percentile(both_maps, 75)
                data_range = max_both - min_both

                # Calculate the selected metric
                metric_function = metric_functions[metric_type]
                value = metric_function(map_a, map_b)
                
                # Calculate percentage representation for interpretable metrics
                if metric_type == 'pearson':
                    value_p = value * 100  # percentage for correlation
                elif metric_type == 'rmse':
                    # RMSE as percentage of range
                    value_p = (value / data_range) * 100 if data_range != 0 else 0
                elif metric_type == 'r2':
                    value_p = value * 100  # percentage for R²
                elif metric_type == 'cosine':
                    value_p = value * 100  # percentage for cosine similarity
                else:
                    # For other metrics, just use the raw value
                    value_p = value

                result_data = {
                    'datetime': epoch,
                    'comparison': f'{comparison[0]} x {comparison[1]}',
                    'dataset_a': dataset_a,
                    'dataset_b': dataset_b,
                    'source_a': comparison[0],
                    'source_b': comparison[1],
                    'map_a': map_a,
                    'map_b': map_b,
                    metric_type: value,
                    f'{metric_type}_p': value_p,
                    # Additional statistics
                    'min_a': min_a,
                    'max_a': max_a,
                    'q3_a': q3_a,
                    'min_b': min_b,
                    'max_b': max_b,
                    'q3_b': q3_b,
                    'min_both': min_both,
                    'max_both': max_both,
                    'q3_both': q3_both,
                    'data_range': data_range
                }
                
                result.append(result_data)

    df = pd.DataFrame(result)
    df.to_csv(f'result_{metric_type}_with_stats.csv', index=False)
    df.sort_values('datetime', inplace=True)

    # Dictionary to store aggregate metrics for each dataset
    dataset_metrics = defaultdict(list)
    dataset_total_metrics = defaultdict(float)
    dataset_count = defaultdict(int)
    # Para armazenar tamanhos de amostras (para transformação Z de Fisher)
    dataset_sample_sizes = defaultdict(list)

    for comparison in comparisons:
        for i, dataset_a in enumerate(datasets[comparison[0]]):
            print(comparison[0].upper(), 'x', comparison[1].upper())

            dataset_b = datasets[comparison[1]][i]
            print(dataset_a, 'x', dataset_b)

            comparison_type = f'{comparison[0]} x {comparison[1]}'
            selection = df.loc[(df['comparison'] == comparison_type)]
            selection = selection.loc[selection['dataset_a'] == dataset_a]
            print(f"Number of comparisons: {len(selection)}")

            map_a_concat = np.concatenate(selection['map_a'].values)
            map_b_concat = np.concatenate(selection['map_b'].values)
            
            # Calculate aggregated statistics
            min_a_all = np.min(map_a_concat)
            max_a_all = np.max(map_a_concat)
            q3_a_all = np.percentile(map_a_concat, 75)
            
            min_b_all = np.min(map_b_concat)
            max_b_all = np.max(map_b_concat)
            q3_b_all = np.percentile(map_b_concat, 75)
            
            both_maps_all = np.concatenate([map_a_concat, map_b_concat])
            min_both_all = np.min(both_maps_all)
            max_both_all = np.max(both_maps_all)
            q3_both_all = np.percentile(both_maps_all, 75)
            data_range_all = max_both_all - min_both_all
            
            # Print statistics for all data
            print("\nStatistics for All Data:")
            print(f"Dataset A - Min: {min_a_all:.4f}, Max: {max_a_all:.4f}, Q3: {q3_a_all:.4f}")
            print(f"Dataset B - Min: {min_b_all:.4f}, Max: {max_b_all:.4f}, Q3: {q3_b_all:.4f}")
            print(f"Combined  - Min: {min_both_all:.4f}, Max: {max_both_all:.4f}, Q3: {q3_both_all:.4f}")
            print(f"Data Range: {data_range_all:.4f}")
            
            # Calculate total metric for all data
            metric_function = metric_functions[metric_type]
            metric_value = metric_function(map_a_concat, map_b_concat)
            
            # Display different formats based on metric type
            if metric_type == 'pearson':
                print(f'Total correlation: {metric_value * 100:.2f}%')
            elif metric_type == 'rmse':
                rmse_percent = (metric_value / data_range_all) * 100 if data_range_all != 0 else 0
                print(f'Total RMSE: {metric_value:.4f} ({rmse_percent:.2f}% of data range)')
            elif metric_type == 'residual':
                print(f'Average Residual Error: {metric_value:.4f}')
            elif metric_type == 'max_residual':
                print(f'Maximum Residual Error: {metric_value:.4f}')
            elif metric_type == 'min_residual':
                print(f'Minimum Residual Error ({min_residual_percentile}th percentile): {metric_value:.4f}')
            elif metric_type == 'r2':
                print(f'R² Score: {metric_value:.4f} ({metric_value * 100:.2f}%)')
            elif metric_type == 'mse':
                print(f'Mean Squared Error: {metric_value:.4f}')
            elif metric_type == 'mae':
                print(f'Mean Absolute Error: {metric_value:.4f}')
            elif metric_type == 'cosine':
                print(f'Cosine Similarity: {metric_value:.4f} ({metric_value * 100:.2f}%)')
            elif metric_type == 'huber':
                print(f'Huber Loss (delta={huber_delta}): {metric_value:.4f}')
            
            # Store metrics for dataset comparison
            dataset_metrics[comparison[0]].append(metric_value)
            dataset_metrics[comparison[1]].append(metric_value)
            
            # Store sample sizes for Fisher Z transformation if needed
            sample_size = len(map_a_concat)
            dataset_sample_sizes[comparison[0]].append(sample_size)
            dataset_sample_sizes[comparison[1]].append(sample_size)
            
            # Also track total metrics and counts for averaging later
            dataset_total_metrics[comparison[0]] += metric_value
            dataset_total_metrics[comparison[1]] += metric_value
            dataset_count[comparison[0]] += 1
            dataset_count[comparison[1]] += 1

            for year in [2022, 2023, 2024]:
                mar = selection.loc[(selection['datetime'].dt.month == 3) &
                                    (selection['datetime'].dt.year == year)]
                jun = selection.loc[(selection['datetime'].dt.month == 6) &
                                    (selection['datetime'].dt.year == year)]
                sep = selection.loc[(selection['datetime'].dt.month == 9) &
                                    (selection['datetime'].dt.year == year)]
                dec = selection.loc[(selection['datetime'].dt.month == 12) &
                                    (selection['datetime'].dt.year == year)]
                
                # Process each month if data exists
                for month_data, month_name in [(mar, f"March/{year}"), 
                                             (jun, f"June/{year}"), 
                                             (sep, f"September/{year}"), 
                                             (dec, f"December/{year}")]:
                    if not month_data.empty:
                        map_a_concat = np.concatenate(month_data['map_a'].values)
                        map_b_concat = np.concatenate(month_data['map_b'].values)
                        
                        # Calculate monthly statistics
                        min_a_month = np.min(map_a_concat)
                        max_a_month = np.max(map_a_concat)
                        q3_a_month = np.percentile(map_a_concat, 75)
                        
                        min_b_month = np.min(map_b_concat)
                        max_b_month = np.max(map_b_concat)
                        q3_b_month = np.percentile(map_b_concat, 75)
                        
                        both_maps_month = np.concatenate([map_a_concat, map_b_concat])
                        min_both_month = np.min(both_maps_month)
                        max_both_month = np.max(both_maps_month)
                        q3_both_month = np.percentile(both_maps_month, 75)
                        data_range_month = max_both_month - min_both_month
                        
                        # Print monthly statistics
                        print(f"\nStatistics for {month_name}:")
                        print(f"Dataset A - Min: {min_a_month:.4f}, Max: {max_a_month:.4f}, Q3: {q3_a_month:.4f}")
                        print(f"Dataset B - Min: {min_b_month:.4f}, Max: {max_b_month:.4f}, Q3: {q3_b_month:.4f}")
                        print(f"Combined  - Min: {min_both_month:.4f}, Max: {max_both_month:.4f}, Q3: {q3_both_month:.4f}")
                        print(f"Data Range: {data_range_month:.4f}")
                        
                        # Calculate and print metric
                        month_metric = metric_function(map_a_concat, map_b_concat)
                        
                        # Display different formats based on metric type
                        if metric_type == 'pearson':
                            print(f'Correlation ({month_name}): {month_metric * 100:.2f}%')
                        elif metric_type == 'rmse':
                            rmse_percent = (month_metric / data_range_month) * 100 if data_range_month != 0 else 0
                            print(f'RMSE ({month_name}): {month_metric:.4f} ({rmse_percent:.2f}% of data range)')
                        elif metric_type == 'residual':
                            print(f'Average Residual Error ({month_name}): {month_metric:.4f}')
                        elif metric_type == 'max_residual':
                            print(f'Maximum Residual Error ({month_name}): {month_metric:.4f}')
                        elif metric_type == 'min_residual':
                            print(f'Minimum Residual Error ({month_name}, {min_residual_percentile}th percentile): {month_metric:.4f}')
                        elif metric_type == 'r2':
                            print(f'R² Score ({month_name}): {month_metric:.4f} ({month_metric * 100:.2f}%)')
                        elif metric_type == 'mse':
                            print(f'Mean Squared Error ({month_name}): {month_metric:.4f}')
                        elif metric_type == 'mae':
                            print(f'Mean Absolute Error ({month_name}): {month_metric:.4f}')
                        elif metric_type == 'cosine':
                            print(f'Cosine Similarity ({month_name}): {month_metric:.4f} ({month_metric * 100:.2f}%)')
                        elif metric_type == 'huber':
                            print(f'Huber Loss ({month_name}, delta={huber_delta}): {month_metric:.4f}')

            print("\n" + "-"*50 + "\n")

    # Calculate average metrics for each dataset
    dataset_avg_metrics = {}
    
    if needs_fisher_transform:
        # Usar transformação Z de Fisher para métricas que precisam
        dataset_z_values = defaultdict(list)
        
        for dataset, values in dataset_metrics.items():
            for i, val in enumerate(values):
                # Aplicar transformação de Fisher
                try:
                    z_value = fisher_z_transform(val)
                    # Obter o tamanho da amostra correspondente
                    sample_size = dataset_sample_sizes[dataset][i]
                    weight = sample_size - 3  # Peso recomendado para transformação Z
                    # Armazenar valor Z e peso
                    dataset_z_values[dataset].append((z_value, weight))
                except:
                    # Ignora valores problemáticos
                    pass
        
        # Calcular média ponderada dos valores Z e converter de volta
        for dataset in dataset_z_values:
            if dataset_z_values[dataset]:
                # Aplicar média ponderada se houver pesos
                total_weighted_z = sum(z * weight for z, weight in dataset_z_values[dataset])
                total_weight = sum(weight for _, weight in dataset_z_values[dataset])
                
                if total_weight > 0:
                    avg_z = total_weighted_z / total_weight
                    dataset_avg_metrics[dataset] = fisher_z_inverse(avg_z)
                else:
                    # Média simples se não houver pesos válidos
                    avg_z = sum(z for z, _ in dataset_z_values[dataset]) / len(dataset_z_values[dataset])
                    dataset_avg_metrics[dataset] = fisher_z_inverse(avg_z)
            else:
                dataset_avg_metrics[dataset] = 0
                
        print(f"\nUsing Fisher Z transformation for averaging {metric_type} metrics")
    else:
        # Para outras métricas, continua com a média normal
        for dataset in dataset_total_metrics:
            dataset_avg_metrics[dataset] = dataset_total_metrics[dataset] / dataset_count[dataset]
    
    # Find the best dataset based on the metric
    if higher_is_better:
        # Para métricas onde maior é melhor (pearson, r2, cosine)
        best_dataset = max(dataset_avg_metrics.items(), key=lambda x: x[1])
        worst_dataset = min(dataset_avg_metrics.items(), key=lambda x: x[1])
        metric_name = metric_type.upper()
        if metric_type == 'pearson':
            metric_name = 'CORRELATION'
        elif metric_type == 'r2':
            metric_name = 'R² SCORE'
        elif metric_type == 'cosine':
            metric_name = 'COSINE SIMILARITY'
            
        print(f"\n===== DATASET {metric_name} ANALYSIS =====")
        
        if metric_type in ['pearson', 'r2', 'cosine']:
            print(f"Best dataset: {best_dataset[0]} with average {metric_name.lower()} of {best_dataset[1]*100:.2f}%")
            print(f"Worst dataset: {worst_dataset[0]} with average {metric_name.lower()} of {worst_dataset[1]*100:.2f}%")
        else:
            print(f"Best dataset: {best_dataset[0]} with average {metric_name.lower()} of {best_dataset[1]:.4f}")
            print(f"Worst dataset: {worst_dataset[0]} with average {metric_name.lower()} of {worst_dataset[1]:.4f}")
    else:
        # Para métricas onde menor é melhor (rmse, residuals, mse, mae, huber)
        best_dataset = min(dataset_avg_metrics.items(), key=lambda x: x[1])
        worst_dataset = max(dataset_avg_metrics.items(), key=lambda x: x[1])
        metric_name = metric_type.upper()
        if metric_type == 'rmse':
            metric_name = 'RMSE'
        elif metric_type == 'residual':
            metric_name = 'RESIDUAL ERROR'
        elif metric_type == 'max_residual':
            metric_name = 'MAXIMUM RESIDUAL ERROR'
        elif metric_type == 'min_residual':
            metric_name = f'MINIMUM RESIDUAL ERROR ({min_residual_percentile}th PERCENTILE)'
        elif metric_type == 'mse':
            metric_name = 'MEAN SQUARED ERROR'
        elif metric_type == 'mae':
            metric_name = 'MEAN ABSOLUTE ERROR'
        elif metric_type == 'huber':
            metric_name = 'HUBER LOSS'
            
        print(f"\n===== DATASET {metric_name} ANALYSIS =====")
        print(f"Best dataset (lowest error): {best_dataset[0]} with average {metric_name.lower()} of {best_dataset[1]:.4f}")
        print(f"Worst dataset (highest error): {worst_dataset[0]} with average {metric_name.lower()} of {worst_dataset[1]:.4f}")
    
    # Print ranking of all datasets
    print("\nDataset Ranking (from best to worst):")
    if higher_is_better:
        # Sort by metric (highest to lowest) for metrics where higher is better
        sorted_datasets = sorted(dataset_avg_metrics.items(), key=lambda x: x[1], reverse=True)
        for i, (dataset, avg_val) in enumerate(sorted_datasets, 1):
            if metric_type in ['pearson', 'r2', 'cosine']:
                print(f"{i}. {dataset}: {avg_val*100:.2f}%")
            else:
                print(f"{i}. {dataset}: {avg_val:.4f}")
    else:
        # Sort by metric (lowest to highest) for metrics where lower is better
        sorted_datasets = sorted(dataset_avg_metrics.items(), key=lambda x: x[1])
        for i, (dataset, avg_val) in enumerate(sorted_datasets, 1):
            print(f"{i}. {dataset}: {avg_val:.4f}")
    
    # Save overall metrics to a separate file
    overall_metrics = pd.DataFrame({
        'dataset': list(dataset_avg_metrics.keys()),
        f'avg_{metric_type}': list(dataset_avg_metrics.values())
    })
    
    if metric_type in ['pearson', 'r2', 'cosine']:
        overall_metrics[f'avg_{metric_type}_percent'] = overall_metrics[f'avg_{metric_type}'] * 100
    
    overall_metrics.to_csv(f'dataset_ranking_{metric_type}.csv', index=False)
    print(f"\nDataset ranking saved to 'dataset_ranking_{metric_type}.csv'")

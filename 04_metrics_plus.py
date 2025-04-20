import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from skimage.metrics import structural_similarity as ssim
from skimage.io import imread
import warnings

warnings.filterwarnings('ignore')

def calculate_pearson(y_true, y_pred):
    """Calcula a correlação de Pearson, tratando casos especiais."""
    if len(y_true) == 0 or len(y_pred) == 0 or np.all(np.isnan(y_true)) or np.all(np.isnan(y_pred)):
        return np.nan
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if np.sum(valid_mask) < 2:
        return np.nan
    return np.corrcoef(y_true[valid_mask], y_pred[valid_mask])[0, 1]

def calculate_r2_score(y_true, y_pred):
    """Calcula o R², tratando casos especiais."""
    if len(y_true) == 0 or len(y_pred) == 0 or np.all(np.isnan(y_true)) or np.all(np.isnan(y_pred)):
        return np.nan
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if np.sum(valid_mask) < 2:
        return np.nan
    return r2_score(y_true[valid_mask], y_pred[valid_mask])

def calculate_rmse(y_true, y_pred):
    """Calcula o RMSE, tratando casos especiais."""
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if np.sum(valid_mask) < 2:
        return np.nan
    return np.sqrt(mean_squared_error(y_true[valid_mask], y_pred[valid_mask]))

def calculate_mse(y_true, y_pred):
    """Calcula o MSE, tratando casos especiais."""
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if np.sum(valid_mask) < 2:
        return np.nan
    return mean_squared_error(y_true[valid_mask], y_pred[valid_mask])

def calculate_mae(y_true, y_pred):
    """Calcula o MAE, tratando casos especiais."""
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if np.sum(valid_mask) < 2:
        return np.nan
    return mean_absolute_error(y_true[valid_mask], y_pred[valid_mask])

def calculate_residual_error(y_true, y_pred):
    """Calcula o erro residual médio absoluto, tratando casos especiais."""
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if np.sum(valid_mask) < 2:
        return np.nan
    residual = np.mean(np.abs(y_true[valid_mask] - y_pred[valid_mask]))
    if residual < 0:
        print(f"WARNING: Negative residual detected: {residual}")
    return residual

def calculate_max_residual_error(y_true, y_pred):
    """Calcula o erro residual máximo, tratando casos especiais."""
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if np.sum(valid_mask) < 2:
        return np.nan
    return np.max(np.abs(y_true[valid_mask] - y_pred[valid_mask]))

def calculate_min_residual_error(y_true, y_pred, percentile=5.0):
    """Calcula o erro residual no percentil especificado, tratando casos especiais."""
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if np.sum(valid_mask) < 2:
        return np.nan
    return np.percentile(np.abs(y_true[valid_mask] - y_pred[valid_mask]), percentile)

def calculate_cosine_similarity(y_true, y_pred):
    """Calcula a similaridade de cosseno, tratando casos especiais."""
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if np.sum(valid_mask) < 2:
        return np.nan
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    dot_product = np.dot(y_true_valid, y_pred_valid)
    norm_y_true = np.linalg.norm(y_true_valid)
    norm_y_pred = np.linalg.norm(y_pred_valid)
    if norm_y_true == 0 or norm_y_pred == 0:
        return np.nan
    return dot_product / (norm_y_true * norm_y_pred)

def calculate_huber_loss(y_true, y_pred, delta=1.0):
    """Calcula a perda de Huber, tratando casos especiais."""
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if np.sum(valid_mask) < 2:
        return np.nan
    errors = y_true[valid_mask] - y_pred[valid_mask]
    abs_errors = np.abs(errors)
    quadratic = np.minimum(abs_errors, delta)
    linear = abs_errors - quadratic
    return np.mean(0.5 * quadratic * quadratic + delta * linear)

def calculate_ssim(y_true, y_pred):
    """Calcula o SSIM, tratando casos especiais."""
    if len(y_true.shape) > 2 and y_true.shape[2] > 1:
        y_true = np.mean(y_true, axis=2)
    if len(y_pred.shape) > 2 and y_pred.shape[2] > 1:
        y_pred = np.mean(y_pred, axis=2)
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if np.sum(valid_mask) < 2:
        return np.nan
    data_range = max(np.max(y_true[valid_mask]) - np.min(y_true[valid_mask]), 
                     np.max(y_pred[valid_mask]) - np.min(y_pred[valid_mask]))
    if data_range == 0:
        data_range = 1
    try:
        return ssim(y_true, y_pred, data_range=data_range)
    except Exception as e:
        print(f"Erro ao calcular SSIM: {str(e)}")
        if y_true.shape != y_pred.shape:
            min_height = min(y_true.shape[0], y_pred.shape[0])
            min_width = min(y_true.shape[1], y_pred.shape[1])
            y_true_resized = y_true[:min_height, :min_width]
            y_pred_resized = y_pred[:min_height, :min_width]
            return ssim(y_true_resized, y_pred_resized, data_range=data_range)
        return np.nan

def fisher_z_transform(r):
    """Transforma correlação r para valor z, evitando valores extremos."""
    if np.isnan(r):
        return np.nan
    r = np.clip(r, -0.9999, 0.9999)
    return 0.5 * np.log((1 + r) / (1 - r))

def fisher_z_inverse(z):
    """Transforma z de volta para correlação r."""
    if np.isnan(z):
        return np.nan
    return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)

def calculate_strict_stats(map_a, map_b):
    """Calcula estatísticas consistentes para os mapas, sem Q1 e Q3."""
    map_a_flat = map_a.flatten() if len(map_a.shape) > 1 else map_a
    map_b_flat = map_b.flatten() if len(map_b.shape) > 1 else map_b
    
    valid_a = map_a_flat[~np.isnan(map_a_flat)]
    valid_b = map_b_flat[~np.isnan(map_b_flat)]
    
    if len(valid_a) == 0 or len(valid_b) == 0:
        return {key: np.nan for key in [
            'min_a', 'max_a', 'mean_a', 'median_a',
            'min_b', 'max_b', 'mean_b', 'median_b',
            'min_both', 'max_both', 'mean_both', 'median_both', 'data_range'
        ]}
    
    min_a = float(np.min(valid_a))
    max_a = float(np.max(valid_a))
    mean_a = float(np.mean(valid_a))
    median_a = float(np.median(valid_a))
    
    min_b = float(np.min(valid_b))
    max_b = float(np.max(valid_b))
    mean_b = float(np.mean(valid_b))
    median_b = float(np.median(valid_b))
    
    min_both = min(min_a, min_b)
    max_both = max(max_a, max_b)
    
    both_flats = np.concatenate([valid_a, valid_b])
    mean_both = float(np.mean(both_flats))
    median_both = float(np.median(both_flats))
    
    data_range = max_both - min_both if max_both != min_both else 1.0
    
    return {
        'min_a': min_a, 'max_a': max_a, 'mean_a': mean_a, 'median_a': median_a,
        'min_b': min_b, 'max_b': max_b, 'mean_b': mean_b, 'median_b': median_b,
        'min_both': min_both, 'max_both': max_both, 'mean_both': mean_both, 
        'median_both': median_both, 'data_range': data_range
    }

def load_image(filepath):
    """Carrega uma imagem ou arquivo .npy e retorna como array numpy."""
    if filepath.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
        try:
            img = imread(filepath)
            if len(img.shape) > 2 and img.shape[2] > 1:
                img = np.mean(img, axis=2)
            return img
        except Exception as e:
            print(f"Erro ao carregar imagem {filepath}: {str(e)}")
            return None
    elif filepath.suffix.lower() == '.npy':
        try:
            return np.load(filepath)
        except Exception as e:
            print(f"Erro ao carregar arquivo npy {filepath}: {str(e)}")
            return None
    else:
        print(f"Formato de arquivo não suportado: {filepath}")
        return None

def verify_combined_stats(selection):
    """Verifica consistência das estatísticas combinadas."""
    inconsistencies = 0
    for idx, row in selection.iterrows():
        max_a = row['max_a']
        max_b = row['max_b']
        max_both = row['max_both']
        if np.isnan(max_a) or np.isnan(max_b) or np.isnan(max_both):
            continue
        correct_max = max(max_a, max_b)
        if abs(max_both - correct_max) > 1e-10:
            inconsistencies += 1
            print(f"INCONSISTÊNCIA DETECTADA no registro {idx}:")
            print(f"  max_a = {max_a:.6f}, max_b = {max_b:.6f}")
            print(f"  max_both armazenado = {max_both:.6f}")
            print(f"  max_both correto = {correct_max:.6f}")
    if inconsistencies > 0:
        print(f"\nTotal de inconsistências encontradas: {inconsistências} de {len(selection)} registros")
    else:
        print("\nVerificação de estatísticas combinadas: Todos os valores corretos!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate metrics between datasets')
    parser.add_argument('--metric', type=str, 
                        choices=['pearson', 'rmse', 'residual', 'max_residual', 'min_residual', 
                                 'r2', 'mse', 'mae', 'cosine', 'huber', 'ssim'], 
                        default='pearson',
                        help='Metric to calculate')
    parser.add_argument('--huber-delta', type=float, default=1.0,
                        help='Delta parameter for Huber loss')
    parser.add_argument('--min-residual-percentile', type=float, default=5.0,
                        help='Percentile for min_residual calculation')
    parser.add_argument('--dataset-suffix', type=str, default=None,
                        help='Override the dataset suffix')
    parser.add_argument('--verify-stats', action='store_true',
                        help='Verify strict consistency of statistics')
    parser.add_argument('--top-n', type=int, default=10,
                        help='Number of top maps to display')
    parser.add_argument('--check-existing', type=str, default=None,
                        help='Check an existing results CSV file')
    parser.add_argument('--filter-mapas3', action='store_true',
                        help='Filter only directories starting with mapas3')
    args = parser.parse_args()
    
    if args.check_existing:
        print(f"Verificando consistência em arquivo existente: {args.check_existing}")
        existing_df = pd.read_csv(args.check_existing)
        verify_combined_stats(existing_df)
        exit(0)
    
    metric_type = args.metric
    huber_delta = args.huber_delta
    min_residual_percentile = args.min_residual_percentile
    dataset_suffix = args.dataset_suffix or ('interp_raster' if metric_type == 'ssim' else 'interp')
    verify_stats = args.verify_stats
    top_n = args.top_n
    filter_mapas3 = args.filter_mapas3
    
    higher_is_better = metric_type in ['pearson', 'r2', 'cosine', 'ssim']
    
    base_datasets = {
        'embrace': [
            'mapas1_embrace_2022_2024_0800',
            'mapas1_embrace_2022_2024_1600',
            'mapas1_embrace_2022_2024_2000_2200_0000_0200_0400',
            'mapas3_embrace_2024_0800_30m',
            'mapas3_embrace_2024_1600_30m',
            'mapas3_embrace_2024_2000_0400_30m'
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
            'mapas2_maggia_2024_1600_30m',
            'mapas2_maggia_2022_2024_2000_2200_0000_0200_0400',
            'mapas3_maggia_2024_0800_30m',
            'mapas3_maggia_2024_1600_30m',
            'mapas3_maggia_2024_2000_0400_30m'
        ],
        'nagoya': [
            'mapas1_nagoya_2022_2024_0800',
            'mapas1_nagoya_2022_2024_1600',
            'mapas1_nagoya_2022_2024_2000_2200_0000_0200_0400',
            'mapas2_nagoya_2022_2024_0800',
            'mapas2_nagoya_2022_2024_1600',
            'mapas2_nagoya_2022_2024_2000_2200_0000_0200_0400',
            'mapas3_nagoya_2024_0800_30m',
            'mapas3_nagoya_2024_1600_30m',
            'mapas3_nagoya_2024_2000_0400_30m'
        ]
    }
    
    datasets = {source: [f"{dataset}_{dataset_suffix}" for dataset in dataset_list 
                        if not filter_mapas3 or dataset.startswith('mapas3')] 
                for source, dataset_list in base_datasets.items()}
    
    comparisons = [
        ['embrace', 'igs'], ['embrace', 'maggia'], ['embrace', 'nagoya'],
        ['igs', 'maggia'], ['igs', 'nagoya'], ['maggia', 'nagoya']
    ]
    
    base_dir = Path('.').resolve() / 'output'
    if not base_dir.exists():
        base_dir = Path('.').resolve()
    
    print(f"Calculating {metric_type.upper()} metrics...")
    print(f"Using dataset suffix: '{dataset_suffix}'")
    print(f"Higher values are better: {'YES' if higher_is_better else 'NO'}")
    
    existing_dirs = [path for path in base_dir.glob('*') if path.is_dir() and dataset_suffix in path.name]
    if not existing_dirs:
        print(f"WARNING: No directories found with suffix '{dataset_suffix}' in {base_dir}")
    else:
        print(f"Found {len(existing_dirs)} directories with the expected suffix.")
    
    processed_files = 0
    result = []
    
    for comparison in comparisons:
        for i in range(min(len(datasets[comparison[0]]), len(datasets[comparison[1]]))):
            dataset_a = datasets[comparison[0]][i]
            dataset_b = datasets[comparison[1]][i]
            print(f"\nProcessing {dataset_a} x {dataset_b}")
            
            dir_a = base_dir / dataset_a
            dir_b = base_dir / dataset_b
            
            if not dir_a.exists() or not dir_b.exists():
                print(f"WARNING: Directory not found: {dir_a if not dir_a.exists() else dir_b}")
                continue
            
            if metric_type == 'ssim':
                files_a = []
                for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                    files_a.extend(list(dir_a.glob(f'*{ext}')))
                files_a = sorted(files_a)
            else:
                files_a = sorted(list(dir_a.glob('*.npy')))
            
            if not files_a:
                print(f"WARNING: No {'image' if metric_type == 'ssim' else '.npy'} files found in {dir_a}")
                continue
            print(f"Found {len(files_a)} {'image' if metric_type == 'ssim' else '.npy'} files in {dataset_a}")
            
            for file_a in files_a:
                if metric_type == 'ssim':
                    try:
                        timestamp = file_a.stem.split('_')[0]
                        matching_files_b = list(dir_b.glob(f"{timestamp}*"))
                        if not matching_files_b:
                            continue
                        file_b = matching_files_b[0]
                    except:
                        continue
                else:
                    file_b = dir_b / file_a.name
                    if not file_b.exists():
                        continue
                
                try:
                    map_a = load_image(file_a)
                    map_b = load_image(file_b)
                    if map_a is None or map_b is None:
                        continue
                    
                    map_a = np.nan_to_num(map_a, nan=np.nan)
                    map_b = np.nan_to_num(map_b, nan=np.nan)
                    map_a_flat = map_a.flatten()
                    map_b_flat = map_b.flatten()
                    
                    processed_files += 1
                    stats = calculate_strict_stats(map_a, map_b)
                    
                    if metric_type == 'pearson':
                        metric_value = calculate_pearson(map_a_flat, map_b_flat)
                    elif metric_type == 'r2':
                        metric_value = calculate_r2_score(map_a_flat, map_b_flat)
                    elif metric_type == 'rmse':
                        metric_value = calculate_rmse(map_a_flat, map_b_flat)
                    elif metric_type == 'mse':
                        metric_value = calculate_mse(map_a_flat, map_b_flat)
                    elif metric_type == 'mae':
                        metric_value = calculate_mae(map_a_flat, map_b_flat)
                    elif metric_type == 'residual':
                        metric_value = calculate_residual_error(map_a_flat, map_b_flat)
                        if not np.isnan(metric_value) and metric_value < 0:
                            print(f"ERROR: Negative residual computed for {file_a}: {metric_value}")
                    elif metric_type == 'max_residual':
                        metric_value = calculate_max_residual_error(map_a_flat, map_b_flat)
                    elif metric_type == 'min_residual':
                        metric_value = calculate_min_residual_error(map_a_flat, map_b_flat, min_residual_percentile)
                    elif metric_type == 'cosine':
                        metric_value = calculate_cosine_similarity(map_a_flat, map_b_flat)
                    elif metric_type == 'huber':
                        metric_value = calculate_huber_loss(map_a_flat, map_b_flat, huber_delta)
                    elif metric_type == 'ssim':
                        metric_value = calculate_ssim(map_a, map_b)
                    
                    if verify_stats and not np.isnan(stats['min_both']) and not np.isnan(stats['max_both']):
                        both_maps = np.concatenate([map_a_flat[~np.isnan(map_a_flat)], map_b_flat[~np.isnan(map_b_flat)]])
                        trad_min = np.min(both_maps) if len(both_maps) > 0 else np.nan
                        trad_max = np.max(both_maps) if len(both_maps) > 0 else np.nan
                        if not np.isnan(trad_min) and not np.isnan(trad_max):
                            if abs(stats['min_both'] - trad_min) > 1e-10 or abs(stats['max_both'] - trad_max) > 1e-10:
                                print(f"INCONSISTENCY DETECTED in file {file_a.name}")
                    
                    try:
                        epoch = np.datetime64(file_a.stem.split('_')[0].replace('.', ':'))
                    except:
                        epoch = np.datetime64('1970-01-01T00:00:00')
                        print(f"Warning: Could not parse datetime from filename {file_a.name}")
                    
                    value_p = metric_value * 100 if metric_type in ['pearson', 'r2', 'cosine', 'ssim'] and not np.isnan(metric_value) else \
                              (metric_value / stats['data_range'] * 100 if metric_type in ['rmse', 'residual'] and stats['data_range'] != 0 and not np.isnan(metric_value) else metric_value)
                    
                    result_data = {
                        'datetime': epoch,
                        'comparison': f'{comparison[0]} x {comparison[1]}',
                        'dataset_a': dataset_a,
                        'dataset_b': dataset_b,
                        'source_a': comparison[0],
                        'source_b': comparison[1],
                        'filename_a': file_a.name,
                        'filename_b': file_b.name if metric_type == 'ssim' else file_a.name,
                        metric_type: metric_value,
                        f'{metric_type}_p': value_p,
                        **stats
                    }
                    result.append(result_data)
                    if processed_files % 10 == 0:
                        print(f"Processed {processed_files} file pairs...")
                except Exception as e:
                    print(f"Error processing files {file_a} and {file_b}: {str(e)}")
    
    if not result:
        print("\nERROR: No data found for analysis.")
        exit(1)
    
    print(f"\nProcessed {processed_files} file pairs successfully.")
    
    df = pd.DataFrame(result)
    df.to_csv(f'result_{metric_type}_with_stats.csv', index=False)
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.sort_values('datetime', inplace=True)
    
    dataset_metrics = defaultdict(list)
    dataset_total_metrics = defaultdict(float)
    dataset_count = defaultdict(int)
    top_maps_by_comparison = {}
    
    for comparison in comparisons:
        for i in range(min(len(datasets[comparison[0]]), len(datasets[comparison[1]]))):
            dataset_a = datasets[comparison[0]][i]
            dataset_b = datasets[comparison[1]][i]
            comparison_type = f'{comparison[0]} x {comparison[1]}'
            selection = df.loc[(df['comparison'] == comparison_type) & 
                             (df['dataset_a'] == dataset_a) & 
                             (df['dataset_b'] == dataset_b)]
            
            if selection.empty:
                print(f"No data found for {dataset_a} x {dataset_b}")
                continue
                
            print(f"\n{comparison[0].upper()} x {comparison[1].upper()}")
            print(f"{dataset_a} x {dataset_b}")
            print(f"Number of comparisons: {len(selection)}")
            
            verify_combined_stats(selection)
            
            metric_values = selection[metric_type].values
            valid_metrics = metric_values[~np.isnan(metric_values)]
            
            if metric_type == 'residual' and any(v < 0 for v in valid_metrics):
                print(f"WARNING: Negative residuals detected in {dataset_a} x {dataset_b}: {valid_metrics}")
            
            if len(valid_metrics) > 0:
                if metric_type in ['pearson', 'r2']:
                    z_values = [fisher_z_transform(v) for v in valid_metrics if not np.isnan(fisher_z_transform(v))]
                    metric_value = fisher_z_inverse(np.mean(z_values)) if z_values else np.nan
                else:
                    metric_value = np.nanmean(np.abs(valid_metrics)) if metric_type == 'residual' else np.nanmean(valid_metrics)
            else:
                metric_value = np.nan
            
            if np.isnan(metric_value):
                print(f"WARNING: Average {metric_type} is NaN for {dataset_a} x {dataset_b}")
            
            dataset_metrics[comparison[0]].extend(valid_metrics)
            dataset_metrics[comparison[1]].extend(valid_metrics)
            
            dataset_total_metrics[comparison[0]] += np.nansum(metric_values)
            dataset_total_metrics[comparison[1]] += np.nansum(metric_values)
            dataset_count[comparison[0]] += len(valid_metrics)
            dataset_count[comparison[1]] += len(valid_metrics)
            
            mean_min_a = selection['min_a'].mean()
            mean_max_a = selection['max_a'].mean()
            mean_mean_a = selection['mean_a'].mean()
            mean_median_a = selection['median_a'].mean()
            
            mean_min_b = selection['min_b'].mean()
            mean_max_b = selection['max_b'].mean()
            mean_mean_b = selection['mean_b'].mean()
            mean_median_b = selection['median_b'].mean()
            
            mean_min_both = min(mean_min_a, mean_min_b) if not np.isnan(mean_min_a) and not np.isnan(mean_min_b) else np.nan
            mean_max_both = max(mean_max_a, mean_max_b) if not np.isnan(mean_max_a) and not np.isnan(mean_max_b) else np.nan
            mean_mean_both = selection['mean_both'].mean()
            mean_median_both = selection['median_both'].mean()
            mean_data_range = mean_max_both - mean_min_both if not np.isnan(mean_max_both) and not np.isnan(mean_min_both) else np.nan
            
            print("\nStatistics for All Data:")
            print("Dataset A:")
            print(f"  Min: {mean_min_a:.4f}, Median: {mean_median_a:.4f}, Mean: {mean_mean_a:.4f}, Max: {mean_max_a:.4f}")
            print("Dataset B:")
            print(f"  Min: {mean_min_b:.4f}, Median: {mean_median_b:.4f}, Mean: {mean_mean_b:.4f}, Max: {mean_max_b:.4f}")
            print("Combined:")
            print(f"  Min: {mean_min_both:.4f}, Median: {mean_median_both:.4f}, Mean: {mean_mean_both:.4f}, Max: {mean_max_both:.4f}")
            print(f"Data Range: {mean_data_range:.4f}")
            
            if metric_type == 'pearson':
                print(f'Average Pearson Correlation: {metric_value:.4f} ({metric_value * 100:.2f}% if not np.isnan(metric_value) else "NaN%") (Fisher Z applied)')
            elif metric_type == 'r2':
                print(f'Average R² Score: {metric_value:.4f} ({metric_value * 100:.2f}% if not np.isnan(metric_value) else "NaN%") (Fisher Z applied)')
            elif metric_type == 'ssim':
                print(f'Average Structural Similarity Index: {metric_value:.4f} ({metric_value * 100:.2f}% if not np.isnan(metric_value) else "NaN%")')
            elif metric_type == 'cosine':
                print(f'Average Cosine Similarity: {metric_value:.4f} ({metric_value * 100:.2f}% if not np.isnan(metric_value) else "NaN%")')
            elif metric_type == 'rmse':
                rmse_percent = (metric_value / mean_data_range * 100) if not np.isnan(metric_value) and mean_data_range != 0 else np.nan
                print(f'Average RMSE: {metric_value:.4f} ({rmse_percent:.2f}% of data range if not np.isnan(rmse_percent) else "NaN%")')
            elif metric_type == 'mse':
                print(f'Average Mean Squared Error: {metric_value:.4f}')
            elif metric_type == 'mae':
                print(f'Average Mean Absolute Error: {metric_value:.4f}')
            elif metric_type == 'residual':
                residual_percent = (metric_value / mean_data_range * 100) if not np.isnan(metric_value) and mean_data_range != 0 else np.nan
                print(f'Average Mean Absolute Residual Error: {metric_value:.4f} ({residual_percent:.2f}% of data range if not np.isnan(residual_percent) else "NaN%")')
            elif metric_type == 'max_residual':
                print(f'Average Maximum Residual Error: {metric_value:.4f}')
            elif metric_type == 'min_residual':
                print(f'Average Minimum Residual Error ({min_residual_percentile}th percentile): {metric_value:.4f}')
            elif metric_type == 'huber':
                print(f'Average Huber Loss (delta={huber_delta}): {metric_value:.4f}')
            
            sorted_maps = selection.sort_values(by=metric_type, ascending=not higher_is_better).head(top_n)
            comp_key = f"{dataset_a} x {dataset_b}"
            top_maps_by_comparison[comp_key] = sorted_maps
            
            print(f"\nTop {top_n} Maps with Best {metric_type.upper()} Values:")
            print("-" * 120)
            for idx, row in enumerate(sorted_maps.itertuples(), 1):
                file_info = f"{row.filename_a} & {row.filename_b}" if metric_type == 'ssim' else row.filename_a
                metric_display = f"{getattr(row, metric_type):.4f} ({getattr(row, f'{metric_type}_p'):.2f}% if not np.isnan(getattr(row, f'{metric_type}_p')) else 'NaN%')" \
                                if metric_type in ['pearson', 'r2', 'cosine', 'ssim', 'rmse', 'residual'] else f"{getattr(row, metric_type):.4f}"
                
                date_str = pd.to_datetime(row.datetime).strftime('%Y-%m-%d %H:%M') if hasattr(row, 'datetime') else 'Unknown'
                
                print(f"{idx}. Data: {date_str}")
                print(f"   Comparação: {row.dataset_a} x {row.dataset_b}")
                print(f"   Arquivos: {file_info}")
                print(f"   {metric_type.upper()}: {metric_display}")
                print(f"   Estatísticas do Dataset A:")
                print(f"     Min: {row.min_a:.4f}, Median: {row.median_a:.4f}, Mean: {row.mean_a:.4f}, Max: {row.max_a:.4f}")
                print(f"   Estatísticas do Dataset B:")
                print(f"     Min: {row.min_b:.4f}, Median: {row.median_b:.4f}, Mean: {row.mean_b:.4f}, Max: {row.max_b:.4f}")
                print(f"   Estatísticas Combinadas:")
                print(f"     Min: {row.min_both:.4f}, Median: {row.median_both:.4f}, Mean: {row.mean_both:.4f}, Max: {row.max_both:.4f}")
                print(f"     Data Range: {row.data_range:.4f}")
                if idx < len(sorted_maps):
                    print("-" * 80)
            print("-" * 120)
            
            if 'datetime' in selection.columns:
                for year in [2022, 2023, 2024]:
                    for month in range(1, 13):
                        month_data = selection.loc[(selection['datetime'].dt.month == month) &
                                                 (selection['datetime'].dt.year == year)]
                        if not month_data.empty:
                            month_name = pd.Timestamp(year=year, month=month, day=1).strftime('%B/%Y')
                            month_metric = np.nanmean(np.abs(month_data[metric_type])) if metric_type == 'residual' else np.nanmean(month_data[metric_type])
                            if metric_type in ['pearson', 'r2', 'cosine', 'ssim', 'rmse', 'residual']:
                                print(f'{month_name}: {metric_type.upper()} = {month_metric:.4f} ({month_metric * 100:.2f}% if not np.isnan(month_metric) else "NaN%")')
                            else:
                                print(f'{month_name}: {metric_type.upper()} = {month_metric:.4f}')
    
    dataset_avg_metrics = {}
    
    print("\nCalculating average metrics...")
    for dataset in dataset_metrics:
        valid_metrics = [v for v in dataset_metrics[dataset] if not np.isnan(v)]
        if valid_metrics:
            if metric_type in ['pearson', 'r2']:
                z_values = [fisher_z_transform(v) for v in valid_metrics if not np.isnan(fisher_z_transform(v))]
                if z_values:
                    avg_z = np.mean(z_values)
                    dataset_avg_metrics[dataset] = fisher_z_inverse(avg_z)
                else:
                    dataset_avg_metrics[dataset] = np.nan
                    print(f"WARNING: No valid Z values for {dataset}")
            else:
                dataset_avg_metrics[dataset] = np.mean(np.abs(valid_metrics)) if metric_type == 'residual' else np.mean(valid_metrics)
        else:
            dataset_avg_metrics[dataset] = np.nan
            print(f"WARNING: No valid metrics for {dataset}")
    
    best_dataset = max(dataset_avg_metrics.items(), key=lambda x: x[1] if not np.isnan(x[1]) else float('-inf')) if higher_is_better else \
                   min(dataset_avg_metrics.items(), key=lambda x: x[1] if not np.isnan(x[1]) else float('inf'))
    worst_dataset = min(dataset_avg_metrics.items(), key=lambda x: x[1] if not np.isnan(x[1]) else float('inf')) if higher_is_better else \
                    max(dataset_avg_metrics.items(), key=lambda x: x[1] if not np.isnan(x[1]) else float('-inf'))
    
    metric_name = {
        'pearson': 'PEARSON CORRELATION',
        'r2': 'R² SCORE',
        'ssim': 'STRUCTURAL SIMILARITY INDEX',
        'cosine': 'COSINE SIMILARITY',
        'rmse': 'ROOT MEAN SQUARED ERROR',
        'mse': 'MEAN SQUARED ERROR',
        'mae': 'MEAN ABSOLUTE ERROR',
        'residual': 'MEAN ABSOLUTE RESIDUAL ERROR',
        'max_residual': 'MAXIMUM RESIDUAL ERROR',
        'min_residual': f'MINIMUM RESIDUAL ERROR ({min_residual_percentile}th PERCENTILE)',
        'huber': f'HUBER LOSS (delta={huber_delta})'
    }.get(metric_type, metric_type.upper())
    
    print(f"\n===== DATASET {metric_name} ANALYSIS =====")
    print(f"Best dataset: {best_dataset[0]} with average {metric_type} of {best_dataset[1]:.4f}" + 
          (f" ({best_dataset[1] * 100:.2f}% if not np.isnan(best_dataset[1]) else 'NaN%')" if metric_type in ['pearson', 'r2', 'cosine', 'ssim'] else ""))
    print(f"Worst dataset: {worst_dataset[0]} with average {metric_type} of {worst_dataset[1]:.4f}" + 
          (f" ({worst_dataset[1] * 100:.2f}% if not np.isnan(worst_dataset[1]) else 'NaN%')" if metric_type in ['pearson', 'r2', 'cosine', 'ssim'] else ""))
    
    print("\nDataset Ranking (from best to worst):")
    sorted_datasets = sorted(dataset_avg_metrics.items(), key=lambda x: x[1] if not np.isnan(x[1]) else float('inf'), reverse=higher_is_better)
    for i, (dataset, avg_val) in enumerate(sorted_datasets, 1):
        print(f"{i}. {dataset}: {avg_val:.4f}" + 
              (f" ({avg_val * 100:.2f}% if not np.isnan(avg_val) else 'NaN%')" if metric_type in ['pearson', 'r2', 'cosine', 'ssim'] else ""))
    
    print(f"\n===== TOP {top_n} MAPS OVERALL =====")
    top_overall = df.sort_values(by=metric_type, ascending=not higher_is_better).head(top_n)
    print("-" * 120)
    for idx, row in enumerate(top_overall.itertuples(), 1):
        file_info = f"{row.filename_a} & {row.filename_b}" if metric_type == 'ssim' else row.filename_a
        metric_display = f"{getattr(row, metric_type):.4f} ({getattr(row, f'{metric_type}_p'):.2f}% if not np.isnan(getattr(row, f'{metric_type}_p')) else 'NaN%')" \
                        if metric_type in ['pearson', 'r2', 'cosine', 'ssim', 'rmse', 'residual'] else f"{getattr(row, metric_type):.4f}"
        
        date_str = pd.to_datetime(row.datetime).strftime('%Y-%m-%d %H:%M') if hasattr(row, 'datetime') else 'Unknown'
        
        print(f"{idx}. Data: {date_str}")
        print(f"   Comparação: {row.dataset_a} x {row.dataset_b}")
        print(f"   Arquivos: {file_info}")
        print(f"   {metric_type.upper()}: {metric_display}")
        print(f"   Estatísticas do Dataset A:")
        print(f"     Min: {row.min_a:.4f}, Median: {row.median_a:.4f}, Mean: {row.mean_a:.4f}, Max: {row.max_a:.4f}")
        print(f"   Estatísticas do Dataset B:")
        print(f"     Min: {row.min_b:.4f}, Median: {row.median_b:.4f}, Mean: {row.mean_b:.4f}, Max: {row.max_b:.4f}")
        print(f"   Estatísticas Combinadas:")
        print(f"     Min: {row.min_both:.4f}, Median: {row.median_both:.4f}, Mean: {row.mean_both:.4f}, Max: {row.max_both:.4f}")
        print(f"     Data Range: {row.data_range:.4f}")
        if idx < len(top_overall):
            print("-" * 80)
    print("-" * 120)
    
    overall_metrics = pd.DataFrame({
        'dataset': list(dataset_avg_metrics.keys()),
        f'avg_{metric_type}': list(dataset_avg_metrics.values()),
        f'avg_{metric_type}_percent': [v * 100 if not np.isnan(v) else np.nan for v in dataset_avg_metrics.values()] 
                                      if metric_type in ['pearson', 'r2', 'cosine', 'ssim'] else [np.nan] * len(dataset_avg_metrics)
    })
    if overall_metrics[f'avg_{metric_type}_percent'].isna().all():
        overall_metrics = overall_metrics.drop(columns=[f'avg_{metric_type}_percent'])
    
    overall_metrics.to_csv(f'dataset_ranking_{metric_type}.csv', index=False)
    print(f"\nDataset ranking saved to 'dataset_ranking_{metric_type}.csv'")
    
    print("\n===== COMPUTATION SUMMARY =====")
    print(f"Metric: {metric_name}")
    print(f"Dataset type: {dataset_suffix}")
    print(f"Files processed: {processed_files}")
    print(f"Datasets compared: {len(dataset_avg_metrics)}")
    print("Robust handling of NaN values in all metric calculations")
    print("Done!")

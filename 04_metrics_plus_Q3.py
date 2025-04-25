import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import mean_squared_error, mean_absolute_error
from skimage.metrics import structural_similarity as ssim
from skimage.io import imread
import warnings

warnings.filterwarnings('ignore')

def calculate_q3_mask(map_data):
    """Calcula Q3 (75º percentil) e retorna máscara para pixels >= Q3."""
    # Create mask of non-NaN pixels
    non_nan_mask = ~np.isnan(map_data)
    
    # Get valid (non-NaN) pixels for percentile calculation
    valid_pixels = map_data[non_nan_mask]
    
    if len(valid_pixels) == 0:
        return None, np.nan
        
    # Calculate Q3 from valid pixels
    q3 = np.percentile(valid_pixels, 75)
    
    # Special handling for cases where Q3 equals max value
    unique_values = np.unique(valid_pixels)
    if q3 == np.max(valid_pixels) and len(unique_values) > 1:
        # If Q3 equals the max value but there are other values,
        # try using a slightly lower threshold to ensure variance
        q3 = unique_values[-2]  # Use second highest value
    
    # Create a mask where pixels are both non-NaN AND >= Q3
    mask = non_nan_mask & (map_data >= q3)
    
    # Ensure we have enough pixels (at least 5% of original valid pixels)
    min_required = max(100, len(valid_pixels) * 0.05)
    if np.sum(mask) < min_required:
        # Try a more lenient threshold
        q3_adjusted = np.percentile(valid_pixels, 65)  # Try 65th percentile instead
        mask = non_nan_mask & (map_data >= q3_adjusted)
        if np.sum(mask) >= min_required:
            return mask, q3_adjusted
        return None, q3
    
    return mask, q3

def calculate_pearson(y_true, y_pred, filename="unknown", pixel_mask=None):
    """Calcula a correlação de Pearson, tratando casos especiais."""
    if pixel_mask is not None:
        if pixel_mask.sum() < 2:
            return np.nan
        y_true = y_true[pixel_mask]
        y_pred = y_pred[pixel_mask]
    if len(y_true) == 0 or len(y_pred) == 0 or np.all(np.isnan(y_true)) or np.all(np.isnan(y_pred)):
        return np.nan
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if np.sum(valid_mask) < 2:
        return np.nan
    return np.corrcoef(y_true[valid_mask], y_pred[valid_mask])[0, 1]

def calculate_r2_score(y_true, y_pred, filename="unknown", pixel_mask=None):
    """Calcula o R² como o quadrado da correlação de Pearson."""
    if pixel_mask is not None:
        if pixel_mask.sum() < 2:
            return np.nan, np.nan
        y_true = y_true[pixel_mask]
        y_pred = y_pred[pixel_mask]
    pearson_r = calculate_pearson(y_true, y_pred, filename)
    if np.isnan(pearson_r):
        return np.nan, np.nan
    return pearson_r ** 2, pearson_r

def calculate_rmse(y_true, y_pred, pixel_mask=None):
    """Calcula o RMSE, tratando casos especiais."""
    if pixel_mask is not None:
        if pixel_mask.sum() < 2:
            return np.nan
        y_true = y_true[pixel_mask]
        y_pred = y_pred[pixel_mask]
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if np.sum(valid_mask) < 2:
        return np.nan
    return np.sqrt(mean_squared_error(y_true[valid_mask], y_pred[valid_mask]))

def calculate_mse(y_true, y_pred, pixel_mask=None):
    """Calcula o MSE, tratando casos especiais."""
    if pixel_mask is not None:
        if pixel_mask.sum() < 2:
            return np.nan
        y_true = y_true[pixel_mask]
        y_pred = y_pred[pixel_mask]
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if np.sum(valid_mask) < 2:
        return np.nan
    return mean_squared_error(y_true[valid_mask], y_pred[valid_mask])

def calculate_mae(y_true, y_pred, pixel_mask=None):
    """Calcula o MAE, tratando casos especiais."""
    if pixel_mask is not None:
        if pixel_mask.sum() < 2:
            return np.nan
        y_true = y_true[pixel_mask]
        y_pred = y_pred[pixel_mask]
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if np.sum(valid_mask) < 2:
        return np.nan
    return mean_absolute_error(y_true[valid_mask], y_pred[valid_mask])

def calculate_residual_error(y_true, y_pred, normalize=False, filename="unknown", pixel_mask=None):
    """Calcula o erro residual médio absoluto com validação robusta."""
    if pixel_mask is not None:
        if pixel_mask.sum() < 2:
            return np.nan
        y_true = y_true[pixel_mask]
        y_pred = y_pred[pixel_mask]
    if len(y_true) == 0 or len(y_pred) == 0 or np.all(np.isnan(y_true)) or np.all(np.isnan(y_pred)):
        return np.nan
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if np.sum(valid_mask) < 2:
        return np.nan
    
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    
    if not (np.isfinite(y_true_valid).all() and np.isfinite(y_pred_valid).all()):
        return np.nan
    
    if normalize:
        mean_true = np.mean(y_true_valid)
        std_true = np.std(y_true_valid)
        mean_pred = np.mean(y_pred_valid)
        std_pred = np.std(y_pred_valid)
        if std_pred != 0 and std_true != 0:
            y_pred_valid = (y_pred_valid - mean_pred) / std_pred * std_true + mean_true
    
    return np.mean(np.abs(y_true_valid - y_pred_valid))

def calculate_max_residual_error(y_true, y_pred, normalize=False, filename="unknown", pixel_mask=None):
    """Calcula o erro residual máximo com validação robusta."""
    if pixel_mask is not None:
        if pixel_mask.sum() < 2:
            return np.nan
        y_true = y_true[pixel_mask]
        y_pred = y_pred[pixel_mask]
    if len(y_true) == 0 or len(y_pred) == 0 or np.all(np.isnan(y_true)) or np.all(np.isnan(y_pred)):
        return np.nan
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if np.sum(valid_mask) < 2:
        return np.nan
    
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    
    if not (np.isfinite(y_true_valid).all() and np.isfinite(y_pred_valid).all()):
        return np.nan
    
    if normalize:
        mean_true = np.mean(y_true_valid)
        std_true = np.std(y_true_valid)
        mean_pred = np.mean(y_pred_valid)
        std_pred = np.std(y_pred_valid)
        if std_pred != 0 and std_true != 0:
            y_pred_valid = (y_pred_valid - mean_pred) / std_pred * std_true + mean_true
    
    return np.max(np.abs(y_true_valid - y_pred_valid))

def calculate_min_residual_error(y_true, y_pred, percentile=5.0, normalize=False, filename="unknown", pixel_mask=None):
    """Calcula o erro residual no percentil especificado com validação robusta."""
    if pixel_mask is not None:
        if pixel_mask.sum() < 2:
            return np.nan
        y_true = y_true[pixel_mask]
        y_pred = y_pred[pixel_mask]
    if len(y_true) == 0 or len(y_pred) == 0 or np.all(np.isnan(y_true)) or np.all(np.isnan(y_pred)):
        return np.nan
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if np.sum(valid_mask) < 2:
        return np.nan
    
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    
    if not (np.isfinite(y_true_valid).all() and np.isfinite(y_pred_valid).all()):
        return np.nan
    
    if normalize:
        mean_true = np.mean(y_true_valid)
        std_true = np.std(y_true_valid)
        mean_pred = np.mean(y_pred_valid)
        std_pred = np.std(y_pred_valid)
        if std_pred != 0 and std_true != 0:
            y_pred_valid = (y_pred_valid - mean_pred) / std_pred * std_true + mean_true
    
    return np.percentile(np.abs(y_true_valid - y_pred_valid), percentile)

def calculate_cosine_similarity(y_true, y_pred, pixel_mask=None):
    """Calcula a similaridade de cosseno, tratando casos especiais."""
    if pixel_mask is not None:
        if pixel_mask.sum() < 2:
            return np.nan
        y_true = y_true[pixel_mask]
        y_pred = y_pred[pixel_mask]
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

def calculate_huber_loss(y_true, y_pred, delta=1.0, pixel_mask=None):
    """Calcula a perda de Huber, tratando casos especiais."""
    if pixel_mask is not None:
        if pixel_mask.sum() < 2:
            return np.nan
        y_true = y_true[pixel_mask]
        y_pred = y_pred[pixel_mask]
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if np.sum(valid_mask) < 2:
        return np.nan
    errors = y_true[valid_mask] - y_pred[valid_mask]
    abs_errors = np.abs(errors)
    quadratic = np.minimum(abs_errors, delta)
    linear = abs_errors - quadratic
    return np.mean(0.5 * quadratic * quadratic + delta * linear)

def calculate_ssim(y_true, y_pred):
    """Calcula o SSIM, tratando corretamente os casos com valores NaN."""
    # Converte para escala de cinza se for imagem colorida
    if len(y_true.shape) > 2 and y_true.shape[2] > 1:
        y_true = np.mean(y_true, axis=2)
    if len(y_pred.shape) > 2 and y_pred.shape[2] > 1:
        y_pred = np.mean(y_pred, axis=2)
    
    # Cria uma máscara de valores válidos (não NaN)
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    
    # Verifica se há pixels válidos suficientes
    if np.sum(valid_mask) < 2:
        return np.nan
    
    # Cria cópias das imagens onde substituímos NaN por um valor válido
    # apenas para cálculo do SSIM
    y_true_valid = y_true.copy()
    y_pred_valid = y_pred.copy()
    
    # Determina um valor de preenchimento (pode ser a média dos valores válidos)
    fill_value_true = np.mean(y_true[valid_mask]) if np.any(valid_mask) else 0
    fill_value_pred = np.mean(y_pred[valid_mask]) if np.any(valid_mask) else 0
    
    # Preenche valores NaN com os valores calculados
    y_true_valid[~valid_mask] = fill_value_true
    y_pred_valid[~valid_mask] = fill_value_pred
    
    # Calcula o intervalo de dados para valores válidos
    data_range = max(np.max(y_true[valid_mask]) - np.min(y_true[valid_mask]), 
                     np.max(y_pred[valid_mask]) - np.min(y_pred[valid_mask]))
    if data_range == 0:
        data_range = 1
    
    # Tenta calcular SSIM
    try:
        # Se a maioria dos pixels (>50%) não for válida, retorna NaN
        if np.sum(valid_mask) < 0.5 * valid_mask.size:
            return np.nan
        
        return ssim(y_true_valid, y_pred_valid, data_range=data_range)
    except Exception as e:
        # Se houver erro por diferença de dimensões
        if y_true.shape != y_pred.shape:
            min_height = min(y_true.shape[0], y_pred.shape[0])
            min_width = min(y_true.shape[1], y_pred.shape[1])
            y_true_resized = y_true_valid[:min_height, :min_width]
            y_pred_resized = y_pred_valid[:min_height, :min_width]
            
            # Tenta novamente com as imagens redimensionadas
            try:
                return ssim(y_true_resized, y_pred_resized, data_range=data_range)
            except Exception:
                return np.nan
        
        print(f"Erro ao calcular SSIM: {e}")
        return np.nan

def fisher_z_transform(r):
    """Transforma correlação r para valor z, evitando valores extremos."""
    if np.isnan(r) or abs(r) >= 1:
        return np.nan
    r = np.clip(r, -0.9999, 0.9999)
    return 0.5 * np.log((1 + r) / (1 - r))

def fisher_z_inverse(z):
    """Transforma z de volta para correlação r."""
    if np.isnan(z):
        return np.nan
    return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)

def calculate_strict_stats(map_a, map_b):
    """Calcula estatísticas consistentes para os mapas, incluindo Q1 e Q3."""
    map_a_flat = map_a.flatten() if len(map_a.shape) > 1 else map_a
    map_b_flat = map_b.flatten() if len(map_b.shape) > 1 else map_b
    
    valid_a = map_a_flat[~np.isnan(map_a_flat)]
    valid_b = map_b_flat[~np.isnan(map_b_flat)]
    
    if len(valid_a) == 0 or len(valid_b) == 0:
        return {key: np.nan for key in [
            'min_a', 'q1_a', 'median_a', 'q3_a', 'max_a', 'mean_a',
            'min_b', 'q1_b', 'median_b', 'q3_b', 'max_b', 'mean_b',
            'min_both', 'q1_both', 'median_both', 'q3_both', 'max_both', 'mean_both', 
            'data_range'
        ]}
    
    min_a = float(np.min(valid_a))
    q1_a = float(np.percentile(valid_a, 25))  # 25º percentil = Q1
    median_a = float(np.median(valid_a))
    q3_a = float(np.percentile(valid_a, 75))  # 75º percentil = Q3
    max_a = float(np.max(valid_a))
    mean_a = float(np.mean(valid_a))
    
    min_b = float(np.min(valid_b))
    q1_b = float(np.percentile(valid_b, 25))
    median_b = float(np.median(valid_b))
    q3_b = float(np.percentile(valid_b, 75))
    max_b = float(np.max(valid_b))
    mean_b = float(np.mean(valid_b))
    
    min_both = min(min_a, min_b)
    max_both = max(max_a, max_b)
    
    both_flats = np.concatenate([valid_a, valid_b])
    q1_both = float(np.percentile(both_flats, 25))
    median_both = float(np.median(both_flats))
    q3_both = float(np.percentile(both_flats, 75))
    mean_both = float(np.mean(both_flats))
    
    data_range = max_both - min_both if max_both > min_both else 1.0
    
    return {
        'min_a': min_a, 'q1_a': q1_a, 'median_a': median_a, 'q3_a': q3_a, 'max_a': max_a, 'mean_a': mean_a,
        'min_b': min_b, 'q1_b': q1_b, 'median_b': median_b, 'q3_b': q3_b, 'max_b': max_b, 'mean_b': mean_b,
        'min_both': min_both, 'q1_both': q1_both, 'median_both': median_both, 'q3_both': q3_both, 
        'max_both': max_both, 'mean_both': mean_both, 'data_range': data_range
    }

def load_image(filepath):
    """Carrega uma imagem ou arquivo .npy e retorna como array numpy."""
    if filepath.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
        try:
            img = imread(filepath)
            if len(img.shape) > 2 and img.shape[2] > 1:
                img = np.mean(img, axis=2)
            return img
        except Exception:
            return None
    elif filepath.suffix.lower() == '.npy':
        try:
            return np.load(filepath)
        except Exception:
            return None
    else:
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
    return inconsistencies == 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate metrics between datasets with simplified R², robust residuals, and Q3-based pixel selection')
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
    parser.add_argument('--top-n', type=int, default=50,
                        help='Number of top maps to display')
    parser.add_argument('--check-existing', type=str, default=None,
                        help='Check an existing results CSV file')
    parser.add_argument('--filter-mapas3', action='store_true',
                        help='Filter only directories starting with mapas3')
    parser.add_argument('--swap-ytrue-ypred', action='store_true',
                        help='Swap y_true and y_pred for R² and residual calculations')
    parser.add_argument('--normalize-residuals', action='store_true',
                        help='Normalize y_pred for residual calculations')
    args = parser.parse_args()
    
    if args.check_existing:
        existing_df = pd.read_csv(args.check_existing)
        if verify_combined_stats(existing_df):
            print("\nVerificação de estatísticas combinadas: Todos os valores corretos!")
        else:
            print(f"\nTotal de inconsistências encontradas em {args.check_existing}")
        exit(0)
    
    metric_type = args.metric
    huber_delta = args.huber_delta
    min_residual_percentile = args.min_residual_percentile
    dataset_suffix = args.dataset_suffix or ('interp_raster' if metric_type == 'ssim' else 'interp')
    verify_stats = args.verify_stats
    top_n = args.top_n
    filter_mapas3 = args.filter_mapas3
    swap_ytrue_ypred = args.swap_ytrue_ypred
    normalize_residuals = args.normalize_residuals
    
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
            'mapas2_nagoya_2024_1600_30m',
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
        ['embrace', 'igs'],
        ['embrace', 'maggia'],
        ['embrace', 'nagoya'],
        ['igs', 'maggia'],
        ['igs', 'nagoya'],
        ['maggia', 'nagoya']
    ]
    
    base_dir = Path('.').resolve() / 'output'
    if not base_dir.exists():
        print(f"Output directory {base_dir} does not exist.")
        exit(1)
    
    print(f"Calculating {metric_type.upper()} metrics with Q3-based pixel selection...")
    print(f"Using dataset suffix: '{dataset_suffix}'")
    print(f"Higher values are better: {'YES' if higher_is_better else 'NO'}")
    if metric_type == 'r2' and swap_ytrue_ypred:
        print("Swapping y_true and y_pred for R² calculation")
    if metric_type in ['residual', 'max_residual', 'min_residual'] and normalize_residuals:
        print("Normalizing y_pred for residual calculations")
    
    # Determine file extension based on metric type
    file_extension = '*.png' if metric_type == 'ssim' else '*.npy'
    
    # Build list of expected directories based on defined datasets
    all_expected_dirs = [base_dir / dataset for source in datasets for dataset in datasets[source]]
    existing_dirs = [d for d in all_expected_dirs if d.is_dir() and any(d.glob(file_extension))]
    
    if not existing_dirs:
        print(f"No valid directories found with suffix '{dataset_suffix}' containing {file_extension} files in {base_dir}")
        exit(1)
    
    print(f"Found {len(existing_dirs)} directories with the expected suffix and {file_extension} files:")
    for d in existing_dirs:
        print(f"  - {d.name}")
    
    processed_files = 0
    skipped_files = 0
    result = []
    
    for comparison in comparisons:
        source_a, source_b = comparison
        if not datasets[source_a] or not datasets[source_b]:
            continue
        
        # Filter datasets to only those with existing directories
        valid_pairs = []
        for i in range(min(len(datasets[source_a]), len(datasets[source_b]))):
            dataset_a = datasets[source_a][i]
            dataset_b = datasets[source_b][i]
            dir_a = base_dir / dataset_a
            dir_b = base_dir / dataset_b
            if dir_a in existing_dirs and dir_b in existing_dirs:
                valid_pairs.append((dataset_a, dataset_b))
        
        if not valid_pairs:
            continue
        
        for dataset_a, dataset_b in valid_pairs:
            dir_a = base_dir / dataset_a
            dir_b = base_dir / dataset_b
            
            files_a = sorted(list(dir_a.glob(file_extension)))
            if not files_a:
                continue
            
            print(f"\nProcessing {dataset_a} x {dataset_b}")
            print(f"Found {len(files_a)} {file_extension} files in {dataset_a}")
            
            for file_a in files_a:
                file_b = dir_b / file_a.name
                if not file_b.exists():
                    skipped_files += 1
                    continue
                
                try:
                    map_a = load_image(file_a)
                    map_b = load_image(file_b)
                    if map_a is None or map_b is None:
                        skipped_files += 1
                        continue
                    
                    map_a = np.nan_to_num(map_a, nan=np.nan)
                    map_b = np.nan_to_num(map_b, nan=np.nan)
                    map_a_flat = map_a.flatten()
                    map_b_flat = map_b.flatten()
                    
                    processed_files += 1
                    stats = calculate_strict_stats(map_a, map_b)
                    
                    # Calculate Q3 masks
                    mask_a_q3, q3_a = calculate_q3_mask(map_a)
                    mask_b_q3, q3_b = calculate_q3_mask(map_b)
                    
                    # Validate Q3 masks for SSIM
                    min_pixels = 100
                    valid_q3_a = mask_a_q3 is not None and mask_a_q3.sum() >= min_pixels
                    valid_q3_b = mask_b_q3 is not None and mask_b_q3.sum() >= min_pixels
                    
                    if swap_ytrue_ypred and metric_type in ['r2', 'residual', 'max_residual', 'min_residual']:
                        y_true = map_b_flat
                        y_pred = map_a_flat
                        y_true_2d = map_b
                        y_pred_2d = map_a
                    else:
                        y_true = map_a_flat
                        y_pred = map_b_flat
                        y_true_2d = map_a
                        y_pred_2d = map_b
                    
                    # Original metric
                    if metric_type == 'pearson':
                        metric_value = calculate_pearson(y_true, y_pred, file_a.name)
                    elif metric_type == 'r2':
                        metric_value, pearson_r = calculate_r2_score(y_true, y_pred, file_a.name)
                    elif metric_type == 'rmse':
                        metric_value = calculate_rmse(y_true, y_pred)
                    elif metric_type == 'mse':
                        metric_value = calculate_mse(y_true, y_pred)
                    elif metric_type == 'mae':
                        metric_value = calculate_mae(y_true, y_pred)
                    elif metric_type == 'residual':
                        metric_value = calculate_residual_error(y_true, y_pred, normalize=normalize_residuals, filename=file_a.name)
                    elif metric_type == 'max_residual':
                        metric_value = calculate_max_residual_error(y_true, y_pred, normalize=normalize_residuals, filename=file_a.name)
                    elif metric_type == 'min_residual':
                        metric_value = calculate_min_residual_error(y_true, y_pred, min_residual_percentile, normalize=normalize_residuals, filename=file_a.name)
                    elif metric_type == 'cosine':
                        metric_value = calculate_cosine_similarity(y_true, y_pred)
                    elif metric_type == 'huber':
                        metric_value = calculate_huber_loss(y_true, y_pred, huber_delta)
                    elif metric_type == 'ssim':
                        metric_value = calculate_ssim(y_true_2d, y_pred_2d)
                    
                    # Q3-based metrics
                    metric_q3_a = np.nan
                    metric_q3_b = np.nan
                    if valid_q3_a and metric_type == 'ssim':
                        # Check valid pixels after masking
                        y_true_masked = np.where(mask_a_q3, y_true_2d, np.nan)
                        y_pred_masked = np.where(mask_a_q3, y_pred_2d, np.nan)
                        valid_mask = ~np.isnan(y_true_masked) & ~np.isnan(y_pred_masked)
                        if valid_mask.sum() >= min_pixels:
                            metric_q3_a = calculate_ssim(y_true_2d, y_pred_2d, pixel_mask=mask_a_q3)
                    elif valid_q3_a:
                        if metric_type == 'pearson':
                            metric_q3_a = calculate_pearson(y_true, y_pred, file_a.name, pixel_mask=mask_a_q3.flatten())
                        elif metric_type == 'r2':
                            metric_q3_a, _ = calculate_r2_score(y_true, y_pred, file_a.name, pixel_mask=mask_a_q3.flatten())
                        elif metric_type == 'rmse':
                            metric_q3_a = calculate_rmse(y_true, y_pred, pixel_mask=mask_a_q3.flatten())
                        elif metric_type == 'mse':
                            metric_q3_a = calculate_mse(y_true, y_pred, pixel_mask=mask_a_q3.flatten())
                        elif metric_type == 'mae':
                            metric_q3_a = calculate_mae(y_true, y_pred, pixel_mask=mask_a_q3.flatten())
                        elif metric_type == 'residual':
                            metric_q3_a = calculate_residual_error(y_true, y_pred, normalize=normalize_residuals, filename=file_a.name, pixel_mask=mask_a_q3.flatten())
                        elif metric_type == 'max_residual':
                            metric_q3_a = calculate_max_residual_error(y_true, y_pred, normalize=normalize_residuals, filename=file_a.name, pixel_mask=mask_a_q3.flatten())
                        elif metric_type == 'min_residual':
                            metric_q3_a = calculate_min_residual_error(y_true, y_pred, min_residual_percentile, normalize=normalize_residuals, filename=file_a.name, pixel_mask=mask_a_q3.flatten())
                        elif metric_type == 'cosine':
                            metric_q3_a = calculate_cosine_similarity(y_true, y_pred, pixel_mask=mask_a_q3.flatten())
                        elif metric_type == 'huber':
                            metric_q3_a = calculate_huber_loss(y_true, y_pred, huber_delta, pixel_mask=mask_a_q3.flatten())
                    
                    if valid_q3_b and metric_type == 'ssim':
                        # Check valid pixels after masking
                        y_true_masked = np.where(mask_b_q3, y_true_2d, np.nan)
                        y_pred_masked = np.where(mask_b_q3, y_pred_2d, np.nan)
                        valid_mask = ~np.isnan(y_true_masked) & ~np.isnan(y_pred_masked)
                        if valid_mask.sum() >= min_pixels:
                            metric_q3_b = calculate_ssim(y_true_2d, y_pred_2d, pixel_mask=mask_b_q3)
                    elif valid_q3_b:
                        if metric_type == 'pearson':
                            metric_q3_b = calculate_pearson(y_true, y_pred, file_a.name, pixel_mask=mask_b_q3.flatten())
                        elif metric_type == 'r2':
                            metric_q3_b, _ = calculate_r2_score(y_true, y_pred, file_a.name, pixel_mask=mask_b_q3.flatten())
                        elif metric_type == 'rmse':
                            metric_q3_b = calculate_rmse(y_true, y_pred, pixel_mask=mask_b_q3.flatten())
                        elif metric_type == 'mse':
                            metric_q3_b = calculate_mse(y_true, y_pred, pixel_mask=mask_b_q3.flatten())
                        elif metric_type == 'mae':
                            metric_q3_b = calculate_mae(y_true, y_pred, pixel_mask=mask_b_q3.flatten())
                        elif metric_type == 'residual':
                            metric_q3_b = calculate_residual_error(y_true, y_pred, normalize=normalize_residuals, filename=file_a.name, pixel_mask=mask_b_q3.flatten())
                        elif metric_type == 'max_residual':
                            metric_q3_b = calculate_max_residual_error(y_true, y_pred, normalize=normalize_residuals, filename=file_a.name, pixel_mask=mask_b_q3.flatten())
                        elif metric_type == 'min_residual':
                            metric_q3_b = calculate_min_residual_error(y_true, y_pred, min_residual_percentile, normalize=normalize_residuals, filename=file_a.name, pixel_mask=mask_b_q3.flatten())
                        elif metric_type == 'cosine':
                            metric_q3_b = calculate_cosine_similarity(y_true, y_pred, pixel_mask=mask_b_q3.flatten())
                        elif metric_type == 'huber':
                            metric_q3_b = calculate_huber_loss(y_true, y_pred, huber_delta, pixel_mask=mask_b_q3.flatten())
                    
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
                    
                    if metric_type in ['rmse', 'mae', 'residual', 'max_residual', 'min_residual', 'huber'] and not np.isnan(metric_value) and stats['data_range'] != 0:
                        value_p = (metric_value / stats['data_range'] * 100)
                    elif metric_type == 'mse' and not np.isnan(metric_value) and stats['data_range'] != 0:
                        value_p = (metric_value / (stats['data_range'] ** 2) * 100)
                    else:
                        value_p = np.nan
                    
                    result_data = {
                        'datetime': epoch,
                        'comparison': f'{source_a} x {source_b}',
                        'dataset_a': dataset_a,
                        'dataset_b': dataset_b,
                        'source_a': source_a,
                        'source_b': source_b,
                        'filename_a': file_a.name,
                        'filename_b': file_b.name if metric_type == 'ssim' else file_a.name,
                        metric_type: metric_value,
                        f'{metric_type}_p': value_p,
                        f'{metric_type}_q3_a': metric_q3_a,
                        f'{metric_type}_q3_b': metric_q3_b,
                        'q3_a': q3_a,
                        'q3_b': q3_b,
                        **stats
                    }
                    if metric_type == 'r2':
                        result_data['pearson_r'] = pearson_r
                    result.append(result_data)
                    if processed_files % 10 == 0:
                        print(f"Processed {processed_files} file pairs, Skipped {skipped_files} files...")
                except Exception:
                    skipped_files += 1
    
    if not result:
        print(f"\nERROR: No valid data pairs found for analysis. Please check dataset directories and {file_extension} files.")
        exit(1)
    
    print(f"\nProcessed {processed_files} file pairs successfully, Skipped {skipped_files} files.")
    
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
        source_a, source_b = comparison
        for i in range(min(len(datasets[source_a]), len(datasets[source_b]))):
            dataset_a = datasets[source_a][i]
            dataset_b = datasets[source_b][i]
            comparison_type = f'{source_a} x {source_b}'
            selection = df.loc[(df['comparison'] == comparison_type) & 
                             (df['dataset_a'] == dataset_a) & 
                             (df['dataset_b'] == dataset_b)]
            
            if selection.empty:
                continue
                
            print(f"\n{source_a.upper()} x {source_b.upper()}")
            print(f"{dataset_a} x {dataset_b}")
            print(f"Number of comparisons: {len(selection)}")
            
            if not verify_combined_stats(selection):
                print(f"Inconsistências encontradas em {dataset_a} x {dataset_b}")
            
            metric_values = selection[metric_type].values
            valid_metrics = metric_values[~np.isnan(metric_values)]
            
            if len(valid_metrics) > 0:
                if metric_type == 'pearson':
                    z_values = [fisher_z_transform(v) for v in valid_metrics if not np.isnan(fisher_z_transform(v))]
                    metric_value = fisher_z_inverse(np.mean(z_values)) if z_values else np.nan
                elif metric_type == 'r2':
                    r_values = selection['pearson_r'].values
                    valid_r = r_values[~np.isnan(r_values)]
                    z_values = [fisher_z_transform(r) for r in valid_r if not np.isnan(fisher_z_transform(r))]
                    metric_value = fisher_z_inverse(np.mean(z_values)) ** 2 if z_values else np.nan
                else:
                    metric_value = np.nanmean(np.abs(valid_metrics)) if metric_type == 'residual' else np.nanmean(valid_metrics)
            else:
                metric_value = np.nan
            
            # Q3-based metric averages
            metric_q3_a_values = selection[f'{metric_type}_q3_a'].values
            metric_q3_b_values = selection[f'{metric_type}_q3_b'].values
            valid_q3_a = metric_q3_a_values[~np.isnan(metric_q3_a_values)]
            valid_q3_b = metric_q3_b_values[~np.isnan(metric_q3_b_values)]
            metric_q3_a_avg = np.nanmean(valid_q3_a) if len(valid_q3_a) > 0 else np.nan
            metric_q3_b_avg = np.nanmean(valid_q3_b) if len(valid_q3_b) > 0 else np.nan
            
            dataset_metrics[source_a].extend(valid_metrics)
            dataset_metrics[source_b].extend(valid_metrics)
            
            dataset_total_metrics[source_a] += np.nansum(metric_values)
            dataset_total_metrics[source_b] += np.nansum(metric_values)
            dataset_count[source_a] += len(valid_metrics)
            dataset_count[source_b] += len(valid_metrics)
            
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

            # Exibindo estatísticas gerais
            print("\nStatistics for All Data:")
            print("Dataset A:")
            print(f"  Min: {mean_min_a:.4f}, Q1: {selection['q1_a'].mean():.4f}, Median: {mean_median_a:.4f}, Q3: {selection['q3_a'].mean():.4f}, Mean: {mean_mean_a:.4f}, Max: {mean_max_a:.4f}")
            print("Dataset B:")
            print(f"  Min: {mean_min_b:.4f}, Q1: {selection['q1_b'].mean():.4f}, Median: {mean_median_b:.4f}, Q3: {selection['q3_b'].mean():.4f}, Mean: {mean_mean_b:.4f}, Max: {mean_max_b:.4f}")
            print("Combined:")
            print(f"  Min: {mean_min_both:.4f}, Q1: {selection['q1_both'].mean():.4f}, Median: {mean_median_both:.4f}, Q3: {selection['q3_both'].mean():.4f}, Mean: {mean_mean_both:.4f}, Max: {mean_max_both:.4f}")
            print(f"  Data Range: {mean_data_range:.4f}")
            
            # Função auxiliar para formatar porcentagens
            def format_percentage(value, data_range=None, is_normalized=False):
                if np.isnan(value):
                    return "NaN%"
                if is_normalized:  # Para métricas já normalizadas (pearson, r2, etc.)
                    return f"{value * 100:.2f}%"
                if data_range and data_range != 0:  # Para métricas de erro (rmse, mae, etc.)
                    return f"{(value / data_range * 100):.2f}% of data range"
                return "NaN%"  # Caso nenhuma condição seja satisfeita
            
            # Função para formatar a exibição da métrica com sua porcentagem
            def format_metric_with_percent(value, data_range=None, is_normalized=False, suffix=""):
                percent = format_percentage(value, data_range, is_normalized)
                return f"{value:.4f} ({percent}){suffix}"
            
            if metric_type == 'pearson':
                print(f'Average Pearson Correlation: {format_metric_with_percent(metric_value, is_normalized=True, suffix=" (Fisher Z applied)")}')
                print(f'Average Pearson (Q3 Map A): {format_metric_with_percent(metric_q3_a_avg, is_normalized=True)}')
                print(f'Average Pearson (Q3 Map B): {format_metric_with_percent(metric_q3_b_avg, is_normalized=True)}')
            elif metric_type == 'r2':
                print(f'Average R² Score: {format_metric_with_percent(metric_value, is_normalized=True, suffix=" (Fisher Z applied on Pearson r)")}')
                print(f'Average R² (Q3 Map A): {format_metric_with_percent(metric_q3_a_avg, is_normalized=True)}')
                print(f'Average R² (Q3 Map B): {format_metric_with_percent(metric_q3_b_avg, is_normalized=True)}')
            elif metric_type == 'ssim':
                print(f'Average Structural Similarity Index: {format_metric_with_percent(metric_value, is_normalized=True)}')
                print(f'Average SSIM (Q3 Map A): {format_metric_with_percent(metric_q3_a_avg, is_normalized=True)}')
                print(f'Average Structural Similarity Index: {format_metric_with_percent(metric_value, is_normalized=True)}')
                print(f'Average SSIM (Q3 Map A): {format_metric_with_percent(metric_q3_a_avg, is_normalized=True)}')
                print(f'Average SSIM (Q3 Map B): {format_metric_with_percent(metric_q3_b_avg, is_normalized=True)}')
            elif metric_type == 'cosine':
                print(f'Average Cosine Similarity: {format_metric_with_percent(metric_value, is_normalized=True)}')
                print(f'Average Cosine (Q3 Map A): {format_metric_with_percent(metric_q3_a_avg, is_normalized=True)}')
                print(f'Average Cosine (Q3 Map B): {format_metric_with_percent(metric_q3_b_avg, is_normalized=True)}')
            elif metric_type == 'rmse':
                print(f'Average RMSE: {format_metric_with_percent(metric_value, mean_data_range)}')
                print(f'Average RMSE (Q3 Map A): {format_metric_with_percent(metric_q3_a_avg, mean_data_range)}')
                print(f'Average RMSE (Q3 Map B): {format_metric_with_percent(metric_q3_b_avg, mean_data_range)}')
            elif metric_type == 'mse':
                print(f'Average Mean Squared Error: {format_metric_with_percent(metric_value, mean_data_range)}')
                print(f'Average MSE (Q3 Map A): {format_metric_with_percent(metric_q3_a_avg, mean_data_range)}')
                print(f'Average MSE (Q3 Map B): {format_metric_with_percent(metric_q3_b_avg, mean_data_range)}')
            elif metric_type == 'mae':
                print(f'Average Mean Absolute Error: {format_metric_with_percent(metric_value, mean_data_range)}')
                print(f'Average MAE (Q3 Map A): {format_metric_with_percent(metric_q3_a_avg, mean_data_range)}')
                print(f'Average MAE (Q3 Map B): {format_metric_with_percent(metric_q3_b_avg, mean_data_range)}')
            elif metric_type == 'residual':
                print(f'Average Mean Absolute Residual Error: {format_metric_with_percent(metric_value, mean_data_range)}')
                print(f'Average Residual (Q3 Map A): {format_metric_with_percent(metric_q3_a_avg, mean_data_range)}')
                print(f'Average Residual (Q3 Map B): {format_metric_with_percent(metric_q3_b_avg, mean_data_range)}')
            elif metric_type == 'max_residual':
                print(f'Average Maximum Residual Error: {format_metric_with_percent(metric_value, mean_data_range)}')
                print(f'Average Max Residual (Q3 Map A): {format_metric_with_percent(metric_q3_a_avg, mean_data_range)}')
                print(f'Average Max Residual (Q3 Map B): {format_metric_with_percent(metric_q3_b_avg, mean_data_range)}')
            elif metric_type == 'min_residual':
                suffix = f" ({min_residual_percentile}th percentile)"
                print(f'Average Minimum Residual Error{suffix}: {format_metric_with_percent(metric_value, mean_data_range)}')
                print(f'Average Min Residual (Q3 Map A): {format_metric_with_percent(metric_q3_a_avg, mean_data_range)}')
                print(f'Average Min Residual (Q3 Map B): {format_metric_with_percent(metric_q3_b_avg, mean_data_range)}')
            elif metric_type == 'huber':
                suffix = f" (delta={huber_delta})"
                print(f'Average Huber Loss{suffix}: {format_metric_with_percent(metric_value, mean_data_range)}')
                print(f'Average Huber Loss (Q3 Map A): {format_metric_with_percent(metric_q3_a_avg, mean_data_range)}')
                print(f'Average Huber Loss (Q3 Map B): {format_metric_with_percent(metric_q3_b_avg, mean_data_range)}')
            
            sorted_maps = selection.sort_values(by=f'{metric_type}_p', ascending=not higher_is_better).head(top_n)
            comp_key = f"{dataset_a} x {dataset_b}"
            top_maps_by_comparison[comp_key] = sorted_maps
            
            # Agora todos são ordenados por percentual
            print(f"\nTop {top_n} Maps with Best {metric_type.upper()} Values (Sorted by Percentage):")
            print("-" * 120)
            for idx, row in enumerate(sorted_maps.itertuples(), 1):
                file_info = f"{row.filename_a} & {row.filename_b}" if metric_type == 'ssim' else row.filename_a
                
                # Verificar se é uma métrica normalizada
                is_normalized = metric_type in ['pearson', 'r2', 'cosine', 'ssim']
                
                # Formatar a métrica principal com porcentagem
                if hasattr(row, f'{metric_type}_p') and not np.isnan(getattr(row, f'{metric_type}_p')):
                    if is_normalized:
                        metric_display = f"{getattr(row, metric_type):.4f} ({getattr(row, f'{metric_type}_p'):.2f}%)"
                    else:
                        metric_display = f"{getattr(row, metric_type):.4f} ({getattr(row, f'{metric_type}_p'):.2f}% of data range)"
                else:
                    # Se não tiver o atributo de porcentagem, calcular
                    metric_value = getattr(row, metric_type)
                    if is_normalized:
                        percent = metric_value * 100 if not np.isnan(metric_value) else np.nan
                        metric_display = f"{metric_value:.4f} ({percent:.2f}%)" if not np.isnan(percent) else f"{metric_value:.4f} (NaN%)"
                    else:
                        percent = (metric_value / row.data_range * 100) if not np.isnan(metric_value) and row.data_range != 0 else np.nan
                        metric_display = f"{metric_value:.4f} ({percent:.2f}% of data range)" if not np.isnan(percent) else f"{metric_value:.4f} (NaN% of data range)"
                
                # Formatar os valores Q3 com porcentagem
                q3_a_value = getattr(row, f'{metric_type}_q3_a')
                q3_b_value = getattr(row, f'{metric_type}_q3_b')
                
                if is_normalized:
                    q3_a_percent = q3_a_value * 100 if not np.isnan(q3_a_value) else np.nan
                    q3_b_percent = q3_b_value * 100 if not np.isnan(q3_b_value) else np.nan
                    q3_a_display = f"{q3_a_value:.4f} ({q3_a_percent:.2f}%)" if not np.isnan(q3_a_value) else 'NaN'
                    q3_b_display = f"{q3_b_value:.4f} ({q3_b_percent:.2f}%)" if not np.isnan(q3_b_value) else 'NaN'
                else:
                    q3_a_percent = (q3_a_value / row.data_range * 100) if not np.isnan(q3_a_value) and row.data_range != 0 else np.nan
                    q3_b_percent = (q3_b_value / row.data_range * 100) if not np.isnan(q3_b_value) and row.data_range != 0 else np.nan
                    q3_a_display = f"{q3_a_value:.4f} ({q3_a_percent:.2f}% of data range)" if not np.isnan(q3_a_value) else 'NaN'
                    q3_b_display = f"{q3_b_value:.4f} ({q3_b_percent:.2f}% of data range)" if not np.isnan(q3_b_value) else 'NaN'
                
                date_str = pd.to_datetime(row.datetime).strftime('%Y-%m-%d %H:%M') if hasattr(row, 'datetime') else 'Unknown'
                
                # Exibindo informações do mapa
                print(f"{idx}. Data: {date_str}")
                print(f"   Comparação: {row.dataset_a} x {row.dataset_b}")
                print(f"   Arquivos: {file_info}")
                print(f"   {metric_type.upper()}: {metric_display}")
                print(f"   {metric_type.upper()} (Q3 Map A, >= {row.q3_a:.4f}): {q3_a_display}")
                print(f"   {metric_type.upper()} (Q3 Map B, >= {row.q3_b:.4f}): {q3_b_display}")
                print(f"   Estatísticas do Dataset A:")
                print(f"     Min: {row.min_a:.4f}, Q1: {row.q1_a:.4f}, Median: {row.median_a:.4f}, Q3: {row.q3_a:.4f}, Mean: {row.mean_a:.4f}, Max: {row.max_a:.4f}")
                print(f"   Estatísticas do Dataset B:")
                print(f"     Min: {row.min_b:.4f}, Q1: {row.q1_b:.4f}, Median: {row.median_b:.4f}, Q3: {row.q3_b:.4f}, Mean: {row.mean_b:.4f}, Max: {row.max_b:.4f}")
                print(f"   Estatísticas Combinadas:")
                print(f"     Min: {row.min_both:.4f}, Q1: {row.q1_both:.4f}, Median: {row.median_both:.4f}, Q3: {row.q3_both:.4f}, Mean: {row.mean_both:.4f}, Max: {row.max_both:.4f}")
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
                            if metric_type == 'pearson':
                                month_metric = np.nanmean(month_data[metric_type])
                            elif metric_type == 'r2':
                                r_values = month_data['pearson_r'].values
                                valid_r = r_values[~np.isnan(r_values)]
                                z_values = [fisher_z_transform(r) for r in valid_r if not np.isnan(fisher_z_transform(r))]
                                month_metric = fisher_z_inverse(np.mean(z_values)) ** 2 if z_values else np.nan
                            else:
                                month_metric = np.nanmean(np.abs(month_data[metric_type])) if metric_type == 'residual' else np.nanmean(month_data[metric_type])
                            
                            # Exibindo métrica mensal com porcentagem
                            if metric_type in ['pearson', 'r2', 'cosine', 'ssim']:
                                # Métricas já normalizadas
                                month_percent = month_metric * 100 if not np.isnan(month_metric) else np.nan
                                percent_suffix = "%"
                            elif metric_type in ['rmse', 'mse', 'mae', 'residual', 'max_residual', 'min_residual', 'huber']:
                                # Métricas baseadas em erro
                                month_percent = (month_metric / mean_data_range * 100) if not np.isnan(month_metric) and mean_data_range != 0 else np.nan
                                percent_suffix = "% of data range"
                            else:
                                month_percent = np.nan
                                percent_suffix = "%"

                            percent_display = f"({month_percent:.2f}{percent_suffix})" if not np.isnan(month_percent) else "(NaN%)"
                            print(f'{month_name}: {metric_type.upper()} = {month_metric:.4f} {percent_display}')
    
    # Função para formatar a exibição de métrica com porcentagem
    def format_dataset_metric(dataset_name, metric_val):
        if metric_type in ['pearson', 'r2', 'cosine', 'ssim']:
            percent = metric_val * 100 if not np.isnan(metric_val) else np.nan
            percent_suffix = "%"
        else:
            # Uso da média dos data_range para este dataset
            data_ranges = df[(df['source_a'] == dataset_name) | (df['source_b'] == dataset_name)]['data_range'].values
            avg_data_range = np.nanmean(data_ranges) if len(data_ranges) > 0 else 1.0
            percent = (metric_val / avg_data_range * 100) if not np.isnan(metric_val) and avg_data_range != 0 else np.nan
            percent_suffix = "% of data range"
        
        percent_display = f"({percent:.2f}{percent_suffix})" if not np.isnan(percent) else "(NaN%)"
        return f"{dataset_name} with average {metric_type} of {metric_val:.4f} {percent_display}"
    
    dataset_avg_metrics = {}
    
    print("\nCalculating average metrics...")
    for dataset in dataset_metrics:
        valid_metrics = [v for v in dataset_metrics[dataset] if not np.isnan(v)]
        if valid_metrics:
            if metric_type == 'pearson':
                z_values = [fisher_z_transform(v) for v in valid_metrics if not np.isnan(fisher_z_transform(v))]
                dataset_avg_metrics[dataset] = fisher_z_inverse(np.mean(z_values)) if z_values else np.nan
            elif metric_type == 'r2':
                r_values = df[(df['source_a'] == dataset) | (df['source_b'] == dataset)]['pearson_r'].values
                valid_r = r_values[~np.isnan(r_values)]
                z_values = [fisher_z_transform(r) for r in valid_r if not np.isnan(fisher_z_transform(r))]
                dataset_avg_metrics[dataset] = fisher_z_inverse(np.mean(z_values)) ** 2 if z_values else np.nan
            else:
                dataset_avg_metrics[dataset] = np.nanmean(valid_metrics)
        else:
            dataset_avg_metrics[dataset] = np.nan
    
    best_dataset = max(dataset_avg_metrics.items(), key=lambda x: x[1] if not np.isnan(x[1]) else float('-inf')) if higher_is_better else \
                   min(dataset_avg_metrics.items(), key=lambda x: x[1] if not np.isnan(x[1]) else float('inf'))
    worst_dataset = min(dataset_avg_metrics.items(), key=lambda x: x[1] if not np.isnan(x[1]) else float('inf')) if higher_is_better else \
                    max(dataset_avg_metrics.items(), key=lambda x: x[1] if not np.isnan(x[1]) else float('inf'))
    
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
    print(f"Best dataset: {format_dataset_metric(best_dataset[0], best_dataset[1])}")
    print(f"Worst dataset: {format_dataset_metric(worst_dataset[0], worst_dataset[1])}")
    
    print("\nDataset Ranking (from best to worst):")
    sorted_datasets = sorted(dataset_avg_metrics.items(), key=lambda x: x[1] if not np.isnan(x[1]) else float('-inf'), reverse=higher_is_better)
    
    for i, (dataset, avg_val) in enumerate(sorted_datasets, 1):
        if metric_type in ['pearson', 'r2', 'cosine', 'ssim']:
            percent = avg_val * 100 if not np.isnan(avg_val) else np.nan
            percent_suffix = "%"
        else:
            # Aproximação usando os valores médios dos data_range
            data_ranges = df[(df['source_a'] == dataset) | (df['source_b'] == dataset)]['data_range'].tolist()
            avg_data_range = np.nanmean(data_ranges) if data_ranges else 1.0
            percent = (avg_val / avg_data_range * 100) if not np.isnan(avg_val) and avg_data_range != 0 else np.nan
            percent_suffix = "% of data range"

        percent_display = f"({percent:.2f}{percent_suffix})" if not np.isnan(percent) else "(NaN%)"
        print(f"{i}. {dataset}: {avg_val:.4f} {percent_display}")
    
    overall_top_count = 50

print(f"\n===== TOP {overall_top_count} MAPS OVERALL (Sorted by Percentage) =====")
top_overall = df.sort_values(by=f'{metric_type}_p', ascending=not higher_is_better).head(overall_top_count)
print("-" * 120)

for idx, row in enumerate(top_overall.itertuples(), 1):
    file_info = f"{row.filename_a} & {row.filename_b}" if metric_type == 'ssim' else row.filename_a
    
    # Exibição consistente de porcentagem para todas as métricas
    if hasattr(row, f'{metric_type}_p') and not np.isnan(getattr(row, f'{metric_type}_p')):
        # Se a métrica já tem um atributo de porcentagem calculado
        if metric_type in ['rmse', 'mse', 'mae', 'residual', 'max_residual', 'min_residual', 'huber']:
            metric_display = f"{getattr(row, metric_type):.4f} ({getattr(row, f'{metric_type}_p'):.2f}% of data range)"
        else:
            metric_display = f"{getattr(row, metric_type):.4f} ({getattr(row, f'{metric_type}_p'):.2f}%)"
    else:
        # Se precisamos calcular a porcentagem agora
        metric_value = getattr(row, metric_type)
        if metric_type in ['pearson', 'r2', 'cosine', 'ssim']:
            # Métricas normalizadas
            percent = metric_value * 100 if not np.isnan(metric_value) else np.nan
            percent_suffix = "%"
        else:
            # Métricas baseadas em erro
            percent = (metric_value / row.data_range * 100) if not np.isnan(metric_value) and row.data_range != 0 else np.nan
            percent_suffix = "% of data range"
        
        metric_display = f"{metric_value:.4f} ({percent:.2f}{percent_suffix})" if not np.isnan(percent) else f"{metric_value:.4f} (NaN%)"
    
    date_str = pd.to_datetime(row.datetime).strftime('%Y-%m-%d %H:%M') if hasattr(row, 'datetime') else 'Unknown'
    
    print(f"{idx}. Data: {date_str}")
    print(f"   Comparação: {row.dataset_a} x {row.dataset_b}")
    print(f"   Arquivos: {file_info}")
    print(f"   {metric_type.upper()}: {metric_display}")
    print(f"   {metric_type.upper()} (Q3 Map A, >= {row.q3_a:.4f}): {q3_a_display}")
    print(f"   {metric_type.upper()} (Q3 Map B, >= {row.q3_b:.4f}): {q3_b_display}")
    print(f"   Estatísticas do Dataset A:")
    print(f"     Min: {row.min_a:.4f}, Q1: {row.q1_a:.4f}, Median: {row.median_a:.4f}, Q3: {row.q3_a:.4f}, Mean: {row.mean_a:.4f}, Max: {row.max_a:.4f}")
    print(f"   Estatísticas do Dataset B:")
    print(f"     Min: {row.min_b:.4f}, Q1: {row.q1_b:.4f}, Median: {row.median_b:.4f}, Q3: {row.q3_b:.4f}, Mean: {row.mean_b:.4f}, Max: {row.max_b:.4f}")
    print(f"   Estatísticas Combinadas:")
    print(f"     Min: {row.min_both:.4f}, Q1: {row.q1_both:.4f}, Median: {row.median_both:.4f}, Q3: {row.q3_both:.4f}, Mean: {row.mean_both:.4f}, Max: {row.max_both:.4f}")
    print(f"     Data Range: {row.data_range:.4f}")
    if idx < len(top_overall):
        print("-" * 80)
print("-" * 120)

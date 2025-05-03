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

def calculate_q3_mask(map_data, verbose=False):
    """Calcula Q3 (75º percentil) e retorna máscara para valores >= Q3."""
    if verbose:
        print(f"Calculating Q3 mask for data shape {map_data.shape}")
    
    # Create mask of non-NaN values
    non_nan_mask = ~np.isnan(map_data)
    non_nan_count = np.sum(non_nan_mask)
    
    if verbose:
        print(f"Non-NaN values: {non_nan_count}/{map_data.size} ({non_nan_count/map_data.size*100:.2f}%)")
    
    # Get valid (non-NaN) values for percentile calculation
    valid_values = map_data[non_nan_mask]
    
    if len(valid_values) == 0:
        if verbose:
            print("No valid values found for Q3 calculation")
        return None, np.nan
        
    # Calculate Q3 from valid values
    q3 = np.percentile(valid_values, 75)
    if verbose:
        print(f"Q3 value: {q3:.4f}")
    
    # Special handling for cases where Q3 equals max value
    unique_values = np.unique(valid_values)
    if q3 == np.max(valid_values) and len(unique_values) > 1:
        if verbose:
            print("Q3 equals max value, using second highest value instead")
        q3 = unique_values[-2]  # Use second highest value
    
    # Create a mask where values are both non-NaN AND >= Q3
    mask = non_nan_mask & (map_data >= q3)
    mask_count = np.sum(mask)
    
    if verbose:
        mask_percent = mask_count / map_data.size * 100
        print(f"Values in Q3 mask: {mask_count} ({mask_percent:.2f}%)")
    
    # Ensure we have enough values (at least 5% of original valid values)
    min_required = max(100, len(valid_values) * 0.05)
    if mask_count < min_required:
        if verbose:
            print(f"Not enough values in Q3 mask, trying more lenient threshold (min: {min_required})")
        
        # Try a more lenient threshold
        q3_adjusted = np.percentile(valid_values, 65)  # Try 65th percentile instead
        if verbose:
            print(f"Adjusted to 65th percentile: {q3_adjusted:.4f}")
        
        mask = non_nan_mask & (map_data >= q3_adjusted)
        mask_count = np.sum(mask)
        
        if verbose:
            print(f"Values in adjusted Q3 mask: {mask_count} ({mask_count/map_data.size*100:.2f}%)")
        
        if mask_count >= min_required:
            return mask, q3_adjusted
        
        if verbose:
            print("Still not enough values with adjusted threshold")
        return None, q3
    
    return mask, q3

def calculate_pearson(y_true, y_pred, filename="unknown", value_mask=None):
    """Calcula a correlação de Pearson, tratando casos especiais."""
    if value_mask is not None:
        if value_mask.sum() < 2:
            return np.nan
        y_true = y_true[value_mask]
        y_pred = y_pred[value_mask]
    
    if len(y_true) == 0 or len(y_pred) == 0 or np.all(np.isnan(y_true)) or np.all(np.isnan(y_pred)):
        return np.nan
    
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if np.sum(valid_mask) < 2:
        return np.nan
    
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    
    # Verificar variância zero
    var_true = np.var(y_true_valid)
    var_pred = np.var(y_pred_valid)
    
    if var_true < 1e-10 or var_pred < 1e-10:
        return 0.0  # Retorna 0 para casos de variância zero em vez de NaN
    
    try:
        corr = np.corrcoef(y_true_valid, y_pred_valid)[0, 1]
        
        # Verificar se o resultado é válido
        if np.isnan(corr) or np.isinf(corr):
            return np.nan
            
        return corr
    except Exception as e:
        return np.nan

def calculate_r2_score(y_true, y_pred, filename="unknown", value_mask=None):
    """Calcula o R² como o quadrado da correlação de Pearson."""
    if value_mask is not None:
        if value_mask.sum() < 2:
            return np.nan, np.nan
        y_true = y_true[value_mask]
        y_pred = y_pred[value_mask]
    pearson_r = calculate_pearson(y_true, y_pred, filename)
    if np.isnan(pearson_r):
        return np.nan, np.nan
    return pearson_r ** 2, pearson_r

def calculate_rmse(y_true, y_pred, value_mask=None):
    """Calcula o RMSE, tratando casos especiais."""
    if value_mask is not None:
        if value_mask.sum() < 2:
            return np.nan
        y_true = y_true[value_mask]
        y_pred = y_pred[value_mask]
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if np.sum(valid_mask) < 2:
        return np.nan
    return np.sqrt(mean_squared_error(y_true[valid_mask], y_pred[valid_mask]))

def calculate_mse(y_true, y_pred, value_mask=None):
    """Calcula o MSE, tratando casos especiais."""
    if value_mask is not None:
        if value_mask.sum() < 2:
            return np.nan
        y_true = y_true[value_mask]
        y_pred = y_pred[value_mask]
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if np.sum(valid_mask) < 2:
        return np.nan
    return mean_squared_error(y_true[valid_mask], y_pred[valid_mask])

def calculate_mae(y_true, y_pred, value_mask=None):
    """Calcula o MAE, tratando casos especiais."""
    if value_mask is not None:
        if value_mask.sum() < 2:
            return np.nan
        y_true = y_true[value_mask]
        y_pred = y_pred[value_mask]
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if np.sum(valid_mask) < 2:
        return np.nan
    return mean_absolute_error(y_true[valid_mask], y_pred[valid_mask])

def calculate_residual_error(y_true, y_pred, normalize=False, filename="unknown", value_mask=None):
    """Calcula o erro residual médio absoluto com validação robusta."""
    if value_mask is not None:
        if value_mask.sum() < 2:
            return np.nan
        y_true = y_true[value_mask]
        y_pred = y_pred[value_mask]
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

def calculate_max_residual_error(y_true, y_pred, normalize=False, filename="unknown", value_mask=None):
    """Calcula o erro residual máximo com validação robusta."""
    if value_mask is not None:
        if value_mask.sum() < 2:
            return np.nan
        y_true = y_true[value_mask]
        y_pred = y_pred[value_mask]
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

def calculate_min_residual_error(y_true, y_pred, percentile=5.0, normalize=False, filename="unknown", value_mask=None):
    """Calcula o erro residual no percentil especificado com validação robusta."""
    if value_mask is not None:
        if value_mask.sum() < 2:
            return np.nan
        y_true = y_true[value_mask]
        y_pred = y_pred[value_mask]
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

def calculate_cosine_similarity(y_true, y_pred, value_mask=None):
    """Calcula a similaridade de cosseno, tratando casos especiais."""
    if value_mask is not None:
        if value_mask.sum() < 2:
            return np.nan
        y_true = y_true[value_mask]
        y_pred = y_pred[value_mask]
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

def calculate_huber_loss(y_true, y_pred, delta=1.0, value_mask=None):
    """Calcula a perda de Huber, tratando casos especiais."""
    if value_mask is not None:
        if value_mask.sum() < 2:
            return np.nan
        y_true = y_true[value_mask]
        y_pred = y_pred[value_mask]
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if np.sum(valid_mask) < 2:
        return np.nan
    errors = y_true[valid_mask] - y_pred[valid_mask]
    abs_errors = np.abs(errors)
    quadratic = np.minimum(abs_errors, delta)
    linear = abs_errors - quadratic
    return np.mean(0.5 * quadratic * quadratic + delta * linear)

def calculate_ssim(y_true, y_pred, verbose=False):
    """Calcula o SSIM apenas nas regiões onde ambas as imagens têm valores não-NaN."""
    # Converte para escala de cinza se for imagem colorida
    if len(y_true.shape) > 2 and y_true.shape[2] > 1:
        y_true = np.mean(y_true, axis=2)
    if len(y_pred.shape) > 2 and y_pred.shape[2] > 1:
        y_pred = np.mean(y_pred, axis=2)
    
    if verbose:
        print(f"\nSSIM calculation:")
        print(f"Input shapes: y_true={y_true.shape}, y_pred={y_pred.shape}")
        print(f"Data ranges: y_true=[{np.nanmin(y_true):.4f}, {np.nanmax(y_true):.4f}], "
              f"y_pred=[{np.nanmin(y_pred):.4f}, {np.nanmax(y_pred):.4f}]")
        print(f"NaN counts: y_true={np.isnan(y_true).sum()}, y_pred={np.isnan(y_pred).sum()}")
    
    # Cria uma máscara de valores válidos (não NaN)
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    valid_value_count = np.sum(valid_mask)
    valid_percentage = valid_value_count / valid_mask.size * 100
    
    if verbose:
        print(f"Valid values: {valid_value_count}/{valid_mask.size} ({valid_percentage:.2f}%)")
    
    # Verifica se há valores válidos suficientes
    min_values = 100  # Mínimo de valores para um SSIM significativo
    if valid_value_count < min_values:
        if verbose:
            print(f"SSIM failed: Too few valid values ({valid_value_count})")
        return np.nan
    
    # Extrair apenas os valores válidos para calcular estatísticas
    y_true_values = y_true[valid_mask]
    y_pred_values = y_pred[valid_mask]
    
    # Verificar se há variância suficiente para um cálculo significativo
    min_variance = 1e-6
    true_var = np.var(y_true_values)
    pred_var = np.var(y_pred_values)
    
    if verbose:
        print(f"Variance check: y_true_var={true_var:.6f}, y_pred_var={pred_var:.6f}, min={min_variance}")
    
    if true_var < min_variance or pred_var < min_variance:
        if verbose:
            print(f"Low variance detected in images")
        
        # Se ambos são praticamente constantes e iguais
        if np.allclose(y_true_values, y_pred_values, rtol=1e-5, atol=1e-8):
            if verbose:
                print("Images are constant and identical, returning 1.0")
            return 1.0
        
        # Se são constantes mas diferentes
        mean_abs_diff = np.mean(np.abs(y_true_values - y_pred_values))
        max_possible_diff = max(np.max(y_true_values), np.max(y_pred_values)) - min(np.min(y_true_values), np.min(y_pred_values))
        
        if max_possible_diff > 0:
            similarity = 1.0 - (mean_abs_diff / max_possible_diff)
            if verbose:
                print(f"Computed similarity for constant images: {similarity:.4f}")
            return similarity
            
        if verbose:
            print("Unable to compute meaningful similarity, returning 0.0")
        return 0.0
    
    # Para calcular o SSIM com a biblioteca, precisamos criar versões das imagens
    # onde regiões não válidas são substituídas por um valor constante que não
    # afeta o cálculo nas regiões válidas
    y_true_for_ssim = y_true.copy()
    y_pred_for_ssim = y_pred.copy()
    
    # Usar o mesmo valor constante para ambas as imagens em regiões não válidas
    # Isso faz com que o SSIM dessas regiões seja 1.0 (idêntico) e não afete o cálculo
    constant_value = np.mean(y_true_values)
    y_true_for_ssim[~valid_mask] = constant_value
    y_pred_for_ssim[~valid_mask] = constant_value
    
    # Calcula o intervalo de dados apenas com valores válidos
    data_range = max(np.max(y_true_values) - np.min(y_true_values), 
                     np.max(y_pred_values) - np.min(y_pred_values))
    if data_range == 0:
        data_range = 1.0
        if verbose:
            print("Zero data range detected, using default value of 1.0")
    
    # Tenta calcular SSIM
    try:
        # Se a maioria dos valores (>50%) não for válida, ainda calculamos, mas com alerta
        if valid_percentage < 50 and verbose:
            print(f"SSIM warning: Less than 50% valid values ({valid_percentage:.2f}%)")
        
        # Calcula o SSIM - as regiões não válidas não afetam o resultado
        # pois têm valores idênticos em ambas as imagens
        ssim_value = ssim(y_true_for_ssim, y_pred_for_ssim, data_range=data_range)
        
        # Verificar se o resultado é válido
        if np.isnan(ssim_value) or np.isinf(ssim_value):
            if verbose:
                print(f"SSIM calculation returned invalid value: {ssim_value}")
            return np.nan
            
        if verbose:
            print(f"SSIM calculation successful: {ssim_value:.4f}")
        return ssim_value
    except Exception as e:
        # Se houver erro por diferença de dimensões
        if y_true.shape != y_pred.shape:
            if verbose:
                print(f"Attempting SSIM with resized arrays")
            
            min_height = min(y_true.shape[0], y_pred.shape[0])
            min_width = min(y_true.shape[1], y_pred.shape[1])
            y_true_resized = y_true_for_ssim[:min_height, :min_width]
            y_pred_resized = y_pred_for_ssim[:min_height, :min_width]
            
            # Tenta novamente com as imagens redimensionadas
            try:
                result = ssim(y_true_resized, y_pred_resized, data_range=data_range)
                if verbose:
                    print(f"SSIM with resized arrays successful: {result:.4f}")
                return result
            except Exception as e2:
                if verbose:
                    print(f"SSIM with resized arrays failed: {str(e2)}")
                return np.nan
        
        if verbose:
            print(f"SSIM calculation failed: {str(e)}")
        return np.nan

def calculate_ssim_with_q3_mask(y_true_2d, y_pred_2d, mask_q3, verbose=False):
    """Calculate SSIM with Q3 mask applied."""
    if mask_q3 is None:
        if verbose:
            print("No Q3 mask provided for SSIM calculation")
        return np.nan
    
    # For SSIM with Q3 mask, we need to:
    # 1. Keep the spatial relationship (2D structure)
    # 2. Only consider values in the Q3 mask
    # 3. Replace non-Q3 values with a constant value
    
    # Create copies for modification
    y_true_q3 = y_true_2d.copy()
    y_pred_q3 = y_pred_2d.copy()
    
    # Get valid values in the Q3 mask
    valid_q3_mask = mask_q3 & ~np.isnan(y_true_2d) & ~np.isnan(y_pred_2d)
    valid_value_count = np.sum(valid_q3_mask)
    
    if verbose:
        valid_percent = valid_value_count / valid_q3_mask.size * 100
        print(f"Valid values in Q3 mask: {valid_value_count}/{valid_q3_mask.size} ({valid_percent:.2f}%)")
    
    # Check if we have enough values
    min_values = 100
    if valid_value_count < min_values:
        if verbose:
            print(f"Too few valid values in Q3 mask: {valid_value_count} < {min_values}")
        return np.nan
    
    # Calculate a constant value for non-Q3 regions
    # (we use mean of valid Q3 values to ensure the non-Q3 regions 
    # don't affect the SSIM calculation)
    y_true_q3_values = y_true_2d[valid_q3_mask]
    if len(y_true_q3_values) == 0:
        if verbose:
            print("No valid values in Q3 mask for SSIM calculation")
        return np.nan
    
    constant_value = np.mean(y_true_q3_values)
    
    # Replace non-Q3 values with constant value
    y_true_q3[~valid_q3_mask] = constant_value
    y_pred_q3[~valid_q3_mask] = constant_value
    
    # Calculate data range based only on Q3 values
    data_range = max(
        np.max(y_true_2d[valid_q3_mask]) - np.min(y_true_2d[valid_q3_mask]),
        np.max(y_pred_2d[valid_q3_mask]) - np.min(y_pred_2d[valid_q3_mask])
    )
    if data_range == 0:
        data_range = 1.0
        if verbose:
            print("Zero data range in Q3 values, using default value of 1.0")
    
    # Calculate SSIM on the modified images
    try:
        ssim_value = ssim(y_true_q3, y_pred_q3, data_range=data_range)
        if verbose:
            print(f"Q3-masked SSIM calculation: {ssim_value:.4f}")
        return ssim_value
    except Exception as e:
        if verbose:
            print(f"Error calculating Q3-masked SSIM: {e}")
        return np.nan

def fisher_z_transform(r):
    """Transforma correlação r para valor z, tratando correlações perfeitas."""
    if np.isnan(r):
        return np.nan
    # Mantém correlações perfeitas em vez de transformá-las em NaN
    if abs(r) >= 1:
        return 4.0 * np.sign(r)  # Um valor grande com o mesmo sinal de r
    r = np.clip(r, -0.9999, 0.9999)
    return 0.5 * np.log((1 + r) / (1 - r))

def fisher_z_inverse(z):
    """Transforma z de volta para correlação r."""
    if np.isnan(z):
        return np.nan
    return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)

def calculate_pearson_avg_with_fisher(values):
    """Calcula a média de correlação de Pearson usando transformação Fisher Z."""
    # Primeiro, converter para um array NumPy se não for
    values_array = np.asarray(values, dtype=float)
    
    # Agora podemos verificar valores NaN com segurança
    valid_values = values_array[~np.isnan(values_array)]
    
    if len(valid_values) == 0:
        return np.nan
        
    z_values = [fisher_z_transform(v) for v in valid_values if not np.isnan(fisher_z_transform(v))]
    if not z_values:
        return np.nan
        
    return fisher_z_inverse(np.mean(z_values))

# NOVA FUNÇÃO: Agregação personalizada para usar com pandas
def pearson_fisher_agg(series):
    """Função de agregação para calcular média de correlação com Fisher Z."""
    return calculate_pearson_avg_with_fisher(series.values)

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
            'min_both', 'q1_both', 'median_both', 'q3_both', 'max_both', 'mean_both', 'data_range'
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

def load_image(filepath, verbose=False):
    """Carrega uma imagem ou arquivo .npy com melhor tratamento de erro."""
    if filepath.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
        try:
            img = imread(filepath)
            
            if verbose:
                print(f"Loaded image {filepath.name}: shape={img.shape}, dtype={img.dtype}")
                if len(img.shape) > 2:
                    print(f"  Image has {img.shape[2]} channels")
                if np.isnan(img).any():
                    print(f"  Image contains {np.isnan(img).sum()} NaN values")
                print(f"  Value range: [{np.nanmin(img)}, {np.nanmax(img)}]")
            
            if len(img.shape) > 2 and img.shape[2] > 1:
                img = np.mean(img, axis=2)
                if verbose:
                    print(f"  Converted to grayscale: shape={img.shape}")
            
            return img
        except Exception as e:
            if verbose:
                print(f"ERROR loading {filepath}: {str(e)}")
            return None
    elif filepath.suffix.lower() == '.npy':
        try:
            img = np.load(filepath)
            if verbose:
                print(f"Loaded numpy array {filepath.name}: shape={img.shape}, dtype={img.dtype}")
                if np.isnan(img).any():
                    print(f"  Array contains {np.isnan(img).sum()} NaN values")
                print(f"  Value range: [{np.nanmin(img)}, {np.nanmax(img)}]")
            return img
        except Exception as e:
            if verbose:
                print(f"ERROR loading {filepath}: {str(e)}")
            return None
    else:
        if verbose:
            print(f"Unsupported file format: {filepath}")
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

def calculate_pair_stats(df, metric_type):
    """Calcula estatísticas para cada par de fontes (source_a, source_b)"""
    pair_stats = defaultdict(list)
    pair_counts = defaultdict(int)
    
    # Agrupa por pares de fontes
    for _, row in df.iterrows():
        source_a, source_b = row['source_a'], row['source_b']
        pair_key = f"{source_a} x {source_b}"
        
        # Só adiciona se houver um valor de métrica válido
        metric_val = row[metric_type]
        if not np.isnan(metric_val):
            pair_stats[pair_key].append(metric_val)
            pair_counts[pair_key] += 1
    
    # Calcula métricas resumidas para cada par
    pair_metrics = {}
    for pair_key, values in pair_stats.items():
        if metric_type == 'pearson':
            z_values = [fisher_z_transform(v) for v in values if not np.isnan(fisher_z_transform(v))]
            pair_metrics[pair_key] = {
                'metric_value': fisher_z_inverse(np.mean(z_values)) if z_values else np.nan,
                'count': pair_counts[pair_key]
            }
        elif metric_type == 'r2':
            # Para R², precisamos dos valores de r original (que é a raiz quadrada do R²)
            source_a, source_b = pair_key.split(' x ')
            r_values = df.loc[(df['source_a'] == source_a) & (df['source_b'] == source_b)]['pearson_r'].values
            valid_r = r_values[~np.isnan(r_values)]
            z_values = [fisher_z_transform(r) for r in valid_r if not np.isnan(fisher_z_transform(r))]
            pair_metrics[pair_key] = {
                'metric_value': fisher_z_inverse(np.mean(z_values)) ** 2 if z_values else np.nan,
                'count': pair_counts[pair_key]
            }
        else:
            # Para outras métricas, calculamos a média direta
            pair_metrics[pair_key] = {
                'metric_value': np.nanmean(values),
                'count': pair_counts[pair_key]
            }
    
    return pair_metrics

# NOVA FUNÇÃO: Calcula média mensal com Fisher Z para métricas de Pearson e derivados
def calculate_monthly_metrics(month_data, metric_type):
    """Calcula média mensal de métricas com tratamento especial para Pearson usando Fisher Z."""
    if metric_type == 'pearson':
        pearson_values = month_data[metric_type].values
        return calculate_pearson_avg_with_fisher(pearson_values)
    elif metric_type == 'r2':
        r_values = month_data['pearson_r'].values
        valid_r = r_values[~np.isnan(r_values)]
        z_values = [fisher_z_transform(r) for r in valid_r if not np.isnan(fisher_z_transform(r))]
        return fisher_z_inverse(np.mean(z_values)) ** 2 if z_values else np.nan
    elif metric_type == 'residual':
        return np.nanmean(np.abs(month_data[metric_type].values))
    else:
        return np.nanmean(month_data[metric_type].values)

# NOVA FUNÇÃO: Calcula estatísticas temporais por agregação manual
def calculate_temporal_stats(df, metric_type):
    """Calcula estatísticas temporais com tratamento adequado para correlações de Pearson."""
    if 'datetime' not in df.columns:
        return pd.DataFrame()
        
    # Preparar dataframe para análise temporal
    temp_df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(temp_df['datetime']):
        temp_df['datetime'] = pd.to_datetime(temp_df['datetime'])
    
    # Adicionar coluna de ano-mês
    temp_df['year_month'] = temp_df['datetime'].dt.strftime('%Y-%m')
    
    # Agrupar por par e ano-mês
    result = []
    for (source_a, source_b), group in temp_df.groupby(['source_a', 'source_b']):
        for year_month, month_group in group.groupby('year_month'):
            # Calcular métricas com tratamento adequado
            if metric_type == 'pearson':
                mean_value = calculate_pearson_avg_with_fisher(month_group[metric_type].values)
                q3_a_mean = calculate_pearson_avg_with_fisher(month_group[f'{metric_type}_q3_a'].values) 
                q3_b_mean = calculate_pearson_avg_with_fisher(month_group[f'{metric_type}_q3_b'].values)
            elif metric_type == 'r2':
                r_values = month_group['pearson_r'].values
                mean_value = calculate_pearson_avg_with_fisher(r_values) ** 2
                q3_a_mean = np.nanmean(month_group[f'{metric_type}_q3_a'].values)
                q3_b_mean = np.nanmean(month_group[f'{metric_type}_q3_b'].values)
            else:
                # Para outras métricas, usar média regular
                mean_value = np.nanmean(month_group[metric_type].values)
                q3_a_mean = np.nanmean(month_group[f'{metric_type}_q3_a'].values)
                q3_b_mean = np.nanmean(month_group[f'{metric_type}_q3_b'].values)
            
            # Calcular desvio padrão e contagem
            count = len(month_group)
            std_value = np.nanstd(month_group[metric_type].values) if count > 1 else np.nan
            
            result.append({
                'source_a': source_a,
                'source_b': source_b,
                'pair': f"{source_a} x {source_b}",
                'year_month': year_month,
                f'{metric_type}_mean': mean_value,
                f'{metric_type}_q3_a_mean': q3_a_mean,
                f'{metric_type}_q3_b_mean': q3_b_mean,
                f'{metric_type}_std': std_value,
                'count': count
            })
    
    return pd.DataFrame(result) if result else pd.DataFrame()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate metrics between datasets with simplified R², robust residuals, and Q3-based value selection')
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
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output for debugging')
    parser.add_argument('--sample-debug', type=int, default=0,
                        help='Number of sample file pairs to debug in detail')
    parser.add_argument('--check-images', action='store_true',
                        help='Verify image loading without running full analysis')
    parser.add_argument('--debug-skipped', action='store_true',
                        help='Enable detailed logging for skipped files')
    parser.add_argument('--debug-file', type=str, default="debug_missing_files.log",
                        help='File to log debug information about missing files')
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
    verbose = args.verbose
    sample_debug = args.sample_debug
    check_images = args.check_images
    debug_skipped = args.debug_skipped
    debug_file = args.debug_file
    
    higher_is_better = metric_type in ['pearson', 'r2', 'cosine', 'ssim']
    
    base_datasets = {
        'embrace': [
            # 'mapas1_embrace_2022_2024_0800',
            # 'mapas1_embrace_2022_2024_1600',
            # 'mapas1_embrace_2022_2024_2000_2200_0000_0200_0400',
            'mapas3_embrace_2024_0800_30m',
            'mapas3_embrace_2024_1600_30m',
            'mapas3_embrace_2024_2000_0400_30m'
        ],
        # 'igs': [
        #     'mapas1_igs_2022_2024_0800',
        #     'mapas1_igs_2022_2024_1600',
        #     'mapas1_igs_2022_2024_2000_2200_0000_0200_0400',
        #     'mapas2_igs_2022_2024_0800',
        #     'mapas2_igs_2022_2024_1600',
        #     'mapas2_igs_2022_2024_2000_2200_0000_0200_0400'
        # ],
        'maggia': [
            # 'mapas1_maggia_2022_2024_0800',
            # 'mapas1_maggia_2022_2024_1600',
            # 'mapas1_maggia_2022_2024_2000_2200_0000_0200_0400',
            # 'mapas2_maggia_2022_2024_0800',
            # 'mapas2_maggia_2024_1600_30m',
            # 'mapas2_maggia_2022_2024_2000_2200_0000_0200_0400',
            'mapas3_maggia_2024_0800_30m',
            'mapas3_maggia_2024_1600_30m',
            'mapas3_maggia_2024_2000_0400_30m'
        ],
        'nagoya': [
            # 'mapas1_nagoya_2022_2024_0800',
            # 'mapas1_nagoya_2022_2024_1600',
            # 'mapas1_nagoya_2022_2024_2000_2200_0000_0200_0400',
            # 'mapas2_nagoya_2022_2024_0800',
            # 'mapas2_nagoya_2024_1600_30m',
            # 'mapas2_nagoya_2022_2024_2000_2200_0000_0200_0400',
            'mapas3_nagoya_2024_0800_30m',
            'mapas3_nagoya_2024_1600_30m',
            'mapas3_nagoya_2024_2000_0400_30m'
        ]
    }
    
    datasets = {source: [f"{dataset}_{dataset_suffix}" for dataset in dataset_list 
                        if not filter_mapas3 or dataset.startswith('mapas3')] 
                for source, dataset_list in base_datasets.items()}
    
    comparisons = [
        # ['embrace', 'igs'],
        ['embrace', 'maggia'],
        ['embrace', 'nagoya'],
        # ['igs', 'maggia'],
        # ['igs', 'nagoya'],
        ['maggia', 'nagoya']
    ]
    
    base_dir = Path('.').resolve() / 'output'
    if not base_dir.exists():
        print(f"Output directory {base_dir} does not exist.")
        exit(1)
    
    print(f"Calculating {metric_type.upper()} metrics with Q3-based value selection...")
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
    
    # If we're just checking images, do that and exit
    if check_images:
        print("\nRunning image loading check for a sample of images...")
        for d in existing_dirs:
            sample_files = list(d.glob(file_extension))[:5]  # Check first 5 files
            print(f"\nChecking {len(sample_files)} files in {d.name}:")
            for file_path in sample_files:
                img = load_image(file_path, verbose=True)
                if img is None:
                    print(f"FAILED to load {file_path.name}")
                else:
                    print(f"SUCCESS loading {file_path.name}")
        exit(0)
    
    processed_files = 0
    skipped_files = 0
    total_missing = 0
    total_errors = 0
    result = []
    
    # Se debug_skipped for verdadeiro, inicialize o arquivo de debug
    debug_log = None
    if debug_skipped:
        debug_log = open(debug_file, 'w', encoding='utf-8')
        debug_log.write(f"Debug log for {metric_type} metric calculation with dataset suffix '{dataset_suffix}'\n")
        debug_log.write("=" * 80 + "\n\n")
    
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
            
            debug_count = 0
            pair_processed = 0
            pair_skipped = 0
            
            for file_a in files_a:
                file_b = dir_b / file_a.name
                if not file_b.exists():
                    skipped_files += 1
                    pair_skipped += 1
                    total_missing += 1
                    if debug_skipped:
                        debug_log.write(f"MISSING: {file_b} (matching {file_a})\n")
                    if verbose:
                        print(f"Skipping {file_a.name} - corresponding file not found in {dataset_b}")
                    continue
                
                # Enhanced debugging for a sample of files
                file_verbose = verbose or (sample_debug > 0 and debug_count < sample_debug)
                if file_verbose:
                    debug_count += 1
                    print(f"\n{'='*40}")
                    print(f"Processing file pair: {file_a.name}")
                    print(f"  Source A: {dir_a}")
                    print(f"  Source B: {dir_b}")
                
                try:
                    map_a = load_image(file_a, verbose=file_verbose)
                    map_b = load_image(file_b, verbose=file_verbose)
                    
                    if map_a is None or map_b is None:
                        skipped_files += 1
                        pair_skipped += 1
                        total_errors += 1
                        if debug_skipped:
                            debug_log.write(f"ERROR: Failed to load {file_a.name} or {file_b.name}\n")
                        if file_verbose:
                            print(f"Skipping {file_a.name} - failed to load one or both images")
                        continue
                    
                    map_a = np.nan_to_num(map_a, nan=np.nan)
                    map_b = np.nan_to_num(map_b, nan=np.nan)
                    map_a_flat = map_a.flatten()
                    map_b_flat = map_b.flatten()
                    
                    processed_files += 1
                    pair_processed += 1
                    stats = calculate_strict_stats(map_a, map_b)
                    
                    if file_verbose:
                        print(f"Statistics calculated for map pair:")
                        print(f"  Map A: Min={stats['min_a']:.4f}, Mean={stats['mean_a']:.4f}, Max={stats['max_a']:.4f}")
                        print(f"  Map B: Min={stats['min_b']:.4f}, Mean={stats['mean_b']:.4f}, Max={stats['max_b']:.4f}")
                        print(f"  Combined data range: {stats['data_range']:.4f}")

                    mask_a_q3, q3_a = calculate_q3_mask(map_a, verbose=file_verbose)
                    mask_b_q3, q3_b = calculate_q3_mask(map_b, verbose=file_verbose)
                    
                    # Validate Q3 masks
                    min_values = 100
                    valid_q3_a = mask_a_q3 is not None and mask_a_q3.sum() >= min_values
                    valid_q3_b = mask_b_q3 is not None and mask_b_q3.sum() >= min_values
                    
                    if file_verbose:
                        print(f"Q3 mask validation:")
                        print(f"  Map A: Q3 value={q3_a:.4f}, Valid mask: {'YES' if valid_q3_a else 'NO'}")
                        print(f"  Map B: Q3 value={q3_b:.4f}, Valid mask: {'YES' if valid_q3_b else 'NO'}")
                    
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
                    
                    # Original metric calculation
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
                        metric_value = calculate_ssim(y_true_2d, y_pred_2d, verbose=file_verbose)
                    
                    if file_verbose:
                        print(f"{metric_type.upper()} calculation result: {metric_value:.4f}")
                    
                    # Q3-based metrics
                    metric_q3_a = np.nan
                    metric_q3_b = np.nan
                    
                    # SSIM with Q3 masks requires special handling
                    if valid_q3_a and metric_type == 'ssim':
                        metric_q3_a = calculate_ssim_with_q3_mask(y_true_2d, y_pred_2d, mask_a_q3, verbose=file_verbose)
                    elif valid_q3_a:
                        if metric_type == 'pearson':
                            metric_q3_a = calculate_pearson(y_true, y_pred, file_a.name, value_mask=mask_a_q3.flatten())
                        elif metric_type == 'r2':
                            metric_q3_a, _ = calculate_r2_score(y_true, y_pred, file_a.name, value_mask=mask_a_q3.flatten())
                        elif metric_type == 'rmse':
                            metric_q3_a = calculate_rmse(y_true, y_pred, value_mask=mask_a_q3.flatten())
                        elif metric_type == 'mse':
                            metric_q3_a = calculate_mse(y_true, y_pred, value_mask=mask_a_q3.flatten())
                        elif metric_type == 'mae':
                            metric_q3_a = calculate_mae(y_true, y_pred, value_mask=mask_a_q3.flatten())
                        elif metric_type == 'residual':
                            metric_q3_a = calculate_residual_error(y_true, y_pred, normalize=normalize_residuals, filename=file_a.name, value_mask=mask_a_q3.flatten())
                        elif metric_type == 'max_residual':
                            metric_q3_a = calculate_max_residual_error(y_true, y_pred, normalize=normalize_residuals, filename=file_a.name, value_mask=mask_a_q3.flatten())
                        elif metric_type == 'min_residual':
                            metric_q3_a = calculate_min_residual_error(y_true, y_pred, min_residual_percentile, normalize=normalize_residuals, filename=file_a.name, value_mask=mask_a_q3.flatten())
                        elif metric_type == 'cosine':
                            metric_q3_a = calculate_cosine_similarity(y_true, y_pred, value_mask=mask_a_q3.flatten())
                        elif metric_type == 'huber':
                            metric_q3_a = calculate_huber_loss(y_true, y_pred, huber_delta, value_mask=mask_a_q3.flatten())
                    
                    if valid_q3_b and metric_type == 'ssim':
                        metric_q3_b = calculate_ssim_with_q3_mask(y_true_2d, y_pred_2d, mask_b_q3, verbose=file_verbose)
                    elif valid_q3_b:
                        if metric_type == 'pearson':
                            metric_q3_b = calculate_pearson(y_true, y_pred, file_a.name, value_mask=mask_b_q3.flatten())
                        elif metric_type == 'r2':
                            metric_q3_b, _ = calculate_r2_score(y_true, y_pred, file_a.name, value_mask=mask_b_q3.flatten())
                        elif metric_type == 'rmse':
                            metric_q3_b = calculate_rmse(y_true, y_pred, value_mask=mask_b_q3.flatten())
                        elif metric_type == 'mse':
                            metric_q3_b = calculate_mse(y_true, y_pred, value_mask=mask_b_q3.flatten())
                        elif metric_type == 'mae':
                            metric_q3_b = calculate_mae(y_true, y_pred, value_mask=mask_b_q3.flatten())
                        elif metric_type == 'residual':
                            metric_q3_b = calculate_residual_error(y_true, y_pred, normalize=normalize_residuals, filename=file_a.name, value_mask=mask_b_q3.flatten())
                        elif metric_type == 'max_residual':
                            metric_q3_b = calculate_max_residual_error(y_true, y_pred, normalize=normalize_residuals, filename=file_a.name, value_mask=mask_b_q3.flatten())
                        elif metric_type == 'min_residual':
                            metric_q3_b = calculate_min_residual_error(y_true, y_pred, min_residual_percentile, normalize=normalize_residuals, filename=file_a.name, value_mask=mask_b_q3.flatten())
                        elif metric_type == 'cosine':
                            metric_q3_b = calculate_cosine_similarity(y_true, y_pred, value_mask=mask_b_q3.flatten())
                        elif metric_type == 'huber':
                            metric_q3_b = calculate_huber_loss(y_true, y_pred, huber_delta, value_mask=mask_b_q3.flatten())
                    
                    # Display Q3 metric results in verbose mode
                    if file_verbose:
                        print(f"{metric_type.upper()} with Q3 A mask: {metric_q3_a:.4f}")
                        print(f"{metric_type.upper()} with Q3 B mask: {metric_q3_b:.4f}")
                    
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
                    
                    if metric_type in ['pearson', 'r2', 'cosine', 'ssim'] and not np.isnan(metric_value):
                        # Métricas já normalizadas
                        value_p = metric_value * 100
                    elif metric_type == 'mse' and not np.isnan(metric_value) and stats['data_range'] != 0:
                        # MSE precisa ser normalizado pelo quadrado do data_range
                        value_p = (metric_value / (stats['data_range'] ** 2) * 100)
                    elif metric_type in ['rmse', 'mae', 'residual', 'max_residual', 'min_residual', 'huber'] and not np.isnan(metric_value) and stats['data_range'] != 0:
                        # Outras métricas baseadas em erro
                        value_p = (metric_value / stats['data_range'] * 100)
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
                    if processed_files % 10 == 0 and not verbose:
                        print(f"Processed {processed_files} file pairs, Skipped {skipped_files} files...")
                except Exception as e:
                    skipped_files += 1
                    pair_skipped += 1
                    total_errors += 1
                    if debug_skipped:
                        debug_log.write(f"ERROR processing {file_a.name}: {str(e)}\n")
                    if file_verbose:
                        print(f"ERROR processing {file_a.name}: {str(e)}")
            
            # Informações resumidas para cada par de datasets
            print(f"\nFor pair {dataset_a} x {dataset_b}: Processed {pair_processed}, Skipped {pair_skipped}")
            if debug_skipped:
                debug_log.write(f"\nSummary for pair {dataset_a} x {dataset_b}:\n")
                debug_log.write(f"  Processed: {pair_processed}, Skipped: {pair_skipped}\n\n")
    
    if debug_skipped and debug_log:
        debug_log.write(f"\n\nFINAL SUMMARY:\n")
        debug_log.write(f"Total files processed: {processed_files}\n")
        debug_log.write(f"Total files skipped: {skipped_files}\n")
        debug_log.write(f"  - Missing files: {total_missing}\n")
        debug_log.write(f"  - Error loading files: {total_errors}\n")
        debug_log.close()
        print(f"\nDebug information saved to {debug_file}")
    
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
    
    # Calcular métricas médias por fonte ANTECIPADAMENTE
    dataset_avg_metrics = {}
    
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
            
            # CORREÇÃO: Aplicação consistente de Fisher Z para médias de Pearson
            if len(valid_metrics) > 0:
                if metric_type == 'pearson':
                    # Usar a função de média com Fisher Z
                    metric_value = calculate_pearson_avg_with_fisher(valid_metrics)
                elif metric_type == 'r2':
                    r_values = selection['pearson_r'].values
                    valid_r = r_values[~np.isnan(r_values)]
                    # Aplicar Fisher Z nas correlações r antes de converter para R²
                    metric_value = calculate_pearson_avg_with_fisher(valid_r) ** 2
                else:
                    metric_value = np.nanmean(np.abs(valid_metrics)) if metric_type == 'residual' else np.nanmean(valid_metrics)
            else:
                metric_value = np.nan
            
            # Calcular médias para métricas Q3 com Fisher Z quando aplicável
            metric_q3_a_values = selection[f'{metric_type}_q3_a'].values
            metric_q3_b_values = selection[f'{metric_type}_q3_b'].values
            valid_q3_a = metric_q3_a_values[~np.isnan(metric_q3_a_values)]
            valid_q3_b = metric_q3_b_values[~np.isnan(metric_q3_b_values)]
            
            # CORREÇÃO: Aplicar Fisher Z para médias Q3 de correlação de Pearson
            if metric_type == 'pearson':
                metric_q3_a_avg = calculate_pearson_avg_with_fisher(valid_q3_a)
                metric_q3_b_avg = calculate_pearson_avg_with_fisher(valid_q3_b)
            elif metric_type == 'r2' and 'pearson_r' in selection:
                # Para R², precisamos transformar os valores r individuais
                r_q3_a = np.sqrt(valid_q3_a)  # R² é o quadrado de r
                r_q3_b = np.sqrt(valid_q3_b)
                metric_q3_a_avg = calculate_pearson_avg_with_fisher(r_q3_a) ** 2
                metric_q3_b_avg = calculate_pearson_avg_with_fisher(r_q3_b) ** 2
            else:
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
            
            # Formatação padronizada para todos os tipos de métricas
            if metric_type in ['pearson', 'r2', 'ssim', 'cosine']:
                # Métricas já são normalizadas (entre -1 e 1 ou 0 e 1)
                metric_percent = metric_value * 100 if not np.isnan(metric_value) else np.nan
                percent_suffix = "%"
                is_normalized = True
            elif metric_type == 'mse' and not np.isnan(metric_value) and mean_data_range != 0:
                # MSE precisa ser normalizado pelo quadrado do data_range
                metric_percent = (metric_value / (mean_data_range ** 2) * 100)
                percent_suffix = "% of squared data range"
                is_normalized = False
            elif metric_type in ['rmse', 'mae', 'residual', 'max_residual', 'min_residual', 'huber'] and not np.isnan(metric_value) and mean_data_range != 0:
                # Outras métricas baseadas em erro, calcular como % do data_range
                metric_percent = (metric_value / mean_data_range * 100)
                percent_suffix = "% of data range"
                is_normalized = False
            else:
                metric_percent = np.nan
                percent_suffix = "%"
                is_normalized = False
            
            # Formatando a exibição da métrica com porcentagem
            percent_display = f"({metric_percent:.2f}{percent_suffix})" if not np.isnan(metric_percent) else "(NaN%)"
            
            if metric_type == 'pearson':
                print(f'Average Pearson Correlation: {metric_value:.4f} {percent_display} (Fisher Z applied)')
                print(f'Average Pearson (Q3 Map A): {format_metric_with_percent(metric_q3_a_avg, is_normalized=True)} (Fisher Z applied)')
                print(f'Average Pearson (Q3 Map B): {format_metric_with_percent(metric_q3_b_avg, is_normalized=True)} (Fisher Z applied)')
            elif metric_type == 'r2':
                print(f'Average R² Score: {metric_value:.4f} {percent_display} (Fisher Z applied on Pearson r)')
                print(f'Average R² (Q3 Map A): {format_metric_with_percent(metric_q3_a_avg, is_normalized=True)} (Fisher Z applied)')
                print(f'Average R² (Q3 Map B): {format_metric_with_percent(metric_q3_b_avg, is_normalized=True)} (Fisher Z applied)')
            elif metric_type == 'ssim':
                print(f'Average Structural Similarity Index: {metric_value:.4f} {percent_display}')
                print(f'Average SSIM (Q3 Map A): {format_metric_with_percent(metric_q3_a_avg, is_normalized=True)}')
                print(f'Average SSIM (Q3 Map B): {format_metric_with_percent(metric_q3_b_avg, is_normalized=True)}')
            elif metric_type == 'cosine':
                print(f'Average Cosine Similarity: {metric_value:.4f} {percent_display}')
                print(f'Average Cosine (Q3 Map A): {format_metric_with_percent(metric_q3_a_avg, is_normalized=True)}')
                print(f'Average Cosine (Q3 Map B): {format_metric_with_percent(metric_q3_b_avg, is_normalized=True)}')
            elif metric_type == 'rmse':
                print(f'Average RMSE: {metric_value:.4f} {percent_display}')
                print(f'Average RMSE (Q3 Map A): {format_metric_with_percent(metric_q3_a_avg, mean_data_range)}')
                print(f'Average RMSE (Q3 Map B): {format_metric_with_percent(metric_q3_b_avg, mean_data_range)}')
            elif metric_type == 'mse':
                print(f'Average Mean Squared Error: {metric_value:.4f} {percent_display}')
                print(f'Average MSE (Q3 Map A): {format_metric_with_percent(metric_q3_a_avg, mean_data_range**2)}')
                print(f'Average MSE (Q3 Map B): {format_metric_with_percent(metric_q3_b_avg, mean_data_range**2)}')
            elif metric_type == 'mae':
                print(f'Average Mean Absolute Error: {metric_value:.4f} {percent_display}')
                print(f'Average MAE (Q3 Map A): {format_metric_with_percent(metric_q3_a_avg, mean_data_range)}')
                print(f'Average MAE (Q3 Map B): {format_metric_with_percent(metric_q3_b_avg, mean_data_range)}')
            elif metric_type == 'residual':
                print(f'Average Mean Absolute Residual Error: {metric_value:.4f} {percent_display}')
                print(f'Average Residual (Q3 Map A): {format_metric_with_percent(metric_q3_a_avg, mean_data_range)}')
                print(f'Average Residual (Q3 Map B): {format_metric_with_percent(metric_q3_b_avg, mean_data_range)}')
            elif metric_type == 'max_residual':
                print(f'Average Maximum Residual Error: {metric_value:.4f} {percent_display}')
                print(f'Average Max Residual (Q3 Map A): {format_metric_with_percent(metric_q3_a_avg, mean_data_range)}')
                print(f'Average Max Residual (Q3 Map B): {format_metric_with_percent(metric_q3_b_avg, mean_data_range)}')
            elif metric_type == 'min_residual':
                suffix = f" ({min_residual_percentile}th percentile)"
                print(f'Average Minimum Residual Error{suffix}: {metric_value:.4f} {percent_display}')
                print(f'Average Min Residual (Q3 Map A): {format_metric_with_percent(metric_q3_a_avg, mean_data_range)}')
                print(f'Average Min Residual (Q3 Map B): {format_metric_with_percent(metric_q3_b_avg, mean_data_range)}')
            elif metric_type == 'huber':
                suffix = f" (delta={huber_delta})"
                print(f'Average Huber Loss{suffix}: {metric_value:.4f} {percent_display}')
                print(f'Average Huber Loss (Q3 Map A): {format_metric_with_percent(metric_q3_a_avg, mean_data_range)}')
                print(f'Average Huber Loss (Q3 Map B): {format_metric_with_percent(metric_q3_b_avg, mean_data_range)}')
            
            sorted_maps = selection.sort_values(by=f'{metric_type}_p', ascending=not higher_is_better).head(top_n)
            comp_key = f"{dataset_a} x {dataset_b}"
            top_maps_by_comparison[comp_key] = sorted_maps
            
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
                    elif metric_type == 'mse':
                        metric_display = f"{getattr(row, metric_type):.4f} ({getattr(row, f'{metric_type}_p'):.2f}% of squared data range)"
                    else:
                        metric_display = f"{getattr(row, metric_type):.4f} ({getattr(row, f'{metric_type}_p'):.2f}% of data range)"
                else:
                    # Se não tiver o atributo de porcentagem, calcular
                    metric_value = getattr(row, metric_type)
                    if is_normalized:
                        percent = metric_value * 100 if not np.isnan(metric_value) else np.nan
                        metric_display = f"{metric_value:.4f} ({percent:.2f}%)" if not np.isnan(percent) else f"{metric_value:.4f} (NaN%)"
                    elif metric_type == 'mse':
                        percent = (metric_value / (row.data_range ** 2) * 100) if not np.isnan(metric_value) and row.data_range != 0 else np.nan
                        metric_display = f"{metric_value:.4f} ({percent:.2f}% of squared data range)" if not np.isnan(percent) else f"{metric_value:.4f} (NaN%)"
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
                elif metric_type == 'mse':
                    q3_a_percent = (q3_a_value / (row.data_range ** 2) * 100) if not np.isnan(q3_a_value) and row.data_range != 0 else np.nan
                    q3_b_percent = (q3_b_value / (row.data_range ** 2) * 100) if not np.isnan(q3_b_value) and row.data_range != 0 else np.nan
                    q3_a_display = f"{q3_a_value:.4f} ({q3_a_percent:.2f}% of squared data range)" if not np.isnan(q3_a_value) else 'NaN'
                    q3_b_display = f"{q3_b_value:.4f} ({q3_b_percent:.2f}% of squared data range)" if not np.isnan(q3_b_value) else 'NaN'
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
            
            # CORREÇÃO: Usar Fisher Z para análises mensais de Pearson
            if 'datetime' in selection.columns:
                for year in [2022, 2023, 2024]:
                    for month in range(1, 13):
                        month_data = selection.loc[(selection['datetime'].dt.month == month) &
                                                 (selection['datetime'].dt.year == year)]
                        if not month_data.empty:
                            month_name = pd.Timestamp(year=year, month=month, day=1).strftime('%B/%Y')
                            
                            # CORREÇÃO: Usar a função de cálculo mensal que aplica Fisher Z quando necessário
                            month_metric = calculate_monthly_metrics(month_data, metric_type)
                            
                            if metric_type in ['pearson', 'r2', 'cosine', 'ssim']:
                                # Métricas já normalizadas
                                month_percent = month_metric * 100 if not np.isnan(month_metric) else np.nan
                                percent_suffix = "%"
                            elif metric_type == 'mse' and not np.isnan(month_metric) and mean_data_range != 0:
                                # MSE precisa ser normalizado pelo quadrado do data_range
                                month_percent = (month_metric / (mean_data_range ** 2) * 100)
                                percent_suffix = "% of squared data range"
                            elif metric_type in ['rmse', 'mae', 'residual', 'max_residual', 'min_residual', 'huber']:
                                # Outras métricas baseadas em erro
                                month_percent = (month_metric / mean_data_range * 100) if not np.isnan(month_metric) and mean_data_range != 0 else np.nan
                                percent_suffix = "% of data range"

                            percent_display = f"({month_percent:.2f}{percent_suffix})" if not np.isnan(month_percent) else "(NaN%)"
                            fisher_note = " (Fisher Z applied)" if metric_type in ['pearson', 'r2'] else ""
                            print(f'{month_name}: {metric_type.upper()} = {month_metric:.4f} {percent_display}{fisher_note}')
    
    # Calcular as métricas médias por dataset usando Fisher Z para Pearson
    print("\nCalculating average metrics...")
    for dataset in dataset_metrics:
        valid_metrics = [v for v in dataset_metrics[dataset] if not np.isnan(v)]
        if valid_metrics:
            # CORREÇÃO: Usar Fisher Z para médias de Pearson
            if metric_type == 'pearson':
                dataset_avg_metrics[dataset] = calculate_pearson_avg_with_fisher(valid_metrics)
            elif metric_type == 'r2':
                r_values = df[(df['source_a'] == dataset) | (df['source_b'] == dataset)]['pearson_r'].values
                valid_r = r_values[~np.isnan(r_values)]
                dataset_avg_metrics[dataset] = calculate_pearson_avg_with_fisher(valid_r) ** 2
            else:
                dataset_avg_metrics[dataset] = np.nanmean(valid_metrics)
        else:
            dataset_avg_metrics[dataset] = np.nan
    
    # NOVA SEÇÃO: ANÁLISE POR PARES DE FONTES
    print("\n===== PAIRS COMPARISON =====")
    pair_metrics = calculate_pair_stats(df, metric_type)

    # Calcular percentuais para cada par
    for pair_key, data in pair_metrics.items():
        metric_val = data['metric_value']
        if np.isnan(metric_val):
            data['percent'] = np.nan
            continue
            
        # Calcular o percentual baseado no tipo de métrica
        if metric_type in ['pearson', 'r2', 'cosine', 'ssim']:
            data['percent'] = metric_val * 100
        else:
            # Para métricas baseadas em erro, precisamos do data_range médio para este par
            source_a, source_b = pair_key.split(' x ')
            pair_data_ranges = df[(df['source_a'] == source_a) & (df['source_b'] == source_b)]['data_range'].values
            avg_data_range = np.nanmean(pair_data_ranges) if len(pair_data_ranges) > 0 else 1.0
            
            if metric_type == 'mse' and avg_data_range != 0:
                data['percent'] = (metric_val / (avg_data_range ** 2) * 100)
            elif avg_data_range != 0:
                data['percent'] = (metric_val / avg_data_range * 100)
            else:
                data['percent'] = np.nan

    # Ordenar os pares por sua métrica (maior para menor ou menor para maior dependendo do tipo)
    sorted_pairs = list(pair_metrics.items())
    if higher_is_better:
        sorted_pairs.sort(key=lambda x: float('-inf') if np.isnan(x[1]['percent']) else x[1]['percent'], reverse=True)
    else:
        sorted_pairs.sort(key=lambda x: float('inf') if np.isnan(x[1]['percent']) else x[1]['percent'], reverse=False)

    # Exibir o ranking de pares
    print("\nPairs Ranking (from best to worst):")
    for i, (pair_key, data) in enumerate(sorted_pairs, 1):
        metric_val = data['metric_value']
        percent = data['percent']
        count = data['count']
        
        if np.isnan(metric_val):
            print(f"{i}. {pair_key}: NaN (unable to calculate metric) - {count} comparisons")
            continue
            
        if metric_type in ['pearson', 'r2', 'cosine', 'ssim']:
            percent_suffix = "%"
        elif metric_type == 'mse':
            percent_suffix = "% of squared data range"
        else:
            percent_suffix = "% of data range"

        percent_display = f"({percent:.2f}{percent_suffix})" if not np.isnan(percent) else "(NaN%)"
        fisher_note = " (Fisher Z applied)" if metric_type in ['pearson', 'r2'] else ""
        print(f"{i}. {pair_key}: {metric_val:.4f} {percent_display}{fisher_note} - {count} comparisons")

    # Exportar métricas de pares para um arquivo CSV
    pair_data = []
    for pair_key, data in pair_metrics.items():
        source_a, source_b = pair_key.split(' x ')
        pair_data.append({
            'source_a': source_a,
            'source_b': source_b,
            'pair': pair_key,
            f'{metric_type}_value': data['metric_value'],
            f'{metric_type}_percent': data['percent'],
            'comparison_count': data['count']
        })

    if pair_data:
        pair_df = pd.DataFrame(pair_data)
        output_file = f'pair_metrics_{metric_type}.csv'
        pair_df.to_csv(output_file, index=False)
        print(f"\nPair metrics exported to {output_file}")
    
    # CORREÇÃO: Usar a nova função calculate_temporal_stats para análise temporal
    if 'datetime' in df.columns:
        print("\n===== TEMPORAL ANALYSIS BY PAIR =====")
        
        # Usar a função corrigida para análise temporal
        temporal_df = calculate_temporal_stats(df, metric_type)
        
        if not temporal_df.empty:
            try:
                temporal_file = f'temporal_analysis_{metric_type}.csv'
                temporal_df.to_csv(temporal_file, index=False)
                print(f"Temporal analysis exported to {temporal_file}")
            except Exception as e:
                print(f"Error in temporal analysis export: {str(e)}")

    print("\n===== SOURCE ANALYSIS =====")
    # Para cada fonte, mostre a qualidade média de suas comparações
    for source in dataset_metrics:
        source_pairs = [pair for pair in pair_metrics.keys() if source in pair]
        if not source_pairs:
            continue
        
        print(f"\nSource: {source}")
        avg_value = dataset_avg_metrics[source]
        if metric_type in ['pearson', 'r2', 'cosine', 'ssim']:
            percent = avg_value * 100 if not np.isnan(avg_value) else np.nan
            percent_suffix = "%"
        else:
            data_ranges = df[(df['source_a'] == source) | (df['source_b'] == source)]['data_range'].values
            avg_data_range = np.nanmean(data_ranges) if len(data_ranges) > 0 else 1.0
            if metric_type == 'mse' and not np.isnan(avg_value) and avg_data_range != 0:
                percent = (avg_value / (avg_data_range ** 2) * 100)
                percent_suffix = "% of squared data range"
            elif not np.isnan(avg_value) and avg_data_range != 0:
                percent = (avg_value / avg_data_range * 100)
                percent_suffix = "% of data range"
            else:
                percent = np.nan
                percent_suffix = "%"
        
        percent_display = f"({percent:.2f}{percent_suffix})" if not np.isnan(percent) else "(NaN%)"
        fisher_note = " (Fisher Z applied)" if metric_type in ['pearson', 'r2'] else ""
        print(f"Average {metric_type}: {avg_value:.4f} {percent_display}{fisher_note}")
        print("Comparisons:")
        
        # Listar todos os pares envolvendo esta fonte
        for pair in source_pairs:
            metric_val = pair_metrics[pair]['metric_value']
            percent = pair_metrics[pair]['percent']
            count = pair_metrics[pair]['count']
            
            # Extrair as fontes do par
            s1, s2 = pair.split(' x ')
            
            # Garantir que a fonte atual esteja primeiro no display
            if s1 != source:
                # Reordenar para mostrar a fonte atual primeiro
                display_pair = f"{source} x {s1}"
            else:
                display_pair = pair
            
            if np.isnan(metric_val):
                print(f"  {display_pair}: NaN (unable to calculate metric) - {count} comparisons")
                continue
                
            if metric_type in ['pearson', 'r2', 'cosine', 'ssim']:
                percent_suffix = "%"
            elif metric_type == 'mse':
                percent_suffix = "% of squared data range"
            else:
                percent_suffix = "% of data range"
        
            percent_display = f"({percent:.2f}{percent_suffix})" if not np.isnan(percent) else "(NaN%)"
            fisher_note = " (Fisher Z applied)" if metric_type in ['pearson', 'r2'] else ""
            print(f"  {display_pair}: {metric_val:.4f} {percent_display}{fisher_note} - {count} comparisons")
            
            # Add monthly breakdown for this pair
            if 'datetime' in df.columns:
                source_a, source_b = pair.split(' x ')
                
                # Determinar qual fonte é a atual (para Q3) e qual é a outra
                if source_a == source:
                    current_source = source_a
                    other_source = source_b
                    is_source_a_current = True
                else:
                    current_source = source_b
                    other_source = source_a
                    is_source_a_current = False
                
                pair_data = df[(df['source_a'] == source_a) & (df['source_b'] == source_b)]
                
                if not pair_data.empty:
                    # Ensure datetime format
                    if not pd.api.types.is_datetime64_any_dtype(pair_data['datetime']):
                        pair_data['datetime'] = pd.to_datetime(pair_data['datetime'])
                    
                    # Sort by date to maintain chronological order
                    pair_data = pair_data.sort_values('datetime')
                    
                    # Group by year and month
                    pair_data['year'] = pair_data['datetime'].dt.year
                    pair_data['month'] = pair_data['datetime'].dt.month
                    
                    print("    Monthly breakdown:")
                    
                    # Process all years and months in the data
                    for year in sorted(pair_data['year'].unique()):
                        for month in sorted(pair_data[pair_data['year'] == year]['month'].unique()):
                            month_data = pair_data[(pair_data['year'] == year) & (pair_data['month'] == month)]
                            month_count = len(month_data)
                            
                            if month_count > 0:
                                # Format month name
                                month_name = f"{month}/{year}"
                                
                                # CORREÇÃO: Usar a função auxiliar para cálculo mensal com Fisher Z
                                month_metric = calculate_monthly_metrics(month_data, metric_type)
                                
                                # Calcular Q3-based metrics com Fisher Z quando apropriado
                                if metric_type == 'pearson':
                                    month_q3_a_metric = calculate_pearson_avg_with_fisher(month_data[f'{metric_type}_q3_a'].values)
                                    month_q3_b_metric = calculate_pearson_avg_with_fisher(month_data[f'{metric_type}_q3_b'].values)
                                elif metric_type == 'r2':
                                    q3_a_values = month_data[f'{metric_type}_q3_a'].values
                                    q3_b_values = month_data[f'{metric_type}_q3_b'].values
                                    # Estimar os valores r a partir de R²
                                    r_q3_a = np.sqrt(q3_a_values[~np.isnan(q3_a_values)])
                                    r_q3_b = np.sqrt(q3_b_values[~np.isnan(q3_b_values)])
                                    month_q3_a_metric = calculate_pearson_avg_with_fisher(r_q3_a) ** 2 if len(r_q3_a) > 0 else np.nan
                                    month_q3_b_metric = calculate_pearson_avg_with_fisher(r_q3_b) ** 2 if len(r_q3_b) > 0 else np.nan
                                else:
                                    month_q3_a_metric = np.nanmean(month_data[f'{metric_type}_q3_a'].values)
                                    month_q3_b_metric = np.nanmean(month_data[f'{metric_type}_q3_b'].values)
                                
                                # Calculate data range for this month
                                month_data_range = np.nanmean(month_data['data_range'].values)
                                
                                # Format the metric display with percentages
                                if metric_type in ['pearson', 'r2', 'cosine', 'ssim']:
                                    # Already normalized metrics
                                    month_percent = month_metric * 100 if not np.isnan(month_metric) else np.nan
                                    q3_a_percent = month_q3_a_metric * 100 if not np.isnan(month_q3_a_metric) else np.nan
                                    q3_b_percent = month_q3_b_metric * 100 if not np.isnan(month_q3_b_metric) else np.nan
                                    percent_suffix = "%"
                                elif metric_type == 'mse':
                                    # MSE is normalized by squared data range
                                    month_percent = (month_metric / (month_data_range ** 2) * 100) if not np.isnan(month_metric) and month_data_range > 0 else np.nan
                                    q3_a_percent = (month_q3_a_metric / (month_data_range ** 2) * 100) if not np.isnan(month_q3_a_metric) and month_data_range > 0 else np.nan
                                    q3_b_percent = (month_q3_b_metric / (month_data_range ** 2) * 100) if not np.isnan(month_q3_b_metric) and month_data_range > 0 else np.nan
                                    percent_suffix = "% of squared data range"
                                else:
                                    # Other error-based metrics
                                    month_percent = (month_metric / month_data_range * 100) if not np.isnan(month_metric) and month_data_range > 0 else np.nan
                                    q3_a_percent = (month_q3_a_metric / month_data_range * 100) if not np.isnan(month_q3_a_metric) and month_data_range > 0 else np.nan
                                    q3_b_percent = (month_q3_b_metric / month_data_range * 100) if not np.isnan(month_q3_b_metric) and month_data_range > 0 else np.nan
                                    percent_suffix = "% of data range"
                                
                                # Prepare displays
                                month_percent_display = f"({month_percent:.2f}{percent_suffix})" if not np.isnan(month_percent) else "(NaN%)"
                                q3_a_percent_display = f"({q3_a_percent:.2f}{percent_suffix})" if not np.isnan(q3_a_percent) else "(NaN%)"
                                q3_b_percent_display = f"({q3_b_percent:.2f}{percent_suffix})" if not np.isnan(q3_b_percent) else "(NaN%)"
                                
                                # Output formatted results with Fisher Z notation when appropriate
                                fisher_note = " (Fisher Z applied)" if metric_type in ['pearson', 'r2'] else ""
                                print(f"      {month_name}: {month_metric:.4f} {month_percent_display}{fisher_note} - {month_count} comparisons")
                                
                                # Determinar os nomes das fontes para as métricas Q3
                                if is_source_a_current:
                                    q3_current_source_metric = month_q3_a_metric
                                    q3_current_source_display = q3_a_percent_display
                                    q3_other_source_metric = month_q3_b_metric
                                    q3_other_source_display = q3_b_percent_display
                                else:
                                    q3_current_source_metric = month_q3_b_metric
                                    q3_current_source_display = q3_b_percent_display
                                    q3_other_source_metric = month_q3_a_metric
                                    q3_other_source_display = q3_a_percent_display
                                
                                # Only show Q3 metrics if they have valid values
                                if not np.isnan(q3_current_source_metric):
                                    print(f"        Q3 {current_source}: {q3_current_source_metric:.4f} {q3_current_source_display}{fisher_note}")
                                if not np.isnan(q3_other_source_metric):
                                    print(f"        Q3 {other_source}: {q3_other_source_metric:.4f} {q3_other_source_display}{fisher_note}")
    
    # Exportar métricas de pares para um arquivo CSV
    pair_data = []
    for pair_key, data in pair_metrics.items():
        source_a, source_b = pair_key.split(' x ')
        pair_data.append({
            'source_a': source_a,
            'source_b': source_b,
            'pair': pair_key,
            f'{metric_type}_value': data['metric_value'],
            f'{metric_type}_percent': data['percent'],
            'comparison_count': data['count']
        })
    
    if pair_data:
        pair_df = pd.DataFrame(pair_data)
        output_file = f'pair_metrics_{metric_type}.csv'
        pair_df.to_csv(output_file, index=False)
        print(f"\nPair metrics exported to {output_file}")
        
    # Exportar resultados de análise temporal por par (se disponível)
    if 'datetime' in df.columns:
        print("\n===== TEMPORAL ANALYSIS BY PAIR =====")
        
        # CORREÇÃO: Já estamos usando a nova função que suporta Fisher Z
        temporal_df = calculate_temporal_stats(df, metric_type)
        
        if not temporal_df.empty:
            try:
                temporal_file = f'temporal_analysis_{metric_type}_with_q3.csv'
                temporal_df.to_csv(temporal_file, index=False)
                print(f"Temporal analysis exported to {temporal_file}")
            except Exception as e:
                print(f"Error in temporal analysis export: {str(e)}")
            
    print(f"\n===== ANALYSIS COMPLETE =====")
    print(f"Total files processed: {processed_files}")
    print(f"Total pairs analyzed: {len(pair_metrics)}")
    print(f"Total files skipped: {skipped_files}")
    print(f"Total missing files: {total_missing}")
    print(f"Total error files: {total_errors}")
    print(f"Results saved to:")
    print(f"  - result_{metric_type}_with_stats.csv (all individual comparisons)")
    print(f"  - pair_metrics_{metric_type}.csv (pair summary metrics)")
    if 'datetime' in df.columns:
        print(f"  - temporal_analysis_{metric_type}_with_q3.csv (monthly metrics by pair)")
    if debug_skipped:
        print(f"  - {debug_file} (detailed log of missing and error files)")

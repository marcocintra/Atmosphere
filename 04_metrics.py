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
    
    if verbose:
        print(f"Calculating Q3 mask for data shape {map_data.shape}")
    
    non_nan_mask = ~np.isnan(map_data)
    non_nan_count = np.sum(non_nan_mask)
    
    if verbose:
        print(f"Non-NaN values: {non_nan_count}/{map_data.size} ({non_nan_count/map_data.size*100:.2f}%)")
    
    valid_values = map_data[non_nan_mask]
    
    if len(valid_values) == 0:
        if verbose:
            print("No valid values found for Q3 calculation")
        return None, np.nan
        
    q3 = np.percentile(valid_values, 75)
    if verbose:
        print(f"Q3 value: {q3:.4f}")
    
    unique_values = np.unique(valid_values)
    if q3 == np.max(valid_values) and len(unique_values) > 1:
        if verbose:
            print("Q3 equals max value, using second highest value instead")
        q3 = unique_values[-2] 
    
    mask = non_nan_mask & (map_data >= q3)
    mask_count = np.sum(mask)
    
    if verbose:
        mask_percent = mask_count / map_data.size * 100
        print(f"Values in Q3 mask: {mask_count} ({mask_percent:.2f}%)")
    
    return mask, q3

def calculate_pearson(y_true, y_pred, filename="unknown", value_mask=None):
    
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
    
    var_true = np.var(y_true_valid)
    var_pred = np.var(y_pred_valid)
    
    if var_true < 1e-10 or var_pred < 1e-10:
        return 0.0
    
    try:
        corr = np.corrcoef(y_true_valid, y_pred_valid)[0, 1]
        
        if np.isnan(corr) or np.isinf(corr):
            return np.nan
            
        return corr
    except Exception as e:
        return np.nan

def calculate_r2_score(y_true, y_pred, filename="unknown", value_mask=None):
    
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

def calculate_ssim(y_true, y_pred, value_mask=None, verbose=False, calculation_context="Overall"):
    
    print(f"\n--- Start of SSIM calculation with anti-correlation (Context: {calculation_context}) ---")
    print(f"y_true shape: {y_true.shape}, dtype: {y_true.dtype}")
    print(f"y_pred shape: {y_pred.shape}, dtype: {y_pred.dtype}")

    if value_mask is not None:
        print(f"Applying provided mask '{calculation_context}'.")
        print(f"Total pixels in mask: {np.sum(value_mask)}")
        valid_mask = value_mask & ~np.isnan(y_true) & ~np.isnan(y_pred)
    else:
        print("No specific mask provided. Using all valid pixels (non-NaN).")
        valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)

    valid_count = np.sum(valid_mask)
    invalid_count = np.sum(~valid_mask)
    valid_percentage = valid_count / valid_mask.size * 100

    print(f"Valid pixels: {valid_count}/{valid_mask.size} ({valid_percentage:.2f}%)")
    print(f"Invalid pixels to be anti-correlated: {invalid_count}")
    
    if verbose:
        print(f"NaN count in y_true: {np.sum(np.isnan(y_true))}")
        print(f"NaN count in y_pred: {np.sum(np.isnan(y_pred))}")
    
    if valid_count < 100:
        print(f"WARNING: Too few valid pixels ({valid_count}). Returning NaN.")
        print(f"--- End of SSIM calculation with WARNING (Context: {calculation_context}) ---")
        return np.nan

    y_true_final = y_true.copy()
    y_pred_final = y_pred.copy()
    
    invalid_mask = ~valid_mask
    
    if np.sum(invalid_mask) > 0:
        print(f"Applying anti-correlation to {np.sum(invalid_mask)} invalid pixels...")
        
        y_true_final[invalid_mask] = -1.0
        y_pred_final[invalid_mask] = +1.0
        
        if verbose:
            print("Invalid pixels set to: y_true=-1.0, y_pred=+1.0 (maximum anti-correlation)")
    
    pred_valid_values = y_pred_final[valid_mask]  
    pred_invalid_values = y_pred_final[invalid_mask]
    
    if len(pred_valid_values) > 0:
        
        all_pred_values = np.concatenate([pred_valid_values, pred_invalid_values]) if len(pred_invalid_values) > 0 else pred_valid_values
        data_min = np.min(all_pred_values)
        data_max = np.max(all_pred_values)
        data_range = data_max - data_min
        
        print(f"Data range from y_pred only: {data_range:.4f} (from {data_min:.4f} to {data_max:.4f})")
    else:
        print("ERROR: No valid pixels found.")
        return np.nan
    
    try:
        ssim_kwargs = {
            'data_range': data_range
        }
        
        ssim_value = ssim(y_true_final, y_pred_final, **ssim_kwargs)
        
        if np.isnan(ssim_value) or np.isinf(ssim_value):
            print(f"ERROR: SSIM returned an invalid value ({ssim_value}).")
            print(f"--- End of SSIM calculation with ERROR (Context: {calculation_context}) ---")
            return np.nan

        print(f"SSIM with anti-correlated invalid pixels: {ssim_value:.4f}")
        
        print(f"--- End of SSIM calculation (Context: {calculation_context}) ---")
        return ssim_value
        
    except Exception as e:
        print(f"ERROR: SSIM calculation failed with exception: {str(e)}")
        import traceback
        print(traceback.format_exc())
        print(f"--- End of SSIM calculation with ERROR (Context: {calculation_context}) ---")
        return np.nan

def fisher_z_transform(r):
    
    if np.isnan(r):
        return np.nan
    if abs(r) >= 1:
        return 4.0 * np.sign(r)
    r = np.clip(r, -0.9999, 0.9999)
    return 0.5 * np.log((1 + r) / (1 - r))

def fisher_z_inverse(z):
    """Transforma z de volta para correlação r."""
    if np.isnan(z):
        return np.nan
    return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)

def calculate_avg_with_fisher(values):
    
    values_array = np.asarray(values, dtype=float)
    
    valid_values = values_array[~np.isnan(values_array)]
    
    if len(valid_values) == 0:
        return np.nan
        
    z_values = [fisher_z_transform(v) for v in valid_values if not np.isnan(fisher_z_transform(v))]
    if not z_values:
        return np.nan
        
    return fisher_z_inverse(np.mean(z_values))

def pearson_fisher_agg(series):
    
    return calculate_avg_with_fisher(series.values)

def calculate_strict_stats(map_a, map_b):
    
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
    q1_a = float(np.percentile(valid_a, 25)) 
    median_a = float(np.median(valid_a))
    q3_a = float(np.percentile(valid_a, 75)) 
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
    
    if filepath.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
        try:
            img = imread(filepath, as_gray=True)
            
            if verbose:
                print(f"Loaded image {filepath.name}: shape={img.shape}, dtype={img.dtype}")
                if len(img.shape) > 2:
                    print(f"  Image has {img.shape[2]} channels")
                if np.isnan(img).any():
                    print(f"  Image contains {np.isnan(img).sum()} NaN values")
                print(f"  Value range: [{np.nanmin(img)}, {np.nanmax(img)}]")
            
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

def calculate_pair_stats(df, metric_type, use_fisher_for_ssim=False):
 
    pair_stats = defaultdict(list)
    pair_counts = defaultdict(int)
    
    for _, row in df.iterrows():
        source_a, source_b = row['source_a'], row['source_b']
        pair_key = f"{source_a} x {source_b}"
        
        metric_val = row[metric_type]
        if not np.isnan(metric_val):
            pair_stats[pair_key].append(metric_val)
            pair_counts[pair_key] += 1
    
    pair_metrics = {}
    for pair_key, values in pair_stats.items():
        if metric_type == 'pearson':
            
            z_values = [fisher_z_transform(v) for v in values if not np.isnan(fisher_z_transform(v))]
            pair_metrics[pair_key] = {
                'metric_value': fisher_z_inverse(np.mean(z_values)) if z_values else np.nan,
                'count': pair_counts[pair_key]
            }
         
        elif metric_type == 'ssim':
            
            pair_metrics[pair_key] = {
                'metric_value': calculate_avg_with_fisher(values) if use_fisher_for_ssim else np.nanmean(values),
                'count': pair_counts[pair_key]
            }
        elif metric_type == 'r2':
            
            source_a, source_b = pair_key.split(' x ')
            r_values = df.loc[(df['source_a'] == source_a) & 
                            (df['source_b'] == source_b)]['pearson_r'].values
            valid_r = r_values[~np.isnan(r_values)]
            z_values = [fisher_z_transform(r) for r in valid_r 
                       if not np.isnan(fisher_z_transform(r))]
            pair_metrics[pair_key] = {
                'metric_value': fisher_z_inverse(np.mean(z_values)) ** 2 if z_values else np.nan,
                'count': pair_counts[pair_key]
            }
        elif metric_type == 'residual':
            
            pair_metrics[pair_key] = {
                'metric_value': np.nanmean(np.abs(values)),
                'count': pair_counts[pair_key]
            }
        else:
            
            pair_metrics[pair_key] = {
                'metric_value': np.nanmean(values),
                'count': pair_counts[pair_key]
            }
    
    return pair_metrics

def calculate_monthly_metrics(month_data, metric_type, use_fisher_for_ssim=False):
    
    if metric_type == 'pearson':
        pearson_values = month_data[metric_type].values
        return calculate_avg_with_fisher(pearson_values)
    elif metric_type == 'r2':
        r_values = month_data['pearson_r'].values
        valid_r = r_values[~np.isnan(r_values)]
        z_values = [fisher_z_transform(r) for r in valid_r if not np.isnan(fisher_z_transform(r))]
        return fisher_z_inverse(np.mean(z_values)) ** 2 if z_values else np.nan
    elif metric_type == 'residual':
        return np.nanmean(np.abs(month_data[metric_type].values))
    elif metric_type == 'ssim':
        ssim_values = month_data[metric_type].values
        if use_fisher_for_ssim:
            return calculate_avg_with_fisher(ssim_values)
        else:
            return np.nanmean(ssim_values)
    else:
        return np.nanmean(month_data[metric_type].values)

def calculate_temporal_stats(df, metric_type):
    
    if 'datetime' not in df.columns:
        return pd.DataFrame()
        
    temp_df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(temp_df['datetime']):
        temp_df['datetime'] = pd.to_datetime(temp_df['datetime'])
    
    temp_df['year_month'] = temp_df['datetime'].dt.strftime('%Y-%m')
    
    result = []
    for (source_a, source_b), group in temp_df.groupby(['source_a', 'source_b']):
        for year_month, month_group in group.groupby('year_month'):
            if metric_type == 'pearson':
                mean_value = calculate_avg_with_fisher(month_group[metric_type].values)
                q3_a_mean = calculate_avg_with_fisher(month_group[f'{metric_type}_q3_a'].values) 
                q3_b_mean = calculate_avg_with_fisher(month_group[f'{metric_type}_q3_b'].values)
            elif metric_type == 'r2':
                r_values = month_group['pearson_r'].values
                mean_value = calculate_avg_with_fisher(r_values) ** 2
                q3_a_mean = np.nanmean(month_group[f'{metric_type}_q3_a'].values)
                q3_b_mean = np.nanmean(month_group[f'{metric_type}_q3_b'].values)
            else:
                mean_value = np.nanmean(month_group[metric_type].values)
                q3_a_mean = np.nanmean(month_group[f'{metric_type}_q3_a'].values)
                q3_b_mean = np.nanmean(month_group[f'{metric_type}_q3_b'].values)
            
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

def calculate_monthly_q3_by_source(df, metric_type):
    
    if 'datetime' not in df.columns:
        return pd.DataFrame()
        
    temp_df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(temp_df['datetime']):
        temp_df['datetime'] = pd.to_datetime(temp_df['datetime'])
    
    temp_df['year_month'] = temp_df['datetime'].dt.strftime('%Y-%m')
    temp_df['month'] = temp_df['datetime'].dt.month
    temp_df['year'] = temp_df['datetime'].dt.year
    
    monthly_q3_stats = defaultdict(lambda: defaultdict(list))
    
    for _, row in temp_df.iterrows():
        source_a, source_b = row['source_a'], row['source_b']
        year_month = row['year_month']
        month = row['month']
        year = row['year']
        
        if not np.isnan(row[f'{metric_type}_q3_a']):
            key = (source_a, year, month)
            monthly_q3_stats[key].append(row[f'{metric_type}_q3_a'])
        
        if not np.isnan(row[f'{metric_type}_q3_b']):
            key = (source_b, year, month)
            monthly_q3_stats[key].append(row[f'{metric_type}_q3_b'])
    
    result = []
    for (source, year, month), values in monthly_q3_stats.items():
        if metric_type == 'pearson':
            mean_value = calculate_avg_with_fisher(values)
        elif metric_type == 'r2':
            r_values = np.sqrt([v for v in values if not np.isnan(v)])
            mean_value = calculate_avg_with_fisher(r_values) ** 2 if len(r_values) > 0 else np.nan
        else:
            mean_value = np.nanmean(values)
            
        result.append({
            'source': source,
            'year': year,
            'month': month,
            'year_month': f"{year}-{month:02d}",
            f'{metric_type}_q3_mean': mean_value,
            'count': len(values)
        })
    
    return pd.DataFrame(result) if result else pd.DataFrame()

def calculate_monthly_general_metrics_by_source(df, metric_type):
    
    if 'datetime' not in df.columns:
        return pd.DataFrame()
        
    temp_df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(temp_df['datetime']):
        temp_df['datetime'] = pd.to_datetime(temp_df['datetime'])
    
    temp_df['year_month'] = temp_df['datetime'].dt.strftime('%Y-%m')
    temp_df['month'] = temp_df['datetime'].dt.month
    temp_df['year'] = temp_df['datetime'].dt.year
    
    monthly_general_metrics = defaultdict(list)
    
    processed_files = {}
    
    for _, row in temp_df.iterrows():
        source_a, source_b = row['source_a'], row['source_b']
        year = row['year']
        month = row['month']
        file_name = row['filename_a']
        
        if not np.isnan(row[metric_type]):
            file_key = f"{file_name}_{source_a}_{year}_{month}"
            if file_key not in processed_files:
                key = (source_a, year, month)
                monthly_general_metrics[key].append(row[metric_type])
                processed_files[file_key] = True
        
        if not np.isnan(row[metric_type]):
            file_key = f"{file_name}_{source_b}_{year}_{month}"
            if file_key not in processed_files:
                key = (source_b, year, month)
                monthly_general_metrics[key].append(row[metric_type])
                processed_files[file_key] = True
    
    result = []
    for (source, year, month), values in monthly_general_metrics.items():
        if metric_type == 'pearson':
            mean_value = calculate_avg_with_fisher(values)
        elif metric_type == 'r2':
            r_values = np.sqrt([v for v in values if v >= 0 and not np.isnan(v)])
            mean_value = calculate_avg_with_fisher(r_values) ** 2 if len(r_values) > 0 else np.nan
        else:
            mean_value = np.nanmean(values)
            
        result.append({
            'source': source,
            'year': year,
            'month': month,
            'year_month': f"{year}-{month:02d}",
            f'{metric_type}_mean': mean_value,
            'count': len(values)
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
    parser.add_argument('--ssim-use-fisher', action='store_true',
                    help='Use Fisher Z-transformation when averaging SSIM values')
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
    swap_ytrue_ypred = args.swap_ytrue_ypred
    normalize_residuals = args.normalize_residuals
    verbose = args.verbose
    sample_debug = args.sample_debug
    check_images = args.check_images
    debug_skipped = args.debug_skipped
    debug_file = args.debug_file
    use_fisher_for_ssim = args.ssim_use_fisher
    
    higher_is_better = metric_type in ['pearson', 'r2', 'cosine', 'ssim']
    
    base_datasets = {
            'embrace': [
                'Seasonal_EMBRACE_TEC_maps'
            ],
            'igs': [
                'Seasonal_IGS_TEC_maps'
            ],
            'maggia': [
                'Seasonal_MAGGIA_TEC_maps'
            ],
            'nagoya': [
                'Seasonal_Nagoya_TEC_maps'
            ]
        }
    datasets = {source: [f"{dataset}_{dataset_suffix}" for dataset in dataset_list] 
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
    
    print(f"Calculating {metric_type.upper()} metrics with Q3-based value selection...")
    print(f"Using dataset suffix: '{dataset_suffix}'")
    print(f"Higher values are better: {'YES' if higher_is_better else 'NO'}")
    if metric_type == 'r2' and swap_ytrue_ypred:
        print("Swapping y_true and y_pred for R² calculation")
    if metric_type in ['residual', 'max_residual', 'min_residual'] and normalize_residuals:
        print("Normalizing y_pred for residual calculations")
    
    file_extension = '*.png' if metric_type == 'ssim' else '*.npy'
    
    all_expected_dirs = [base_dir / dataset for source in datasets for dataset in datasets[source]]
    existing_dirs = [d for d in all_expected_dirs if d.is_dir() and any(d.glob(file_extension))]
    
    if not existing_dirs:
        print(f"No valid directories found with suffix '{dataset_suffix}' containing {file_extension} files in {base_dir}")
        exit(1)
    
    print(f"Found {len(existing_dirs)} directories with the expected suffix and {file_extension} files:")
    for d in existing_dirs:
        print(f"  - {d.name}")
    
    if check_images:
        print("\nRunning image loading check for a sample of images...")
        for d in existing_dirs:
            sample_files = list(d.glob(file_extension))[:5]  
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
    
    debug_log = None
    if debug_skipped:
        debug_log = open(debug_file, 'w', encoding='utf-8')
        debug_log.write(f"Debug log for {metric_type} metric calculation with dataset suffix '{dataset_suffix}'\n")
        debug_log.write("=" * 80 + "\n\n")
    
    for comparison in comparisons:
        source_a, source_b = comparison
        if not datasets[source_a] or not datasets[source_b]:
            continue
        
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
                    
                    metric_q3_a = np.nan
                    metric_q3_b = np.nan
                    
                    if valid_q3_a and metric_type == 'ssim':
                        metric_q3_a = calculate_ssim(y_true_2d, y_pred_2d, mask_a_q3, verbose=file_verbose, 
                                 calculation_context=f"Q3 Mask ({source_a})")
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
                        metric_q3_b = calculate_ssim(y_true_2d, y_pred_2d, mask_b_q3, verbose=file_verbose, 
                                 calculation_context=f"Q3 Mask ({source_b})")
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
                        value_p = metric_value * 100
                    elif metric_type == 'mse' and not np.isnan(metric_value) and stats['data_range'] != 0:
                        value_p = (metric_value / (stats['data_range'] ** 2) * 100)
                    elif metric_type in ['rmse', 'mae', 'residual', 'max_residual', 'min_residual', 'huber'] and not np.isnan(metric_value) and stats['data_range'] != 0:
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
    
    dataset_avg_metrics = {}
    
    q3_values_by_source = defaultdict(list)
    monthly_q3_by_source = defaultdict(list)
    
    processed_monthly_files = {}
    
    for _, row in df.iterrows():
        source_a, source_b = row['source_a'], row['source_b']
        file_name = row['filename_a']
        
        if not np.isnan(row[f'{metric_type}_q3_a']):
            q3_values_by_source[source_a].append(row[f'{metric_type}_q3_a'])
            
            if 'datetime' in df.columns:
                date = pd.to_datetime(row['datetime'])
                year, month = date.year, date.month
                
                file_key = f"{file_name}_{year}_{month}"
                
                if (source_a, file_key) not in processed_monthly_files:
                    key = (source_a, year, month)
                    monthly_q3_by_source[key].append(row[f'{metric_type}_q3_a'])
                    processed_monthly_files[(source_a, file_key)] = True
        
        if not np.isnan(row[f'{metric_type}_q3_b']):
            q3_values_by_source[source_b].append(row[f'{metric_type}_q3_b'])
            
            if 'datetime' in df.columns:
                date = pd.to_datetime(row['datetime'])
                year, month = date.year, date.month
                
                file_key = f"{file_name}_{year}_{month}"
                
                if (source_b, file_key) not in processed_monthly_files:
                    key = (source_b, year, month)
                    monthly_q3_by_source[key].append(row[f'{metric_type}_q3_b'])
                    processed_monthly_files[(source_b, file_key)] = True
    
    monthly_q3_metrics = []
    
    for (source, year, month), values in monthly_q3_by_source.items():
        if metric_type == 'pearson':
            mean_value = calculate_avg_with_fisher(values)
        elif metric_type == 'r2':
            r_values = np.sqrt([v for v in values if not np.isnan(v)])
            mean_value = calculate_avg_with_fisher(r_values) ** 2 if len(r_values) > 0 else np.nan
        else:
            mean_value = np.nanmean(values)
        
        comparison_count = len(values)
            
        monthly_q3_metrics.append({
            'source': source,
            'year': year,
            'month': month,
            'year_month': f"{year}-{month:02d}",
            f'{metric_type}_q3_mean': mean_value,
            'count': comparison_count
        })
    
    monthly_general_metrics = None
    if 'datetime' in df.columns:
        monthly_general_metrics = calculate_monthly_general_metrics_by_source(df, metric_type)
        if not monthly_general_metrics.empty:
            monthly_general_file = f'monthly_metrics_{metric_type}_general_by_source.csv'
            monthly_general_metrics.to_csv(monthly_general_file, index=False)
            print(f"Monthly general metrics by source exported to {monthly_general_file}")
    
    q3_avg_metrics = {}
    for source, values in q3_values_by_source.items():
        if not values:
            q3_avg_metrics[source] = np.nan
            continue
            
        if metric_type == 'pearson':
            q3_avg_metrics[source] = calculate_avg_with_fisher(values)
        elif metric_type == 'r2':
            r_values = np.sqrt([v for v in values if not np.isnan(v)])
            q3_avg_metrics[source] = calculate_avg_with_fisher(r_values) ** 2 if len(r_values) > 0 else np.nan
        else:
            q3_avg_metrics[source] = np.nanmean(values)
    
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
                    metric_value = calculate_avg_with_fisher(valid_metrics)
                elif metric_type == 'r2':
                    r_values = selection['pearson_r'].values
                    valid_r = r_values[~np.isnan(r_values)]
                    metric_value = calculate_avg_with_fisher(valid_r) ** 2
                else:
                    metric_value = np.nanmean(np.abs(valid_metrics)) if metric_type == 'residual' else np.nanmean(valid_metrics)
            else:
                metric_value = np.nan
            
            metric_q3_a_values = selection[f'{metric_type}_q3_a'].values
            metric_q3_b_values = selection[f'{metric_type}_q3_b'].values
            valid_q3_a = metric_q3_a_values[~np.isnan(metric_q3_a_values)]
            valid_q3_b = metric_q3_b_values[~np.isnan(metric_q3_b_values)]
            
            if metric_type == 'pearson':
                metric_q3_a_avg = calculate_avg_with_fisher(valid_q3_a)
                metric_q3_b_avg = calculate_avg_with_fisher(valid_q3_b)
            elif metric_type == 'r2' and 'pearson_r' in selection:
                r_q3_a = np.sqrt(valid_q3_a)  
                r_q3_b = np.sqrt(valid_q3_b)
                metric_q3_a_avg = calculate_avg_with_fisher(r_q3_a) ** 2
                metric_q3_b_avg = calculate_avg_with_fisher(r_q3_b) ** 2
            else:
                metric_q3_a_avg = np.nanmean(valid_q3_a) if len(valid_q3_a) > 0 else np.nan
                metric_q3_b_avg = np.nanmean(valid_q3_b) if len(valid_q3_b) > 0 else np.nan
            
            dataset_metrics[source_a].extend(valid_metrics)
            dataset_metrics[source_b].extend(valid_metrics)
            
            dataset_total_metrics[source_a] += np.nansum(metric_values)
            dataset_total_metrics[source_b] += np.nansum(metric_values)
            dataset_count[source_a] += len(valid_metrics)
            dataset_count[source_b] += len(valid_metrics)
            
            mean_min_a = selection['min_a'].min()
            mean_max_a = selection['max_a'].max() 
            mean_mean_a = selection['mean_a'].mean()
            mean_median_a = selection['median_a'].mean()
            
            mean_min_b = selection['min_b'].min()
            mean_max_b = selection['max_b'].max() 
            mean_mean_b = selection['mean_b'].mean()
            mean_median_b = selection['median_b'].mean()
            
            mean_min_both = selection['min_both'].min()
            mean_max_both = selection['max_both'].max()
            mean_mean_both = selection['mean_both'].mean()
            mean_median_both = selection['median_both'].mean()
            mean_data_range = mean_max_both - mean_min_both if not np.isnan(mean_max_both) and not np.isnan(mean_min_both) else np.nan

            print("\nStatistics for All Data:")
            print("Dataset A:")
            print(f"  Min: {mean_min_a:.4f}, Q1: {selection['q1_a'].mean():.4f}, Median: {mean_median_a:.4f}, Q3: {selection['q3_a'].mean():.4f}, Mean: {mean_mean_a:.4f}, Max: {mean_max_a:.4f}")
            print("Dataset B:")
            print(f"  Min: {mean_min_b:.4f}, Q1: {selection['q1_b'].mean():.4f}, Median: {mean_median_b:.4f}, Q3: {selection['q3_b'].mean():.4f}, Mean: {mean_mean_b:.4f}, Max: {mean_max_b:.4f}")
            print("Combined:")
            print(f"  Min: {mean_min_both:.4f}, Q1: {selection['q1_both'].mean():.4f}, Median: {mean_median_both:.4f}, Q3: {selection['q3_both'].mean():.4f}, Mean: {mean_mean_both:.4f}, Max: {mean_max_both:.4f}")
            print(f"  Data Range: {mean_data_range:.4f}")
            
            def format_percentage(value, data_range=None, is_normalized=False):
                if np.isnan(value):
                    return "NaN%"
                if is_normalized:  
                    return f"{value * 100:.2f}%"
                if data_range and data_range != 0: 
                    return f"{(value / data_range * 100):.2f}% of data range"
                return "NaN%" 
            
            def format_metric_with_percent(value, data_range=None, is_normalized=False, suffix=""):
                percent = format_percentage(value, data_range, is_normalized)
                return f"{value:.4f} ({percent}){suffix}"
            
            if metric_type in ['pearson', 'r2', 'ssim', 'cosine']:
                metric_percent = metric_value * 100 if not np.isnan(metric_value) else np.nan
                percent_suffix = "%"
                is_normalized = True
            elif metric_type == 'mse' and not np.isnan(metric_value) and mean_data_range != 0:
                metric_percent = (metric_value / (mean_data_range ** 2) * 100)
                percent_suffix = "% of squared data range"
                is_normalized = False
            elif metric_type in ['rmse', 'mae', 'residual', 'max_residual', 'min_residual', 'huber'] and not np.isnan(metric_value) and mean_data_range != 0:
                metric_percent = (metric_value / mean_data_range * 100)
                percent_suffix = "% of data range"
                is_normalized = False
            else:
                metric_percent = np.nan
                percent_suffix = "%"
                is_normalized = False
            
            percent_display = f"({metric_percent:.2f}{percent_suffix})" if not np.isnan(metric_percent) else "(NaN%)"
            
            if metric_type == 'pearson':
                print(f'Average Pearson Correlation: {metric_value:.4f} {percent_display} (Fisher Z applied)')
                print(f'Average Pearson (Q3 {source_a}): {format_metric_with_percent(metric_q3_a_avg, is_normalized=True)} (Fisher Z applied)')
                print(f'Average Pearson (Q3 {source_b}): {format_metric_with_percent(metric_q3_b_avg, is_normalized=True)} (Fisher Z applied)')
            elif metric_type == 'r2':
                print(f'Average R² Score: {metric_value:.4f} {percent_display} (Fisher Z applied on Pearson r)')
                print(f'Average R² (Q3 {source_a}): {format_metric_with_percent(metric_q3_a_avg, is_normalized=True)} (Fisher Z applied)')
                print(f'Average R² (Q3 {source_b}): {format_metric_with_percent(metric_q3_b_avg, is_normalized=True)} (Fisher Z applied)')
            elif metric_type == 'ssim':
                fisher_note = " (Fisher Z applied)" if use_fisher_for_ssim else ""
                print(f'Average Structural Similarity Index: {metric_value:.4f} {percent_display}{fisher_note}')
                print(f'Average SSIM (Q3 {source_a}): {format_metric_with_percent(metric_q3_a_avg, is_normalized=True)}{fisher_note}')
                print(f'Average SSIM (Q3 {source_b}): {format_metric_with_percent(metric_q3_b_avg, is_normalized=True)}{fisher_note}')
            elif metric_type == 'cosine':
                print(f'Average Cosine Similarity: {metric_value:.4f} {percent_display}')
                print(f'Average Cosine (Q3 {source_a}): {format_metric_with_percent(metric_q3_a_avg, is_normalized=True)}')
                print(f'Average Cosine (Q3 {source_b}): {format_metric_with_percent(metric_q3_b_avg, is_normalized=True)}')
            elif metric_type == 'rmse':
                print(f'Average RMSE: {metric_value:.4f} {percent_display}')
                print(f'Average RMSE (Q3 {source_a}): {format_metric_with_percent(metric_q3_a_avg, mean_data_range)}')
                print(f'Average RMSE (Q3 {source_b}): {format_metric_with_percent(metric_q3_b_avg, mean_data_range)}')
            elif metric_type == 'mse':
                print(f'Average Mean Squared Error: {metric_value:.4f} {percent_display}')
                print(f'Average MSE (Q3 {source_a}): {format_metric_with_percent(metric_q3_a_avg, mean_data_range**2)}')
                print(f'Average MSE (Q3 {source_b}): {format_metric_with_percent(metric_q3_b_avg, mean_data_range**2)}')
            elif metric_type == 'mae':
                print(f'Average Mean Absolute Error: {metric_value:.4f} {percent_display}')
                print(f'Average MAE (Q3 {source_a}): {format_metric_with_percent(metric_q3_a_avg, mean_data_range)}')
                print(f'Average MAE (Q3 {source_b}): {format_metric_with_percent(metric_q3_b_avg, mean_data_range)}')
            elif metric_type == 'residual':
                print(f'Average Mean Absolute Residual Error: {metric_value:.4f} {percent_display}')
                print(f'Average Residual (Q3 {source_a}): {format_metric_with_percent(metric_q3_a_avg, mean_data_range)}')
                print(f'Average Residual (Q3 {source_b}): {format_metric_with_percent(metric_q3_b_avg, mean_data_range)}')
            elif metric_type == 'max_residual':
                print(f'Average Maximum Residual Error: {metric_value:.4f} {percent_display}')
                print(f'Average Max Residual (Q3 {source_a}): {format_metric_with_percent(metric_q3_a_avg, mean_data_range)}')
                print(f'Average Max Residual (Q3 {source_b}): {format_metric_with_percent(metric_q3_b_avg, mean_data_range)}')
            elif metric_type == 'min_residual':
                suffix = f" ({min_residual_percentile}th percentile)"
                print(f'Average Minimum Residual Error{suffix}: {metric_value:.4f} {percent_display}')
                print(f'Average Min Residual (Q3 {source_a}): {format_metric_with_percent(metric_q3_a_avg, mean_data_range)}')
                print(f'Average Min Residual (Q3 {source_b}): {format_metric_with_percent(metric_q3_b_avg, mean_data_range)}')
            elif metric_type == 'huber':
                suffix = f" (delta={huber_delta})"
                print(f'Average Huber Loss{suffix}: {metric_value:.4f} {percent_display}')
                print(f'Average Huber Loss (Q3 {source_a}): {format_metric_with_percent(metric_q3_a_avg, mean_data_range)}')
                print(f'Average Huber Loss (Q3 {source_b}): {format_metric_with_percent(metric_q3_b_avg, mean_data_range)}')
            
            sorted_maps = selection.sort_values(by=f'{metric_type}_p', ascending=not higher_is_better).head(top_n)
            comp_key = f"{dataset_a} x {dataset_b}"
            top_maps_by_comparison[comp_key] = sorted_maps
            
            print(f"\nTop {top_n} Maps with Best {metric_type.upper()} Values (Sorted by Percentage):")
            print("-" * 120)
            for idx, row in enumerate(sorted_maps.itertuples(), 1):
                file_info = f"{row.filename_a} & {row.filename_b}" if metric_type == 'ssim' else row.filename_a
                
                is_normalized = metric_type in ['pearson', 'r2', 'cosine', 'ssim']
                
                if hasattr(row, f'{metric_type}_p') and not np.isnan(getattr(row, f'{metric_type}_p')):
                    if is_normalized:
                        metric_display = f"{getattr(row, metric_type):.4f} ({getattr(row, f'{metric_type}_p'):.2f}%)"
                    elif metric_type == 'mse':
                        metric_display = f"{getattr(row, metric_type):.4f} ({getattr(row, f'{metric_type}_p'):.2f}% of squared data range)"
                    else:
                        metric_display = f"{getattr(row, metric_type):.4f} ({getattr(row, f'{metric_type}_p'):.2f}% of data range)"
                else:
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
                
                print(f"{idx}. Data: {date_str}")
                print(f"   Comparação: {row.dataset_a} x {row.dataset_b}")
                print(f"   Arquivos: {file_info}")
                print(f"   {metric_type.upper()}: {metric_display}")
                print(f"   {metric_type.upper()} (Q3 {row.source_a}, >= {row.q3_a:.4f}): {q3_a_display}")
                print(f"   {metric_type.upper()} (Q3 {row.source_b}, >= {row.q3_b:.4f}): {q3_b_display}")
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
                            
                            month_metric = calculate_monthly_metrics(month_data, metric_type, 
                                      use_fisher_for_ssim=args.ssim_use_fisher)
                            
                            if metric_type == 'pearson':
                                month_q3_a_metric = calculate_avg_with_fisher(month_data[f'{metric_type}_q3_a'].values)
                                month_q3_b_metric = calculate_avg_with_fisher(month_data[f'{metric_type}_q3_b'].values)
                            elif metric_type == 'r2':
                                q3_a_values = month_data[f'{metric_type}_q3_a'].values
                                q3_b_values = month_data[f'{metric_type}_q3_b'].values
                                r_q3_a = np.sqrt(q3_a_values[~np.isnan(q3_a_values)])
                                r_q3_b = np.sqrt(q3_b_values[~np.isnan(q3_b_values)])
                                month_q3_a_metric = calculate_avg_with_fisher(r_q3_a) ** 2 if len(r_q3_a) > 0 else np.nan
                                month_q3_b_metric = calculate_avg_with_fisher(r_q3_b) ** 2 if len(r_q3_b) > 0 else np.nan
                            else:
                                month_q3_a_metric = np.nanmean(month_data[f'{metric_type}_q3_a'].values)
                                month_q3_b_metric = np.nanmean(month_data[f'{metric_type}_q3_b'].values)
                            
                            if metric_type in ['pearson', 'r2', 'cosine', 'ssim']:
                                month_percent = month_metric * 100 if not np.isnan(month_metric) else np.nan
                                percent_suffix = "%"
                            elif metric_type == 'mse' and not np.isnan(month_metric) and mean_data_range != 0:
                                month_percent = (month_metric / (mean_data_range ** 2) * 100)
                                percent_suffix = "% of squared data range"
                            elif metric_type in ['rmse', 'mae', 'residual', 'max_residual', 'min_residual', 'huber']:
                                month_percent = (month_metric / mean_data_range * 100) if not np.isnan(month_metric) and mean_data_range != 0 else np.nan
                                percent_suffix = "% of data range"

                            percent_display = f"({month_percent:.2f}{percent_suffix})" if not np.isnan(month_percent) else "(NaN%)"
                            fisher_note = " (Fisher Z applied)" if metric_type in ['pearson', 'r2'] else ""
                            print(f'{month_name}: {metric_type.upper()} = {month_metric:.4f} {percent_display}{fisher_note} - {len(month_data)} comparisons')
                            
                            if not np.isnan(month_q3_a_metric):
                                q3_a_percent = month_q3_a_metric * 100 if metric_type in ['pearson', 'r2', 'cosine', 'ssim'] else (month_q3_a_metric / mean_data_range * 100)
                                q3_a_display = f"({q3_a_percent:.2f}{percent_suffix})" if not np.isnan(q3_a_percent) else "(NaN%)"
                                print(f'  Q3 {source_a}: {month_q3_a_metric:.4f} {q3_a_display}{fisher_note}')
                                
                            if not np.isnan(month_q3_b_metric):
                                q3_b_percent = month_q3_b_metric * 100 if metric_type in ['pearson', 'r2', 'cosine', 'ssim'] else (month_q3_b_metric / mean_data_range * 100)
                                q3_b_display = f"({q3_b_percent:.2f}{percent_suffix})" if not np.isnan(q3_b_percent) else "(NaN%)"
                                print(f'  Q3 {source_b}: {month_q3_b_metric:.4f} {q3_b_display}{fisher_note}')
                            
    print("\nCalculating average metrics...")
    for dataset in dataset_metrics:
        valid_metrics = [v for v in dataset_metrics[dataset] if not np.isnan(v)]
        if valid_metrics:
            if metric_type == 'pearson':
                dataset_avg_metrics[dataset] = calculate_avg_with_fisher(valid_metrics)
            elif metric_type == 'r2':
                r_values = df[(df['source_a'] == dataset) | (df['source_b'] == dataset)]['pearson_r'].values
                valid_r = r_values[~np.isnan(r_values)]
                dataset_avg_metrics[dataset] = calculate_avg_with_fisher(valid_r) ** 2
            else:
                dataset_avg_metrics[dataset] = np.nanmean(valid_metrics)
        else:
            dataset_avg_metrics[dataset] = np.nan
    
    if q3_avg_metrics:
        print("\n===== SOURCE RANKING BY Q3 VALUES =====")
        print("(Using values where each source has Q3 >= 75th percentile)")
        
        sorted_sources = list(q3_avg_metrics.items())
        if higher_is_better:
            sorted_sources.sort(key=lambda x: float('-inf') if np.isnan(x[1]) else x[1], reverse=True)
        else:
            sorted_sources.sort(key=lambda x: float('inf') if np.isnan(x[1]) else x[1], reverse=False)
        
        print("\nSource Ranking by Q3 Values (from best to worst):")
        for i, (source, q3_val) in enumerate(sorted_sources, 1):
            if np.isnan(q3_val):
                print(f"{i}. {source}: NaN (no valid Q3 values)")
                continue
                
            if metric_type in ['pearson', 'r2', 'cosine', 'ssim']:
                percent = q3_val * 100
                percent_suffix = "%"
            else:
                data_ranges = df[(df['source_a'] == source) | (df['source_b'] == source)]['data_range'].values
                avg_data_range = np.nanmean(data_ranges) if len(data_ranges) > 0 else 1.0
                
                if metric_type == 'mse' and avg_data_range != 0:
                    percent = (q3_val / (avg_data_range ** 2) * 100)
                    percent_suffix = "% of squared data range"
                elif avg_data_range != 0:
                    percent = (q3_val / avg_data_range * 100)
                    percent_suffix = "% of data range"
                else:
                    percent = np.nan
                    percent_suffix = "%"
            
            q3_count = len(q3_values_by_source[source])
            percent_display = f"({percent:.2f}{percent_suffix})" if not np.isnan(percent) else "(NaN%)"
            fisher_note = " (Fisher Z applied)" if metric_type in ['pearson', 'r2'] else ""
            print(f"{i}. {source}: {q3_val:.4f} {percent_display}{fisher_note} - based on {q3_count} Q3 comparisons")
        
        print("\nComparison of Regular vs. Q3-based Metrics:")
        print("-" * 80)
        print(f"{'Source':<10} | {'Regular':<25} | {'Q3-based':<25}")
        print("-" * 80)
        for source in dataset_metrics.keys():
            reg_val = dataset_avg_metrics[source]
            q3_val = q3_avg_metrics[source]
            
            if np.isnan(reg_val):
                reg_display = "NaN"
            else:
                if metric_type in ['pearson', 'r2', 'cosine', 'ssim']:
                    reg_percent = reg_val * 100
                    reg_display = f"{reg_val:.4f} ({reg_percent:.2f}%)"
                else:
                    reg_display = f"{reg_val:.4f}"
                    
            if np.isnan(q3_val):
                q3_display = "NaN"
            else:
                if metric_type in ['pearson', 'r2', 'cosine', 'ssim']:
                    q3_percent = q3_val * 100
                    q3_display = f"{q3_val:.4f} ({q3_percent:.2f}%)"
                else:
                    q3_display = f"{q3_val:.4f}"
            
            print(f"{source:<10} | {reg_display:<25} | {q3_display:<25}")
        print("-" * 80)
        fisher_note = "(Fisher Z applied for Pearson correlations)" if metric_type in ['pearson', 'r2'] else ""
        print(f"Note: {fisher_note}")
    
    print("\n===== PAIRS COMPARISON =====")
    
    pair_metrics = calculate_pair_stats(df, metric_type, use_fisher_for_ssim=args.ssim_use_fisher)  

    for pair_key, data in pair_metrics.items():
        metric_val = data['metric_value']
        if np.isnan(metric_val):
            data['percent'] = np.nan
            continue
            
        if metric_type in ['pearson', 'r2', 'cosine', 'ssim']:
            data['percent'] = metric_val * 100
        else:
            source_a, source_b = pair_key.split(' x ')
            pair_data_ranges = df[(df['source_a'] == source_a) & (df['source_b'] == source_b)]['data_range'].values
            avg_data_range = np.nanmean(pair_data_ranges) if len(pair_data_ranges) > 0 else 1.0
            
            if metric_type == 'mse' and avg_data_range != 0:
                data['percent'] = (metric_val / (avg_data_range ** 2) * 100)
            elif avg_data_range != 0:
                data['percent'] = (metric_val / avg_data_range * 100)
            else:
                data['percent'] = np.nan

    sorted_pairs = list(pair_metrics.items())
    if higher_is_better:
        sorted_pairs.sort(key=lambda x: float('-inf') if np.isnan(x[1]['percent']) else x[1]['percent'], reverse=True)
    else:
        sorted_pairs.sort(key=lambda x: float('inf') if np.isnan(x[1]['percent']) else x[1]['percent'], reverse=False)

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
    
    if q3_avg_metrics:
        q3_data = []
        for source, q3_val in q3_avg_metrics.items():
            if metric_type in ['pearson', 'r2', 'cosine', 'ssim'] and not np.isnan(q3_val):
                q3_percent = q3_val * 100
            else:
                data_ranges = df[(df['source_a'] == source) | (df['source_b'] == source)]['data_range'].values
                avg_data_range = np.nanmean(data_ranges) if len(data_ranges) > 0 else 1.0
                
                if metric_type == 'mse' and not np.isnan(q3_val) and avg_data_range != 0:
                    q3_percent = (q3_val / (avg_data_range ** 2) * 100)
                elif not np.isnan(q3_val) and avg_data_range != 0:
                    q3_percent = (q3_val / avg_data_range * 100)
                else:
                    q3_percent = np.nan
                    
            q3_data.append({
                'source': source,
                f'{metric_type}_q3_value': q3_val,
                f'{metric_type}_q3_percent': q3_percent,
                'q3_count': len(q3_values_by_source[source]),
                f'{metric_type}_regular_value': dataset_avg_metrics[source]
            })
        
        if q3_data:
            q3_df = pd.DataFrame(q3_data)
            q3_output_file = f'source_q3_metrics_{metric_type}.csv'
            q3_df.to_csv(q3_output_file, index=False)
            print(f"Source Q3 metrics exported to {q3_output_file}")
    
    if monthly_q3_metrics:
        monthly_q3_df = pd.DataFrame(monthly_q3_metrics)
        monthly_q3_output_file = f'monthly_q3_metrics_{metric_type}_by_source.csv'
        monthly_q3_df.to_csv(monthly_q3_output_file, index=False)
        print(f"Monthly Q3 metrics by source exported to {monthly_q3_output_file}")
    
    if 'datetime' in df.columns:
        print("\n===== TEMPORAL ANALYSIS BY PAIR =====")
        
        temporal_df = calculate_temporal_stats(df, metric_type)
        
        if not temporal_df.empty:
            try:
                temporal_file = f'temporal_analysis_{metric_type}_with_q3.csv'
                temporal_df.to_csv(temporal_file, index=False)
                print(f"Temporal analysis exported to {temporal_file}")
            except Exception as e:
                print(f"Error in temporal analysis export: {str(e)}")

    print("\n===== SOURCE ANALYSIS =====")
    
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
        
        if source in q3_avg_metrics and not np.isnan(q3_avg_metrics[source]):
            q3_val = q3_avg_metrics[source]
            if metric_type in ['pearson', 'r2', 'cosine', 'ssim']:
                q3_percent = q3_val * 100
                q3_percent_suffix = "%"
            else:
                if metric_type == 'mse' and avg_data_range != 0:
                    q3_percent = (q3_val / (avg_data_range ** 2) * 100)
                    q3_percent_suffix = "% of squared data range"
                elif avg_data_range != 0:
                    q3_percent = (q3_val / avg_data_range * 100)
                    q3_percent_suffix = "% of data range"
                else:
                    q3_percent = np.nan
                    q3_percent_suffix = "%"
            
            q3_percent_display = f"({q3_percent:.2f}{q3_percent_suffix})" if not np.isnan(q3_percent) else "(NaN%)"
            q3_count = len(q3_values_by_source[source])
            print(f"Average {metric_type} (Q3 values only): {q3_val:.4f} {q3_percent_display}{fisher_note} (based on {q3_count} Q3 comparisons)")
        else:
            print(f"Average {metric_type} (Q3 values only): No valid Q3 values available")
        
        if 'datetime' in df.columns:
            print("\n  Monthly Aggregated Values:")
            
            years_months = set()
            
            for key in monthly_q3_by_source.keys():
                if key[0] == source: 
                    years_months.add((key[1], key[2]))
            
            if monthly_general_metrics is not None and not monthly_general_metrics.empty:
                for _, row in monthly_general_metrics.iterrows():
                    if row['source'] == source:
                        years_months.add((row['year'], row['month']))
            
            for year, month in sorted(years_months):
            
                total_comparison_count = 0
                source_pairs_this_month = []
                
                for pair in source_pairs:
                    src_a, src_b = pair.split(' x ')
                    pair_data = df[(df['source_a'] == src_a) & (df['source_b'] == src_b)]
                    
                    if 'datetime' in pair_data.columns:
                        pair_data = pair_data[
                            (pair_data['datetime'].dt.year == year) & 
                            (pair_data['datetime'].dt.month == month)
                        ]
                        
                        if not pair_data.empty:
                            total_comparison_count += len(pair_data)
                            source_pairs_this_month.append(pair)
                
                general_value = None
                
                if monthly_general_metrics is not None and not monthly_general_metrics.empty:
                    
                    month_metrics = monthly_general_metrics[
                        (monthly_general_metrics['source'] == source) & 
                        (monthly_general_metrics['year'] == year) & 
                        (monthly_general_metrics['month'] == month)
                    ]
                    
                    if not month_metrics.empty:
                        general_value = month_metrics.iloc[0][f'{metric_type}_mean']
                
                if general_value is not None and not np.isnan(general_value):
                    if metric_type in ['pearson', 'r2', 'cosine', 'ssim']:
                        percent = general_value * 100
                        percent_suffix = "%"
                    else:
                        data_ranges = []
                        for pair in source_pairs_this_month:
                            src_a, src_b = pair.split(' x ')
                            month_data = df[
                                (df['source_a'] == src_a) & 
                                (df['source_b'] == src_b) &
                                (df['datetime'].dt.year == year) & 
                                (df['datetime'].dt.month == month)
                            ]
                            if not month_data.empty:
                                data_ranges.extend(month_data['data_range'].values)
                        
                        avg_data_range = np.nanmean(data_ranges) if data_ranges else 1.0
                        
                        if metric_type == 'mse' and avg_data_range > 0:
                            percent = (general_value / (avg_data_range ** 2) * 100)
                            percent_suffix = "% of squared data range"
                        elif avg_data_range > 0:
                            percent = (general_value / avg_data_range * 100)
                            percent_suffix = "% of data range"
                        else:
                            percent = np.nan
                            percent_suffix = "%"
                    
                    percent_display = f"({percent:.2f}{percent_suffix})" if not np.isnan(percent) else "(NaN%)"
                    fisher_note = " (Fisher Z applied)" if metric_type in ['pearson', 'r2'] else ""
                    
                    print(f"    {month}/{year}/{source}: {general_value:.4f} {percent_display}{fisher_note} (across {total_comparison_count} comparisons)")
                
                q3_value = None
                
                key = (source, year, month)
                
                if key in monthly_q3_by_source:
                    for metric in monthly_q3_metrics:
                        if metric['source'] == source and metric['year'] == year and metric['month'] == month:
                            q3_value = metric[f'{metric_type}_q3_mean']
                            break
                
                if q3_value is not None and not np.isnan(q3_value):
                    if metric_type in ['pearson', 'r2', 'cosine', 'ssim']:
                        percent = q3_value * 100
                        percent_suffix = "%"
                    else:
                        if metric_type == 'mse' and avg_data_range > 0:
                            percent = (q3_value / (avg_data_range ** 2) * 100)
                            percent_suffix = "% of squared data range"
                        elif avg_data_range > 0:
                            percent = (q3_value / avg_data_range * 100)
                            percent_suffix = "% of data range"
                        else:
                            percent = np.nan
                            percent_suffix = "%"
                    
                    percent_display = f"({percent:.2f}{percent_suffix})" if not np.isnan(percent) else "(NaN%)"
                    fisher_note = " (Fisher Z applied)" if metric_type in ['pearson', 'r2'] else ""
                    
                    print(f"    {month}/{year}/Q3/{source}: {q3_value:.4f} {percent_display}{fisher_note} (across {total_comparison_count} comparisons)")
        
        print("\n  Comparisons:")
        
        for pair in source_pairs:
            metric_val = pair_metrics[pair]['metric_value']
            percent = pair_metrics[pair]['percent']
            count = pair_metrics[pair]['count']
            
            s1, s2 = pair.split(' x ')
            
            if s1 != source:
                display_pair = f"{source} x {s1}"
                other_source = s1
            else:
                display_pair = pair
                other_source = s2
            
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
            
            if 'datetime' in df.columns:
                source_a, source_b = pair.split(' x ')
                
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
                    if not pd.api.types.is_datetime64_any_dtype(pair_data['datetime']):
                        pair_data['datetime'] = pd.to_datetime(pair_data['datetime'])
                    
                    pair_data = pair_data.sort_values('datetime')
                    
                    pair_data['year'] = pair_data['datetime'].dt.year
                    pair_data['month'] = pair_data['datetime'].dt.month
                    
                    print("    Monthly breakdown:")
                    
                    for year in sorted(pair_data['year'].unique()):
                        for month in sorted(pair_data[pair_data['year'] == year]['month'].unique()):
                            month_data = pair_data[(pair_data['year'] == year) & (pair_data['month'] == month)]
                            month_count = len(month_data)
                            
                            if month_count > 0:
                                month_name = f"{month}/{year}"
                                
                                month_metric = calculate_monthly_metrics(month_data, metric_type)
                                
                                if metric_type == 'pearson':
                                    month_q3_a_metric = calculate_avg_with_fisher(month_data[f'{metric_type}_q3_a'].values)
                                    month_q3_b_metric = calculate_avg_with_fisher(month_data[f'{metric_type}_q3_b'].values)
                                elif metric_type == 'r2':
                                    q3_a_values = month_data[f'{metric_type}_q3_a'].values
                                    q3_b_values = month_data[f'{metric_type}_q3_b'].values
                                    r_q3_a = np.sqrt(q3_a_values[~np.isnan(q3_a_values)])
                                    r_q3_b = np.sqrt(q3_b_values[~np.isnan(q3_b_values)])
                                    month_q3_a_metric = calculate_avg_with_fisher(r_q3_a) ** 2 if len(r_q3_a) > 0 else np.nan
                                    month_q3_b_metric = calculate_avg_with_fisher(r_q3_b) ** 2 if len(r_q3_b) > 0 else np.nan
                                else:
                                    month_q3_a_metric = np.nanmean(month_data[f'{metric_type}_q3_a'].values)
                                    month_q3_b_metric = np.nanmean(month_data[f'{metric_type}_q3_b'].values)
                                
                                month_data_range = np.nanmean(month_data['data_range'].values)
                                
                                if metric_type in ['pearson', 'r2', 'cosine', 'ssim']:
                                    month_percent = month_metric * 100 if not np.isnan(month_metric) else np.nan
                                    q3_a_percent = month_q3_a_metric * 100 if not np.isnan(month_q3_a_metric) else np.nan
                                    q3_b_percent = month_q3_b_metric * 100 if not np.isnan(month_q3_b_metric) else np.nan
                                    
                                    percent_suffix = "%"
                                elif metric_type == 'mse':
                                    month_percent = (month_metric / (month_data_range ** 2) * 100) if not np.isnan(month_metric) and month_data_range > 0 else np.nan
                                    q3_a_percent = (month_q3_a_metric / (month_data_range ** 2) * 100) if not np.isnan(month_q3_a_metric) and month_data_range > 0 else np.nan
                                    q3_b_percent = (month_q3_b_metric / (month_data_range ** 2) * 100) if not np.isnan(month_q3_b_metric) and month_data_range > 0 else np.nan
                                    
                                    percent_suffix = "% of squared data range"
                                else:
                                    month_percent = (month_metric / month_data_range * 100) if not np.isnan(month_metric) and month_data_range > 0 else np.nan
                                    q3_a_percent = (month_q3_a_metric / month_data_range * 100) if not np.isnan(month_q3_a_metric) and month_data_range > 0 else np.nan
                                    q3_b_percent = (month_q3_b_metric / month_data_range * 100) if not np.isnan(month_q3_b_metric) and month_data_range > 0 else np.nan
                                    
                                    percent_suffix = "% of data range"
                                
                                month_percent_display = f"({month_percent:.2f}{percent_suffix})" if not np.isnan(month_percent) else "(NaN%)"
                                q3_a_percent_display = f"({q3_a_percent:.2f}{percent_suffix})" if not np.isnan(q3_a_percent) else "(NaN%)"
                                q3_b_percent_display = f"({q3_b_percent:.2f}{percent_suffix})" if not np.isnan(q3_b_percent) else "(NaN%)"
                                
                                fisher_note = " (Fisher Z applied)" if metric_type in ['pearson', 'r2'] else ""
                                print(f"      {month_name}: {month_metric:.4f} {month_percent_display}{fisher_note} - {month_count} comparisons")
                                
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
                                
                                if not np.isnan(q3_current_source_metric):
                                    print(f"        Q3 {current_source}: {q3_current_source_metric:.4f} {q3_current_source_display}{fisher_note}")
                                if not np.isnan(q3_other_source_metric):
                                    print(f"        Q3 {other_source}: {q3_other_source_metric:.4f} {q3_other_source_display}{fisher_note}")
    
    if 'datetime' in df.columns:
        temporal_df = calculate_temporal_stats(df, metric_type)
        
        if not temporal_df.empty:
            try:
                temporal_file = f'temporal_analysis_{metric_type}_with_q3.csv'
                temporal_df.to_csv(temporal_file, index=False)
                print(f"\nTemporal analysis exported to {temporal_file}")
            except Exception as e:
                print(f"\nError in temporal analysis export: {str(e)}")
            
    print(f"\n===== ANALYSIS COMPLETE =====")
    print(f"Total files processed: {processed_files}")
    print(f"Total pairs analyzed: {len(pair_metrics)}")
    print(f"Total files skipped: {skipped_files}")
    print(f"Total missing files: {total_missing}")
    print(f"Total error files: {total_errors}")
    print(f"Results saved to:")
    print(f"  - result_{metric_type}_with_stats.csv (all individual comparisons)")
    print(f"  - pair_metrics_{metric_type}.csv (pair summary metrics)")
    print(f"  - source_q3_metrics_{metric_type}.csv (source metrics based on Q3 values)")
    print(f"  - monthly_q3_metrics_{metric_type}_by_source.csv (monthly Q3 metrics by source)")
    print(f"  - monthly_metrics_{metric_type}_general_by_source.csv (monthly general metrics by source)")
    if 'datetime' in df.columns:
        print(f"  - temporal_analysis_{metric_type}_with_q3.csv (monthly metrics by pair)")
    if debug_skipped:
        print(f"  - {debug_file} (detailed log of missing and error files)")        

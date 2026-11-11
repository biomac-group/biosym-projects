import numpy as np
import pandas as pd
from scipy.io import loadmat
import os

def get_reference_data(speed, root_path='reference_data'):
    """
    Loads reference data (Fukuchi or Reference 0.8-1.6).
    Returns a dict with 'hip', 'knee', 'ankle', 'Fx', 'Fy'.
    Each key has 'mean' and 'std'.
    
    Speeds in [0.6, 0.8, 1.0, 1.2, 1.4, 1.6] use the consolidated NPZ file.
    Speeds > 1.6 (e.g., 2.5, 3.5, 4.5) use the converted Fukuchi files.
    """
    res = {}
    
    # Selection logic for speed
    available_speeds = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
    
    # 1. Check if speed matches consolidated experimental data (0.6 - 1.6)
    if any(abs(speed - s) < 0.05 for s in available_speeds) or abs(speed - 1.33) < 0.02:
        npz_path = os.path.join(root_path, 'gait_speed_mean_var.npz')
        
        # Special case: 1.33 often refers to the Falisse 2019 reference
        if abs(speed - 1.33) < 0.02:
            # Try various potential locations for Falisse mat
            falisse_paths = [
                os.path.join(root_path, 'Falisse_133_noMet.mat'),
                'Falisse_133_noMet.mat',
                # Fallback to absolute path if known, but try to stay relative
            ]
            
            for f_path in falisse_paths:
                if os.path.exists(f_path):
                    try:
                        data = loadmat(f_path, squeeze_me=True, struct_as_record=False)
                        # The file might have the data in 'ExperimentalData' or 'ans'
                        exp_data = getattr(data, 'ExperimentalData', None)
                        if exp_data is None and hasattr(data, 'ans'):
                            exp_data = data['ans']
                            # If it's simplify_cells style
                            if isinstance(exp_data, dict) and 'ExperimentalData' in exp_data:
                                exp_data = exp_data['ExperimentalData']
                        
                        if exp_data:
                            # Try to navigate the structure subject1.Qs
                            # This handles both struct_as_record and simplified dicts
                            if hasattr(exp_q := getattr(getattr(exp_data, 'Q', exp_data.get('Q', {})), 'subject1', {}), 'Qs'):
                                q_obj = exp_q.Qs
                            elif 'Q' in exp_data and 'subject1' in exp_data['Q'] and 'Qs' in exp_data['Q']['subject1']:
                                q_obj = exp_q = exp_data['Q']['subject1']['Qs']
                            else:
                                continue

                            # [6, 9, 10] are Hip, Knee, Ankle Flexion
                            mean_q = getattr(q_obj, 'mean', q_obj.get('mean'))
                            std_q = getattr(q_obj, 'std', q_obj.get('std'))
                            
                            res['hip'] = {'mean': mean_q[:, 0], 'std': std_q[:, 0]}
                            res['knee'] = {'mean': -mean_q[:, 1], 'std': std_q[:, 1]} 
                            res['ankle'] = {'mean': mean_q[:, 2], 'std': std_q[:, 2]}
                            
                            # GRF data
                            grf_obj = getattr(getattr(exp_data, 'GRF', exp_data.get('GRF', {})), 'subject1', {})
                            mean_f = getattr(grf_obj, 'mean', grf_obj.get('mean'))
                            std_f = getattr(grf_obj, 'std', grf_obj.get('std'))
                            
                            res['Fx'] = {'mean': mean_f[:, 0] / 9.81, 'std': std_f[:, 0] / 9.81}
                            res['Fy'] = {'mean': mean_f[:, 1] / 9.81, 'std': std_f[:, 1] / 9.81}
                            
                            return res
                    except Exception:
                        continue
    
        if os.path.exists(npz_path):
            data_npz = np.load(npz_path, allow_pickle=True)
            closest_speed = min(available_speeds, key=lambda x: abs(x - speed))
            s_str = f"{closest_speed:.1f}".replace('.', '_')
            
            for joint, key_part in [('hip', 'hip_flexion'), ('knee', 'knee_angle'), ('ankle', 'ankle_angle')]:
                mean = data_npz[f'{s_str}_{key_part}_mean']
                var = data_npz[f'{s_str}_{key_part}_var']
                if joint == 'knee': mean = -mean
                res[joint] = {'mean': mean, 'std': np.sqrt(var)}
            
            for axis in ['x', 'y']:
                key = 'Fx' if axis == 'x' else 'Fy'
                mean = data_npz[f'{s_str}_F{axis}_mean'] / 9.81
                var = data_npz[f'{s_str}_F{axis}_var'] / (9.81**2)
                res[key] = {'mean': mean, 'std': np.sqrt(var)}
            return res

    # 2. Check for Fukuchi data (2.5, 3.5, 4.5)
    fukuchi_speeds = {2.5: 'T25', 3.5: 'T35', 4.5: 'T45'}
    closest_f_speed = min(fukuchi_speeds.keys(), key=lambda x: abs(x - speed))
    
    if abs(speed - closest_f_speed) < 0.04:
        # Try to load consolidated files first (usually higher quality/complete)
        # These are expected in root or root_path
        speed_int = int(closest_f_speed * 10)
        mean_file = os.path.join(root_path, f'Fukuchi_{speed_int}_mean.mat')
        var_file = os.path.join(root_path, f'Fukuchi_{speed_int}_var.mat')
        
        if os.path.exists(mean_file):
            try:
                m_data = loadmat(mean_file, squeeze_me=True, simplify_cells=True)['arr']
                v_data = loadmat(var_file, squeeze_me=True, simplify_cells=True)['arr'] if os.path.exists(var_file) else np.zeros_like(m_data)
                
                # Rows: 0=Hip, 1=Knee, 2=Ankle (usually)
                # But let's check max values to be sure
                # Hip flex is usually ~0.8-1.0 rad
                # Knee flex is usually ~1.5+ rad
                # Ankle is ~0.4-0.5 rad
                
                # Find which row is which based on max flexion
                maxs = np.max(np.abs(m_data), axis=1)
                knee_idx = np.argmax(maxs) # Knee has largest range
                other_indices = [i for i in range(3) if i != knee_idx]
                # Compare remaining two
                if np.max(np.abs(m_data[other_indices[0]])) > np.max(np.abs(m_data[other_indices[1]])):
                    hip_idx, ankle_idx = other_indices[0], other_indices[1]
                else:
                    hip_idx, ankle_idx = other_indices[1], other_indices[0]
                
                # All Fukuchi data needs unit conversion from radians to degrees
                res['hip'] = {'mean': m_data[hip_idx] * 180/np.pi, 'std': np.sqrt(v_data[hip_idx]) * 180/np.pi}
                res['knee'] = {'mean': m_data[knee_idx] * 180/np.pi, 'std': np.sqrt(v_data[knee_idx]) * 180/np.pi}
                res['ankle'] = {'mean': m_data[ankle_idx] * 180/np.pi, 'std': np.sqrt(v_data[ankle_idx]) * 180/np.pi}
                
                # GRF from processed files (kinetics are usually there)
                suffix = fukuchi_speeds[closest_f_speed]
                proc_dir = os.path.join(root_path, 'fukuchi_processed')
                if os.path.exists(proc_dir):
                    fx_means, fy_means = [], []
                    for f in [f for f in os.listdir(proc_dir) if f.endswith(f'{suffix}.mat')]:
                        d = loadmat(os.path.join(proc_dir, f), squeeze_me=True, simplify_cells=True)
                        if 'Fx' in d: fx_means.append(d['Fx']['mean'])
                        if 'Fy' in d: fy_means.append(d['Fy']['mean'])
                    if fx_means:
                        res['Fx'] = {'mean': np.nanmean(fx_means, axis=0), 'std': np.nanstd(fx_means, axis=0)}
                        res['Fy'] = {'mean': np.nanmean(fy_means, axis=0), 'std': np.nanstd(fy_means, axis=0)}
                return res
            except Exception:
                pass

        suffix = fukuchi_speeds[closest_f_speed]
        proc_dir = os.path.join(root_path, 'fukuchi_processed')
        
        # We need to average across subjects if we want a single global reference,
        # or just pick one. The request said "loads the fukuchi (mean+std at that speed)".
        # Let's average all processed subjects for that speed.
        if os.path.exists(proc_dir):
            all_files = [f for f in os.listdir(proc_dir) if f.endswith(f'{suffix}.mat')]
            if all_files:
                temp_data = {k: {'means': [], 'stds': []} for k in ['hip', 'knee', 'ankle', 'Fx', 'Fy']}
                for f in all_files:
                    path = os.path.join(proc_dir, f)
                    data = loadmat(path, squeeze_me=True, simplify_cells=True)
                    for k in temp_data.keys():
                        if k in data:
                            # Handle different ways means and stds might be stored
                            val = data[k]
                            if isinstance(val, dict):
                                temp_data[k]['means'].append(val.get('mean', val.get('means')))
                                std = val.get('std', val.get('stds', val.get('var')))
                                if std is not None:
                                    # If it was variance, take sqrt
                                    if 'var' in val and std is val['var']:
                                        temp_data[k]['stds'].append(np.sqrt(std))
                                    else:
                                        temp_data[k]['stds'].append(std)
                            else:
                                # Fallback if it's just the mean array
                                temp_data[k]['means'].append(val)
                
                for k in temp_data.keys():
                    if temp_data[k]['means']:
                        res[k] = {
                            'mean': np.nanmean(temp_data[k]['means'], axis=0),
                            'std': np.nanmean(temp_data[k]['stds'], axis=0) if temp_data[k]['stds'] else np.zeros(100)
                        }
                    else:
                        res[k] = {'mean': np.full(100, np.nan), 'std': np.full(100, np.nan)}
                return res

    return None

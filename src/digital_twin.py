import numpy as np
import tensorflow as tf

def run_digital_twin(model, initial_sequence, n_steps, output_frames=3):
    """
    Run the AI twin recursively.
    initial_sequence: (1, input_days, H, W, C) normalized
    n_steps: number of prediction steps (each step = output_frames days)
    Returns: (n_steps, output_frames, H, W, C) normalized
    """
    state = initial_sequence.copy()
    predictions = []
    for _ in range(n_steps):
        pred = model.predict(state, verbose=0)          # (1, 3, H, W, C)
        predictions.append(pred[0])
        # Shift state: drop oldest output_frames days, append prediction
        state = np.roll(state, -output_frames, axis=1)
        state[0, -output_frames:] = pred[0]
    return np.stack(predictions)   # (steps, 3, H, W, C)

def apply_scenario(init_seq, temp_offset=0.0, rain_scale=1.0,
                   norm_params=None, variables=None):
    """
    Modify the last input frame (the "current" state) with what-if parameters.
    init_seq: (1, 7, H, W, C) normalized
    temp_offset: °C change
    rain_scale: multiplier for rainfall
    Returns modified sequence.
    """
    modified = init_seq.copy()
    v = variables or ['rain','tmax','tmin']
    if norm_params is None:
        return modified
    # Denormalize -> apply -> renormalize for the last frame only
    last_frame = modified[0, -1].copy()  # (H, W, C)
    # Indices: 0=rain, 1=tmax, 2=tmin
    # Rain (scale)
    rain_min = norm_params['rain']['vmin']
    rain_max = norm_params['rain']['vmax']
    last_frame[..., 0] = np.clip(last_frame[..., 0] * (rain_max - rain_min) + rain_min, 0, None)
    last_frame[..., 0] *= rain_scale
    last_frame[..., 0] = (last_frame[..., 0] - rain_min) / (rain_max - rain_min)

    # Tmax (offset)
    tmax_min = norm_params['tmax']['vmin']
    tmax_max = norm_params['tmax']['vmax']
    last_frame[..., 1] = last_frame[..., 1] * (tmax_max - tmax_min) + tmax_min
    last_frame[..., 1] += temp_offset
    last_frame[..., 1] = np.clip(last_frame[..., 1], tmax_min, tmax_max)
    last_frame[..., 1] = (last_frame[..., 1] - tmax_min) / (tmax_max - tmax_min)

    # Tmin (offset)
    tmin_min = norm_params['tmin']['vmin']
    tmin_max = norm_params['tmin']['vmax']
    last_frame[..., 2] = last_frame[..., 2] * (tmin_max - tmin_min) + tmin_min
    last_frame[..., 2] += temp_offset
    last_frame[..., 2] = np.clip(last_frame[..., 2], tmin_min, tmin_max)
    last_frame[..., 2] = (last_frame[..., 2] - tmin_min) / (tmin_max - tmin_min)

    modified[0, -1] = last_frame
    return modified
import numpy as np


def flare_injection(flux: np.ndarray, cadence_minute: float, rng: np.random.Generator):
    n = len(flux)
    t = np.arange(n) * cadence_minute
    peak_idx = rng.integers(int(n * 0.2), int(n * 0.8))
    peak_time = t[peak_idx]

    amplitude = rng.uniform(0.005, 0.08)
    rise_min = rng.uniform(5, 20)
    decay_min = rng.uniform(30, 180)

    rise = np.exp(-(0.5 * ((t - peak_time) / rise_min) ** 2))
    decay = np.exp(-(t - peak_time) / decay_min)

    shape = np.where(t <= peak_time, rise, decay)
    shape = np.clip(shape, 0, None)

    output = flux + amplitude * shape
    meta = {
        "type": "flare",
        "amplitude": amplitude,
        "decay_min": decay_min,
        "peak_idx": int(peak_idx)
    }
    return output, meta


def artifact_injection(flux: np.ndarray, rng: np.random.Generator):
    n = len(flux)
    kind = rng.choice(["cosmic_ray", "spsd", "jump"])
    output = flux.copy()
    idx = rng.integers(int(0.1 * n), int(n * 0.9))

    if kind == "cosmic_ray":

        output[idx] += rng.uniform(0.03, 0.15) * rng.choice([-1, 1])
        meta = {"type": "artifact", "subtype": "cosmic_ray", "idx": int(idx)}

    elif kind == "spsd":
        drop = rng.uniform(0.01, 0.06)
        recovery_l = rng.integers(10, 40)

        end = min(n, idx + recovery_l)
        recovery = np.linspace(drop, 0, end - idx)

        output[idx:end] -= recovery
        meta = {"type": "artifact", "subtype": "spsd", "idx": int(idx), "recovery_len": int(recovery_l)}

    else:
        shift = rng.uniform(0.01, 0.05) * rng.choice([-1, 1])
        output[idx:] -= shift
        meta = {"type": "artifact", "subtype": "jump", "idx": int(idx), "shift": float(shift)}

    return output, meta


def variability_injection(flux, cadence_minute: float, rng: np.random.Generator):
    n = len(flux)
    t = np.arange(n) * (cadence_minute / 60.0)

    period = rng.uniform(12, 96)
    amplitude = rng.uniform(0.002, 0.02)
    phase = rng.uniform(0, 2 * np.pi)

    signal = amplitude * np.sin(((2 * np.pi * t) / period) + phase)

    output = flux + signal
    meta = {"type": "variability", "period": period, "amplitude": amplitude, "phase": phase}
    return output, meta


def inject(flux_2d: np.ndarray, kind: str, cadence_min: float, rng: np.random.Generator):

    injected_fluxes = []
    metadata_list = []

    for row in flux_2d:
        if kind == "flare":
            new_flux, meta = flare_injection(row, cadence_min, rng)
        elif kind == "artifact":
            new_flux, meta = artifact_injection(row, rng)
        elif kind == "variability":
            new_flux, meta = variability_injection(row, cadence_min, rng)
        else:
            raise ValueError(f"unknown injection kind: {kind}")

        injected_fluxes.append(new_flux)
        metadata_list.append(meta)

    return np.array(injected_fluxes), metadata_list
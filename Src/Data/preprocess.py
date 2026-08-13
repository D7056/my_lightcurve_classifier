import pandas as pd
from pathlib import Path
import numpy as np
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split, GroupShuffleSplit


from Src.Data.simulate_anomalies import inject

FLARE_RATE = 0.25
ARTIFACT_RATE = 0.20
VARIABILITY_RATE = 0.15
CADENCE_MINUTE = 29.4


def detrend(flux):
    if np.isnan(flux).mean() > 0.2:
        return None
    flux = pd.Series(flux).interpolate(limit_direction='both').to_numpy(dtype=float)
    w_length = 101 if len(flux) > 101 else (len(flux) // 2 * 2 + 1)
    trend = savgol_filter(flux, w_length, polyorder=2)
    return flux / trend


def find_centers(time_array, period, time0):
    end_time = time_array.max()
    num_orbits = int((end_time - time0) // period)
    return [time0 + (k * period) for k in range(0, num_orbits + 1)]


def cadence_transit(kepid, koi_df):
    file_path = f"Data/raw/KIC{kepid}.csv"
    target_row = koi_df[koi_df["kepid"] == int(kepid)]
    if len(target_row) == 0:
        return []

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        return []

    flux = df["flux"].to_numpy(dtype=float)
    time = df["time"].to_numpy(dtype=float)
    detrended_flux = detrend(flux)
    if detrended_flux is None:
        return []

    window_size = 201
    stride = 50
    half = window_size // 2
    chunks = []
    is_transit = np.zeros(len(time), dtype=bool)

    # FIX: Corrected iterrows unpacking
    for _, row in target_row.iterrows():
        period = float(row["koi_period"])
        time0 = float(row["koi_time0bk"])

        centers = find_centers(time, period, time0)

        for c in centers:
            if c < time.min() or c > time.max():
                continue
            idx = int(np.argmin(np.abs(time - c)))
            start_slice = idx - half
            end_slice = idx + half + 1
            if start_slice >= 0 and end_slice <= len(detrended_flux):
                # FIX: Added kepid tracking
                chunks.append({
                    "kepid": kepid,
                    "flux": detrended_flux[start_slice:end_slice],
                    "label": "transit"
                })
                is_transit[start_slice:end_slice] = True

    for k in range(0, len(detrended_flux) - window_size + 1, stride):
        window_mask = is_transit[k: k + window_size]
        if not np.any(window_mask):
            chunks.append({
                "kepid": kepid,
                "flux": detrended_flux[k: k + window_size],
                "label": "quiet"
            })
    return chunks


def find_eb_dip(detrended_flux, time):
    deepest_idx = np.argmin(detrended_flux)
    return time[deepest_idx]


def cadence_ebs(kepid, eb_df):
    file_path = f"Data/raw/KIC{kepid}.csv"
    target_row = eb_df[eb_df["kepid"] == int(kepid)]
    if len(target_row) == 0:
        return []

    period = float(target_row.iloc[0]["period"])

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        return []

    flux = df["flux"].to_numpy(dtype=float)
    time = df["time"].to_numpy(dtype=float)
    detrended_flux = detrend(flux)

    if detrended_flux is None:
        return []

    window_size = 201
    stride = 50
    half = window_size // 2

    time0 = find_eb_dip(detrended_flux, time)
    centers = find_centers(time, period, time0)
    chunks = []
    is_transit = np.zeros(len(time), dtype=bool)

    for c in centers:
        if c < time.min() or c > time.max():
            continue

        idx = int(np.argmin(np.abs(time - c)))
        start_slice = idx - half
        end_slice = idx + half + 1

        if start_slice >= 0 and end_slice <= len(detrended_flux):
            chunks.append({
                "kepid": kepid,
                "flux": detrended_flux[start_slice:end_slice],
                "label": "ebs"
            })
            is_transit[start_slice:end_slice] = True

    for k in range(0, len(detrended_flux) - window_size + 1, stride):
        window_mask = is_transit[k: k + window_size]
        if not np.any(window_mask):
            chunks.append({
                "kepid": kepid,
                "flux": detrended_flux[k: k + window_size],
                "label": "quiet"
            })
    return chunks


def cadence_stellar(kepid):
    file_path = f"Data/raw/KIC{kepid}.csv"
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        return []

    flux = df["flux"].to_numpy(dtype=float)
    detrended_flux = detrend(flux)

    if detrended_flux is None:
        return []

    window_size = 201
    stride = 50
    chunks = []

    for k in range(0, len(detrended_flux) - window_size + 1, stride):
        chunks.append({
            "kepid": kepid,
            "flux": detrended_flux[k: k + window_size],
            "label": "quiet"
        })
    return chunks


def stitch():
    print("Loading catalogs...")
    manifest = pd.read_csv("Data/raw/_target_manifest.csv")
    koi_df = pd.read_csv("Data/cumulative_koi_data.csv")
    eb_df = pd.read_csv("Data/eb_data.csv")

    transits = manifest[manifest["source"] == "transit"]
    ebs = manifest[manifest["source"] == "eclipsing_binary"]
    quiets = manifest[manifest["source"] == "quiet"]

    all_data = []

    print("Processing Transits...")
    for kepid in transits["kepid"]:
        all_data.extend(cadence_transit(kepid, koi_df))

    print("Processing Eclipsing Binaries...")
    for kepid in ebs["kepid"]:
        all_data.extend(cadence_ebs(kepid, eb_df))

    print("Processing Quiet Stars...")
    for kepid in quiets["kepid"]:
        all_data.extend(cadence_stellar(kepid))

    print("Executing Data Augmentation (Injection)...")
    final_dataset = pd.DataFrame(all_data)

    # 1. Isolate the quiet windows
    quiet_windows = final_dataset[final_dataset["label"] == "quiet"].copy()

    # 2. Keep the real transits and EBs safely to the side
    final_dataset = final_dataset[final_dataset["label"] != "quiet"].copy()

    # 3. Calculate exact integer numbers for our fractions based on the ORIGINAL total
    total_quiet = len(quiet_windows)
    n_flares = int(total_quiet * FLARE_RATE)
    n_artifacts = int(total_quiet * ARTIFACT_RATE)
    n_variability = int(total_quiet * VARIABILITY_RATE)

    # 4. Sample and drop safely
    flares = quiet_windows.sample(n=n_flares, random_state=42)
    quiet_windows = quiet_windows.drop(flares.index)

    artifacts = quiet_windows.sample(n=n_artifacts, random_state=42)
    quiet_windows = quiet_windows.drop(artifacts.index)

    variability = quiet_windows.sample(n=n_variability, random_state=42)
    quiet_windows = quiet_windows.drop(variability.index)

    # 5. Extract the FLUX arrays (stacked as a 2D NumPy array) to pass to inject()
    flares_flux = np.stack(flares["flux"].values)
    artifacts_flux = np.stack(artifacts["flux"].values)
    variability_flux = np.stack(variability["flux"].values)

    rng = np.random.default_rng(42)

    # 6. Inject the anomalies!

    sy_flares, meta_flare = inject(flares_flux, "flare", CADENCE_MINUTE, rng)
    sy_variability, meta_variability = inject(variability_flux, "variability", CADENCE_MINUTE, rng)
    sy_artifacts, meta_artifacts = inject(artifacts_flux, "artifact", CADENCE_MINUTE, rng)

    # 7. Convert the newly injected NumPy arrays back into Pandas DataFrames with correct labels
    flares_df = pd.DataFrame({
        "kepid": flares["kepid"].values,
        "flux": list(sy_flares),
        "label": "flare"
    })

    variability_df = pd.DataFrame({
        "kepid": variability["kepid"].values,
        "flux": list(sy_variability),
        "label": "variability"
    })

    artifacts_df = pd.DataFrame({
        "kepid": artifacts["kepid"].values,
        "flux": list(sy_artifacts),
        "label": "artifact"
    })


    final_dataset = pd.concat([
        final_dataset,
        quiet_windows,
        flares_df,
        variability_df,
        artifacts_df
    ], ignore_index=True)

    CLASSES_TO_IDX={
        "quiet":-1,
        "transit":0,
        "ebs":1,
        "flare":2,
        "variability":4,
        "artifact":3

    }



    final_dataset["y"]=final_dataset["label"].map(lambda x: CLASSES_TO_IDX.get(x,-1))

    X=np.stack(final_dataset["flux"].values).astype(np.float32)
    y=final_dataset['y'].to_numpy(dtype=np.int64)
    groups=final_dataset['kepid'].to_numpy()

    idx=np.arange(len(X))
    gss1=GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, temp_idx = next(gss1.split(idx, y, groups=groups))

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_rel, test_rel = next(gss2.split(temp_idx, y[temp_idx], groups=groups[temp_idx]))
    val_idx = temp_idx[val_rel]
    test_idx = temp_idx[test_rel]

    out_dir = Path("Data/Processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, sel in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        np.savez(out_dir / f"{name}.npz", X=X[sel], y=y[sel])
        print(f"{name}: {len(sel)} windows -> {out_dir / (name + '.npz')}")



    print(f"\nDataset complete! Total chunks extracted: {len(final_dataset)}")
    print(final_dataset["label"].value_counts())









if __name__ == "__main__":
    stitch()
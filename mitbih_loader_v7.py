# -*- coding: utf-8 -*-
"""
MIT-BIH Arrhythmia Database — Veri Yükleme Modülü v7
loader tarafında değişiklik yok
"""

import numpy as np
import wfdb
import os

CHRONIC_ARRHYTHMIA = {'107', '109', '111', '118'}

ALL_RECORDS = [
    '100','101','102','103','104','105','106',
    '108',            '112','113','114','115','116',
    '117',       '119','121','122','123','124','200',
    '201','202','203','205','207','208','209','210',
    '212','213','214','215','217','219','220','221',
    '222','223','228','230','231','232','233','234'
]

ARRHYTHMIA_SYMBOLS = {
    'V', 'A', 'F', 'f', 'E', '!', 'x',
}

WINDOW_SEC       = 5
STRIDE_SEC       = 3
TARGET_STEPS     = 60
FS               = 360
ARRHYTHMIA_RATIO = 0.20


def _resample(signal, target_len):
    if len(signal) == target_len:
        return signal.astype(np.float32)
    idx = np.linspace(0, len(signal) - 1, target_len)
    return np.interp(idx, np.arange(len(signal)), signal).astype(np.float32)


def _extract_features(segment):
    ch0   = segment[:, 0].astype(np.float64)
    ch1   = (segment[:, 1].astype(np.float64)
             if segment.shape[1] > 1 else ch0.copy())
    deriv = np.gradient(ch0)
    d_std = np.std(deriv)
    if d_std > 0:
        deriv = deriv / d_std
    return np.stack([
        _resample(ch0,   TARGET_STEPS),
        _resample(ch1,   TARGET_STEPS),
        _resample(deriv, TARGET_STEPS),
    ], axis=1).astype(np.float32)


def _arrhythmia_ratio(ann, win_start, win_end):
    total = aritmik = 0
    for i, s in enumerate(ann.sample):
        if win_start <= s < win_end:
            total += 1
            if ann.symbol[i] in ARRHYTHMIA_SYMBOLS:
                aritmik += 1
    return aritmik / total if total > 0 else 0.0


def load_mitbih(records=None, max_records=44, data_dir='mitbih',
                exclude_chronic=True, verbose=True):
    if records is None:
        pool = ALL_RECORDS[:]
        if not exclude_chronic:
            pool = ['107','109','111','118'] + pool
        records = pool[:max_records]
    if exclude_chronic:
        records = [r for r in records if r not in CHRONIC_ARRHYTHMIA]

    os.makedirs(data_dir, exist_ok=True)

    X_list, y_list, pid_list = [], [], []
    hr_base_list, spo2_base_list = [], []

    win_s    = int(WINDOW_SEC  * FS)
    stride_s = int(STRIDE_SEC  * FS)

    for p_idx, rec_name in enumerate(records):
        rec_path = os.path.join(data_dir, rec_name)
        try:
            record = wfdb.rdrecord(rec_path)
            ann    = wfdb.rdann(rec_path, 'atr')
        except Exception:
            if verbose:
                print(f"  Indiriliyor: {rec_name} ...", end=' ', flush=True)
            try:
                wfdb.dl_database('mitdb', dl_dir=data_dir, records=[rec_name])
                record = wfdb.rdrecord(rec_path)
                ann    = wfdb.rdann(rec_path, 'atr')
                if verbose: print("OK")
            except Exception as e:
                if verbose: print(f"HATA ({e})")
                continue

        signal = record.p_signal
        if signal is None or signal.ndim < 2:
            continue

        calib_end = min(2 * 60 * FS, len(signal))
        hr_base_list.append(float(np.nanmean(signal[:calib_end, 0])))
        spo2_base_list.append(
            float(np.nanmean(signal[:calib_end, 1]))
            if signal.shape[1] > 1
            else float(np.nanmean(signal[:calib_end, 0]))
        )

        n_wins = n_pos = 0
        start = 0
        while start + win_s <= len(signal):
            seg = signal[start:start + win_s]
            if not np.isnan(seg).any():
                ratio = _arrhythmia_ratio(ann, start, start + win_s)
                label = 1 if ratio >= ARRHYTHMIA_RATIO else 0
                X_list.append(_extract_features(seg))
                y_list.append(label)
                pid_list.append(p_idx)
                n_wins += 1
                if label == 1: n_pos += 1
            start += stride_s

        if verbose:
            pct = n_pos / max(n_wins, 1) * 100
            print(f"  [{p_idx+1:02d}/{len(records)}] Kayit {rec_name}: "
                  f"{n_wins} pencere, {n_pos} aritmik (%{pct:.0f})")

    if not X_list:
        raise RuntimeError("Hic veri yuklenemedi.")

    X          = np.array(X_list,         dtype=np.float32)
    y          = np.array(y_list,         dtype=np.float32)
    patient_id = np.array(pid_list,       dtype=int)
    hr_base    = np.array(hr_base_list,   dtype=np.float32)
    spo2_base  = np.array(spo2_base_list, dtype=np.float32)

    if verbose:
        print(f"\n  Toplam pencere : {len(X)}")
        print(f"  Normal  (y=0)  : {int(np.sum(y==0))}")
        print(f"  Aritmik (y=1)  : {int(np.sum(y==1))}")
        print(f"  Sinif dengesi  : %{np.sum(y==1)/len(y)*100:.1f} pozitif")

    return X, y, patient_id, hr_base, spo2_base


def normalize_mitbih(X, y, patient_id, hr_base, spo2_base):
    HR_SCALE   = 0.5
    SPO2_SCALE = 0.3
    ECG_SCALE  = 1.0

    X_norm = X.copy()
    for p in range(len(hr_base)):
        mask = (patient_id == p)
        if not np.any(mask): continue
        X_norm[mask, :, 0] = (X[mask, :, 0] - hr_base[p])   / HR_SCALE
        X_norm[mask, :, 1] = (X[mask, :, 1] - spo2_base[p]) / SPO2_SCALE
        X_norm[mask, :, 2] =  X[mask, :, 2]                  / ECG_SCALE

    return np.clip(X_norm, -4, 4)

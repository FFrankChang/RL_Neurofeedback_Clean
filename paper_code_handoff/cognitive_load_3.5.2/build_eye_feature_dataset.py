"""
原始眼动/EEG -> 12维眼动特征 + MWI 标签 数据集构建（论文5.2.3）
===========================================================

本脚本补全论文5.2.3的数据工程前半段：
1) 原始眼动质量控制（无效值、质量分数、异常范围）
2) 眨眼/丢帧处理（无效片段检测）
3) 线性插值 + 平滑
4) 2秒窗口、50%重叠分段
5) 提取12维特征：瞳孔8 + 注视2 + 扫视1 + 眨眼1
6) 与EEG窗口计算的MWI对齐，输出训练CSV

输出可直接供 arousal_classification.py 使用：
- 特征列: DEFAULT_FEATURE_COLS
- 标签列: mwi（连续）与 mwi_label（二值）
"""

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_FEATURE_COLS: List[str] = [
    "pupil_mean",
    "pupil_baseline",
    "pupil_rel_change",
    "pupil_velocity",
    "pupil_std",
    "pupil_mean_1s",
    "pupil_mean_5s",
    "pupil_trend",
    "fixation_duration_mean",
    "fixation_rate",
    "saccade_amplitude_mean",
    "blink_rate",
]


@dataclass
class PipelineConfig:
    window_sec: float = 2.0
    overlap: float = 0.5
    min_window_coverage: float = 0.6
    smooth_sec: float = 0.15
    pupil_quality_min: float = 0.5
    pupil_min_mm: float = 1.5
    pupil_max_mm: float = 9.0
    fixation_vel_thresh: float = 30.0      # deg/s
    fixation_min_ms: float = 100.0
    saccade_vel_thresh: float = 100.0      # deg/s
    saccade_min_ms: float = 20.0
    blink_min_ms: float = 80.0
    blink_max_ms: float = 500.0
    theta_band: Tuple[float, float] = (4.0, 7.0)
    alpha_band: Tuple[float, float] = (8.0, 12.0)
    beta_band: Tuple[float, float] = (13.0, 30.0)
    mwi_eps: float = 1e-6


def _normalize_timestamps(ts: pd.Series) -> pd.Series:
    s = pd.to_numeric(ts, errors="coerce").astype(float)
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        raise ValueError("时间戳列为空或无法解析。")
    median_val = float(np.median(s))
    scale = 1.0
    if median_val > 1e15:   # ns
        scale = 1e9
    elif median_val > 1e12: # us
        scale = 1e6
    elif median_val > 1e10: # ms
        scale = 1e3
    out = pd.to_numeric(ts, errors="coerce").astype(float) / scale
    return out - float(np.nanmin(out))


def _find_first_existing(cols: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    col_set = set(cols)
    for c in candidates:
        if c in col_set:
            return c
    return None


def _moving_average(values: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return values
    ser = pd.Series(values)
    return ser.rolling(window=win, center=True, min_periods=1).mean().to_numpy()


def _get_fs_from_time(ts: np.ndarray) -> float:
    dt = np.diff(ts)
    dt = dt[np.isfinite(dt) & (dt > 1e-6)]
    if len(dt) == 0:
        return 60.0
    return float(1.0 / np.median(dt))


def _segment_events(mask: np.ndarray, ts: np.ndarray, min_dur_s: float) -> List[Tuple[int, int]]:
    events: List[Tuple[int, int]] = []
    start = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        if (not flag) and start is not None:
            end = i - 1
            dur = float(ts[end] - ts[start]) if end > start else 0.0
            if dur >= min_dur_s:
                events.append((start, end))
            start = None
    if start is not None:
        end = len(mask) - 1
        dur = float(ts[end] - ts[start]) if end > start else 0.0
        if dur >= min_dur_s:
            events.append((start, end))
    return events


def _fft_bandpower(x: np.ndarray, fs: float, band: Tuple[float, float]) -> float:
    x = np.asarray(x, dtype=float)
    if len(x) < 8:
        return 0.0
    x = x - np.nanmean(x)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    n = len(x)
    fft = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / max(fs, 1e-6))
    psd = (np.abs(fft) ** 2) / max(n * fs, 1e-6)
    low, high = band
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return 0.0
    return float(np.trapz(psd[mask], freqs[mask]))


def _read_config_headers(config_path: Optional[str]) -> List[str]:
    if not config_path:
        return []
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"未找到 config 文件: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _preprocess_eye_dataframe(df_eye: pd.DataFrame, cfg: PipelineConfig) -> Dict[str, np.ndarray]:
    t_col = _find_first_existing(
        df_eye.columns,
        ["timestamp", "TimeStamp", "UserTimeStamp", "RealTimeClock"],
    )
    if t_col is None:
        raise ValueError("眼动数据缺少时间戳列（timestamp/TimeStamp/UserTimeStamp/RealTimeClock）。")
    ts = _normalize_timestamps(df_eye[t_col]).to_numpy()

    pupil_col = _find_first_existing(
        df_eye.columns,
        [
            "FilteredPupilDiameter", "PupilDiameter", "RightPupilDiameter",
            "LeftPupilDiameter", "FilteredRightPupilDiameter", "FilteredLeftPupilDiameter"
        ],
    )
    if pupil_col is None:
        raise ValueError("眼动数据缺少瞳孔直径列。")
    pupil = pd.to_numeric(df_eye[pupil_col], errors="coerce").to_numpy(dtype=float)

    quality_col = _find_first_existing(
        df_eye.columns,
        ["FilteredPupilDiameterQ", "PupilDiameterQ", "RightPupilDiameterQ", "LeftPupilDiameterQ"],
    )
    quality = (
        pd.to_numeric(df_eye[quality_col], errors="coerce").to_numpy(dtype=float)
        if quality_col is not None else np.ones_like(pupil)
    )

    heading_col = _find_first_existing(
        df_eye.columns,
        ["FilteredGazeHeading", "GazeHeading", "EstimatedGazeHeading", "FilteredLeftGazeHeading", "FilteredRightGazeHeading"],
    )
    pitch_col = _find_first_existing(
        df_eye.columns,
        ["FilteredGazePitch", "GazePitch", "EstimatedGazePitch", "FilteredLeftGazePitch", "FilteredRightGazePitch"],
    )
    if heading_col is None or pitch_col is None:
        raise ValueError("眼动数据缺少注视方向列（Heading/Pitch）。")
    heading = pd.to_numeric(df_eye[heading_col], errors="coerce").to_numpy(dtype=float)
    pitch = pd.to_numeric(df_eye[pitch_col], errors="coerce").to_numpy(dtype=float)

    eyelid_col = _find_first_existing(df_eye.columns, ["EyelidOpening", "RightEyelidOpening", "LeftEyelidOpening"])
    eyelid = (
        pd.to_numeric(df_eye[eyelid_col], errors="coerce").to_numpy(dtype=float)
        if eyelid_col is not None else None
    )

    if np.nanmedian(np.abs(pupil)) < 0.1:  # 可能是米，转毫米
        pupil = pupil * 1000.0

    valid = np.isfinite(pupil) & np.isfinite(ts) & (quality >= cfg.pupil_quality_min)
    valid &= (pupil >= cfg.pupil_min_mm) & (pupil <= cfg.pupil_max_mm)

    pupil_qc = pupil.copy()
    pupil_qc[~valid] = np.nan
    pupil_qc = pd.Series(pupil_qc, index=ts).interpolate(method="index", limit_direction="both").to_numpy(dtype=float)

    heading = pd.Series(heading, index=ts).interpolate(method="index", limit_direction="both").to_numpy(dtype=float)
    pitch = pd.Series(pitch, index=ts).interpolate(method="index", limit_direction="both").to_numpy(dtype=float)

    fs_eye = _get_fs_from_time(ts)
    smooth_win = max(1, int(round(cfg.smooth_sec * fs_eye)))
    pupil_sm = _moving_average(pupil_qc, smooth_win)
    heading_sm = _moving_average(heading, smooth_win)
    pitch_sm = _moving_average(pitch, smooth_win)

    dt = np.diff(ts, prepend=ts[0])
    dt[dt <= 1e-6] = np.nanmedian(dt[dt > 1e-6]) if np.any(dt > 1e-6) else 1.0 / fs_eye

    if eyelid is not None:
        eyelid_thr = float(np.nanpercentile(eyelid[np.isfinite(eyelid)], 20)) if np.any(np.isfinite(eyelid)) else 0.0
        blink_mask = (eyelid <= eyelid_thr) | (~valid)
    else:
        blink_mask = ~valid
    blink_events = _segment_events(blink_mask.astype(bool), ts, cfg.blink_min_ms / 1000.0)
    blink_events = [
        (s, e) for s, e in blink_events
        if (ts[e] - ts[s]) <= cfg.blink_max_ms / 1000.0
    ]

    ang_vel = np.sqrt(np.diff(heading_sm, prepend=heading_sm[0]) ** 2 + np.diff(pitch_sm, prepend=pitch_sm[0]) ** 2) / dt
    fix_events = _segment_events(ang_vel < cfg.fixation_vel_thresh, ts, cfg.fixation_min_ms / 1000.0)
    sac_events = _segment_events(ang_vel > cfg.saccade_vel_thresh, ts, cfg.saccade_min_ms / 1000.0)

    return {
        "ts": ts,
        "pupil": pupil_sm,
        "pupil_valid": valid.astype(int),
        "heading": heading_sm,
        "pitch": pitch_sm,
        "blink_events": np.array(blink_events, dtype=int) if blink_events else np.zeros((0, 2), dtype=int),
        "fix_events": np.array(fix_events, dtype=int) if fix_events else np.zeros((0, 2), dtype=int),
        "sac_events": np.array(sac_events, dtype=int) if sac_events else np.zeros((0, 2), dtype=int),
        "fs_eye": fs_eye,
    }


def _events_in_window(events: np.ndarray, ts: np.ndarray, t0: float, t1: float) -> np.ndarray:
    if len(events) == 0:
        return events
    mask = []
    for s, e in events:
        es = ts[s]
        ee = ts[e]
        mask.append(not (ee < t0 or es > t1))
    return events[np.array(mask, dtype=bool)]


def _build_eye_feature_windows(eye: Dict[str, np.ndarray], cfg: PipelineConfig) -> pd.DataFrame:
    ts = eye["ts"]
    pupil = eye["pupil"]
    fs = eye["fs_eye"]
    step = cfg.window_sec * (1.0 - cfg.overlap)
    if step <= 0:
        raise ValueError("overlap 必须小于1。")

    starts = np.arange(ts[0], ts[-1] - cfg.window_sec + 1e-9, step)
    global_base = float(np.nanmedian(pupil[(ts >= ts[0]) & (ts <= ts[0] + 30.0)]))
    if not np.isfinite(global_base):
        global_base = float(np.nanmedian(pupil))

    rows: List[Dict[str, float]] = []
    for t0 in starts:
        t1 = t0 + cfg.window_sec
        mask = (ts >= t0) & (ts < t1)
        n = int(np.sum(mask))
        expected = max(1, int(round(cfg.window_sec * fs)))
        if n < int(expected * cfg.min_window_coverage):
            continue

        win_t = ts[mask]
        win_p = pupil[mask]
        dt = np.diff(win_t)
        vel = np.abs(np.diff(win_p)) / np.where(dt <= 1e-6, np.nan, dt)
        vel = vel[np.isfinite(vel)]

        hist1 = pupil[(ts >= t1 - 1.0) & (ts < t1)]
        hist5 = pupil[(ts >= t1 - 5.0) & (ts < t1)]
        if len(hist1) == 0:
            hist1 = win_p
        if len(hist5) == 0:
            hist5 = win_p

        if len(win_p) > 2:
            x = np.arange(len(win_p), dtype=float)
            slope = float(np.polyfit(x, win_p, 1)[0])
            trend = float(np.clip(slope, -1.0, 1.0))
        else:
            trend = 0.0

        fix_in = _events_in_window(eye["fix_events"], ts, t0, t1)
        sac_in = _events_in_window(eye["sac_events"], ts, t0, t1)
        blink_in = _events_in_window(eye["blink_events"], ts, t0, t1)

        fix_durs = [(ts[e] - ts[s]) for s, e in fix_in] if len(fix_in) > 0 else []
        sac_amps = [
            float(np.sqrt((eye["heading"][e] - eye["heading"][s]) ** 2 + (eye["pitch"][e] - eye["pitch"][s]) ** 2))
            for s, e in sac_in
        ] if len(sac_in) > 0 else []

        row = {
            "window_start": float(t0),
            "window_end": float(t1),
            "window_mid": float((t0 + t1) / 2.0),
            "pupil_mean": float(np.nanmean(win_p)),
            "pupil_baseline": float(global_base),
            "pupil_rel_change": float((np.nanmean(win_p) - global_base) / (global_base + 1e-6) * 100.0),
            "pupil_velocity": float(np.nanmean(vel) if len(vel) > 0 else 0.0),
            "pupil_std": float(np.nanstd(win_p)),
            "pupil_mean_1s": float(np.nanmean(hist1)),
            "pupil_mean_5s": float(np.nanmean(hist5)),
            "pupil_trend": trend,
            "fixation_duration_mean": float(np.mean(fix_durs) if len(fix_durs) > 0 else 0.0),
            "fixation_rate": float(len(fix_durs) / cfg.window_sec),
            "saccade_amplitude_mean": float(np.mean(sac_amps) if len(sac_amps) > 0 else 0.0),
            "blink_rate": float(len(blink_in) / cfg.window_sec),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _pick_region_channels(cols: Sequence[str], region: str) -> List[str]:
    region = region.lower()
    if region == "frontal":
        pat = re.compile(r"^(Fp|FP|AF|F)\d*Z?$", re.IGNORECASE)
    elif region == "parietal":
        pat = re.compile(r"^P\d*Z?$", re.IGNORECASE)
    elif region == "temporal":
        pat = re.compile(r"^(T|TP|FT)\d*Z?$", re.IGNORECASE)
    else:
        return []
    return [c for c in cols if pat.match(c)]


def _compute_mwi_for_windows(
    df_eeg: pd.DataFrame,
    windows: pd.DataFrame,
    cfg: PipelineConfig,
    theta_region: str,
    alpha_region: str,
    beta_region: str,
) -> pd.DataFrame:
    t_col = _find_first_existing(df_eeg.columns, ["timestamp", "machine_timestamp", "TimeStamp"])
    if t_col is None:
        raise ValueError("EEG数据缺少时间戳列（timestamp/machine_timestamp/TimeStamp）。")
    ts = _normalize_timestamps(df_eeg[t_col]).to_numpy()

    exclude = {t_col, "machine_timestamp", "timestamp", "TimeStamp"}
    eeg_cols = [c for c in df_eeg.columns if c not in exclude]
    eeg_data = df_eeg[eeg_cols].apply(pd.to_numeric, errors="coerce")
    eeg_data = eeg_data.ffill().bfill().fillna(0.0)
    fs_eeg = _get_fs_from_time(ts)

    frontal_cols = _pick_region_channels(eeg_cols, theta_region)
    alpha_cols = _pick_region_channels(eeg_cols, alpha_region)
    beta_cols = _pick_region_channels(eeg_cols, beta_region)
    if len(frontal_cols) == 0:
        frontal_cols = eeg_cols
    if len(alpha_cols) == 0:
        alpha_cols = eeg_cols
    if len(beta_cols) == 0:
        beta_cols = eeg_cols

    rows = []
    for _, w in windows.iterrows():
        t0, t1 = float(w["window_start"]), float(w["window_end"])
        mask = (ts >= t0) & (ts < t1)
        if np.sum(mask) < max(8, int(0.5 * cfg.window_sec * fs_eeg)):
            rows.append({"mwi": np.nan, "theta_power": np.nan, "alpha_power": np.nan, "beta_power": np.nan})
            continue

        def _region_power(cols: List[str], band: Tuple[float, float]) -> float:
            powers = []
            for c in cols:
                sig = eeg_data.loc[mask, c].to_numpy(dtype=float)
                powers.append(_fft_bandpower(sig, fs_eeg, band))
            return float(np.mean(powers)) if len(powers) > 0 else 0.0

        theta_p = _region_power(frontal_cols, cfg.theta_band)
        alpha_p = _region_power(alpha_cols, cfg.alpha_band)
        beta_p = _region_power(beta_cols, cfg.beta_band)
        mwi = (theta_p + beta_p) / (alpha_p + cfg.mwi_eps)
        rows.append({
            "mwi": float(mwi),
            "theta_power": float(theta_p),
            "alpha_power": float(alpha_p),
            "beta_power": float(beta_p),
        })

    out = pd.DataFrame(rows)
    median_mwi = float(np.nanmedian(out["mwi"].to_numpy(dtype=float)))
    out["mwi_label"] = (out["mwi"] >= median_mwi).astype(int)
    out["mwi_median"] = median_mwi
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="构建论文5.2.3原始眼动->12维特征->MWI标签数据集")
    parser.add_argument("--eye-csv", required=True, help="raw_data.py 输出的眼动CSV")
    parser.add_argument("--eeg-csv", default="", help="raw_data.py 输出的EEG CSV（可选，部署阶段可不提供）")
    parser.add_argument("--config", default="config.txt", help="眼动字段配置（逐行列名）")
    parser.add_argument("--output-csv", default="data/eye_cognitive_load_from_raw.csv", help="输出样本级CSV")
    parser.add_argument("--window-sec", type=float, default=2.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--theta-region", default="frontal", choices=["frontal", "parietal", "temporal", "all"])
    parser.add_argument("--alpha-region", default="parietal", choices=["frontal", "parietal", "temporal", "all"])
    parser.add_argument("--beta-region", default="parietal", choices=["frontal", "parietal", "temporal", "all"])
    args = parser.parse_args()

    cfg = PipelineConfig(window_sec=args.window_sec, overlap=args.overlap)
    cfg_headers = _read_config_headers(args.config)

    if not os.path.exists(args.eye_csv):
        raise FileNotFoundError(f"未找到眼动CSV: {args.eye_csv}")
    df_eye = pd.read_csv(args.eye_csv)
    if cfg_headers:
        missing_in_eye = [c for c in cfg_headers if c not in df_eye.columns]
        if missing_in_eye:
            print(f"[提示] config中 {len(missing_in_eye)} 个字段在当前眼动CSV中不存在（可忽略）")

    eye_proc = _preprocess_eye_dataframe(df_eye, cfg)
    feature_df = _build_eye_feature_windows(eye_proc, cfg)
    if feature_df.empty:
        raise RuntimeError("未生成有效滑窗特征，请检查原始眼动数据质量或时间戳。")

    if args.eeg_csv:
        if not os.path.exists(args.eeg_csv):
            raise FileNotFoundError(f"未找到EEG CSV: {args.eeg_csv}")
        df_eeg = pd.read_csv(args.eeg_csv)
        mwi_df = _compute_mwi_for_windows(
            df_eeg=df_eeg,
            windows=feature_df[["window_start", "window_end", "window_mid"]],
            cfg=cfg,
            theta_region=args.theta_region,
            alpha_region=args.alpha_region,
            beta_region=args.beta_region,
        )
        dataset = pd.concat([feature_df, mwi_df], axis=1)
    else:
        dataset = feature_df.copy()
        dataset["mwi"] = np.nan
        dataset["mwi_label"] = np.nan
        dataset["mwi_median"] = np.nan

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(out_path, index=False, encoding="utf-8")

    meta = {
        "eye_csv": args.eye_csv,
        "eeg_csv": args.eeg_csv if args.eeg_csv else None,
        "config": args.config,
        "n_windows": int(len(dataset)),
        "feature_columns": DEFAULT_FEATURE_COLS,
        "has_mwi": bool(args.eeg_csv),
        "window_sec": args.window_sec,
        "overlap": args.overlap,
        "theta_region": args.theta_region,
        "alpha_region": args.alpha_region,
        "beta_region": args.beta_region,
    }
    with open(out_path.with_suffix(".meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[完成] 样本级数据已保存: {out_path}")
    print(f"[完成] 共 {len(dataset)} 个窗口样本，特征维度={len(DEFAULT_FEATURE_COLS)}")
    if args.eeg_csv:
        valid_mwi = int(np.sum(np.isfinite(dataset["mwi"].to_numpy(dtype=float))))
        print(f"[完成] 有效MWI标签窗口: {valid_mwi}/{len(dataset)}")


if __name__ == "__main__":
    main()

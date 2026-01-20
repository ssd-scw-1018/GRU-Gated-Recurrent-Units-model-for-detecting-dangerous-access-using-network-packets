# build_scenarios_final.py
from pathlib import Path
import numpy as np
import pandas as pd
import json

# -----------------------------
# 경로 설정
# -----------------------------
# 이 파일(build_scenarios_final.py)이 있는 폴더 = 프로젝트 루트
ROOT_DIR = Path(__file__).resolve().parent

# 전처리 파라미터 위치
PREPROC_PARAMS_PATH = ROOT_DIR / "preproc_params.json"

# 원본 CSV 폴더와 파일 이름
DATA_DIR   = "data_4_split"
TEST_CSVS  = ("UNSW-NB15_4.csv",)

# 포트는 메타에만 보관(학습 입력 X)
INCLUDE_PORT_FEATURES = False

# 사용할 열 정의 (preprocessing.py와 동일)
USE = [
    "srcip","sport","dstip","dsport",
    "proto","state","dur","sbytes","dbytes","sttl","dttl","sloss","dloss","service",
    "Sload","Dload","Spkts","Dpkts","swin","dwin","stcpb","dtcpb","trans_depth",
    "res_bdy_len","Stime","Ltime","Sintpkt","Dintpkt","tcprtt","synack","ackdat",
    "is_sm_ips_ports","ct_state_ttl","ct_flw_http_mthd","is_ftp_login","ct_ftp_cmd",
    "ct_srv_src","ct_srv_dst","ct_dst_ltm","ct_src_ltm","ct_src_dport_ltm",
    "ct_dst_sport_ltm","ct_dst_src_ltm","attack_cat","Label"
]

CAT_COLS   = ["proto","state","service"]
PORT_COLS  = ["sport","dsport"]
TTL_COLS   = ["sttl","dttl"]
SEQ_COLS   = ["stcpb","dtcpb"]
BOOL_COLS  = ["is_sm_ips_ports","is_ftp_login"]
NUM_LOGZ = [
    "dur","sbytes","dbytes","Sload","Dload","Spkts","Dpkts","swin","dwin",
    "trans_depth","res_bdy_len","Sintpkt","Dintpkt","tcprtt","synack","ackdat",
    "sloss","dloss",
    "ct_state_ttl","ct_flw_http_mthd","ct_ftp_cmd",
    "ct_srv_src","ct_srv_dst","ct_dst_ltm","ct_src_ltm",
    "ct_src_dport_ltm","ct_dst_sport_ltm","ct_dst_src_ltm",
]
META_COLS = ["event_id","srcip","sport","dstip","dsport", "f_name"]

# train/val과 겹치지 않게 하는 test event_id 오프셋
TEST_BASE     = 10_000_000_000  # 1e10쯤이면 충분

# =========================
# artifacts dir / RNG
# =========================
ART = ROOT_DIR / "artifacts_parquet"
RNG = np.random.default_rng(20241201)  # 시나리오 재현용 시드

# =========================
# 결측치 시나리오용 설정
# =========================
INJECT_MISSING_VALUES = True
MISSING_RATE_NUM = 0.15   # 수치형 값 중 15%를 NaN
MISSING_RATE_TTL = 0.15   # TTL 값 중 15%를 NaN
MISSING_RATE_CAT = 0.10   # 범주형 값 중 10%를 NaN

# 결측치 시나리오에서 사용할 최대 행 수
MAX_MISSING_ROWS = 100_000


def _cast_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _log1p_clip_standardize(col_s, clip_val, mu, sd):
    v = pd.to_numeric(col_s, errors="coerce").astype("float64").values
    v = np.minimum(v, clip_val if np.isfinite(clip_val) else v)
    v = np.log1p(np.clip(v, 0, None))
    sd = (sd if sd not in (0, None, 0.0) else 1.0)
    mu = (mu if mu is not None else 0.0)
    return (v - mu) / sd

def _encode_cats_to_int(s, vocab):
    idx = {tok: i+1 for i, tok in enumerate(vocab)}  # 0=UNK
    return s.map(idx).fillna(0).astype("int32")

def _load_concat_csvs(root: Path, files):
    na_tokens = ["-", "--", "None", "none", "NULL", "null", ""]
    frames = []
    for f in files:
        df_i = pd.read_csv(root / f, low_memory=False, na_values=na_tokens)
        if "f_name" not in df_i.columns:
            df_i["f_name"] = Path(f).name   # UNSW-NB15_4.csv 같은 태그
        use_cols_present = [c for c in USE if c in df_i.columns]
        if "f_name" not in use_cols_present:
            use_cols_present.append("f_name")
        frames.append(df_i[use_cols_present])
    return pd.concat(frames, ignore_index=True)

def _make_meta(df: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in META_COLS if c in df.columns]
    meta = df[keep].copy()
    for c in ["srcip","dstip","sport","dsport", "f_name"]:
        if c in meta.columns:
            meta[c] = meta[c].astype(str)
    return meta.set_index("event_id")

# =========================
# 값 수준 NaN 주입 함수
# =========================
def _inject_value_missing(
    df: pd.DataFrame,
    num_rate: float = MISSING_RATE_NUM,
    ttl_rate: float = MISSING_RATE_TTL,
    cat_rate: float = MISSING_RATE_CAT,
    random_state: int = 20241205,
) -> pd.DataFrame:
    """
    일부 열에 결측값(NaN)을 랜덤하게 섞어 넣는 함수.
    - NUM_LOGZ, TTL_COLS, CAT_COLS에 대해 각각 비율만큼 NaN 주입.
    - transform_all()의 결측 대응 로직이 잘 작동하는지 테스트용.
    """
    rng = np.random.default_rng(random_state)
    df = df.copy()
    n_rows = len(df)
    if n_rows == 0:
        return df

    # 1) 로그 수치형 피처
    for c in NUM_LOGZ:
        if c in df.columns:
            mask = rng.random(n_rows) < num_rate
            df.loc[mask, c] = np.nan

    # 2) TTL 피처
    for c in TTL_COLS:
        if c in df.columns:
            mask = rng.random(n_rows) < ttl_rate
            df.loc[mask, c] = np.nan

    # 3) 범주형 피처
    for c in CAT_COLS:
        if c in df.columns:
            mask = rng.random(n_rows) < cat_rate
            df.loc[mask, c] = np.nan

    print(
        f"[inject_missing] num_rate={num_rate}, ttl_rate={ttl_rate}, "
        f"cat_rate={cat_rate} 로 NaN 주입 완료"
    )
    return df


def transform_all(df: pd.DataFrame, params: dict,
                  make_time_features=True, drop_time_raw=True,
                  keep_attack_cat=False):
    """
    전처리 함수 (결측 칼럼 대응 로직 추가됨)
    - 범주형 누락 시: 최빈값(Mode)으로 대체
    - 수치형 누락 시: 훈련 데이터의 평균(Mean) 또는 기하평균으로 대체
    """
    X = df.copy()

    # 타깃 분리
    y = None
    if "Label" in X.columns:
        y = pd.to_numeric(X["Label"], errors="coerce").fillna(0).astype(int)
        X = X.drop(columns=["Label"])
    if ("attack_cat" in X.columns) and (not keep_attack_cat):
        X = X.drop(columns=["attack_cat"])

    # -------------------------------------------------------------------------
    # [Robustness] 결측 칼럼 자동 보완 로직 (Missing Column Imputation)
    # -------------------------------------------------------------------------
    
    # 1. 범주형 결측 처리 (최빈값 사용)
    for c in CAT_COLS:
        if c not in X.columns:
            # vocabs의 첫 번째 요소가 최빈값(Mode)
            most_freq = params.get("vocabs", {}).get(c, ["unknown"])[0]
            X[c] = most_freq

    # 2. 로그 수치형 결측 처리 (기하 평균 사용)
    for c in NUM_LOGZ:
        if c not in X.columns:
            mu = params["mu_log"].get(c, 0.0)
            fill_val = np.expm1(mu) # 역연산으로 원본 스케일 복원
            X[c] = fill_val

    # 3. TTL 등 일반 수치형 결측 처리 (산술 평균 사용)
    for c in TTL_COLS:
        if c not in X.columns:
            mu = params["mu_ttl"].get(c, 0.0)
            X[c] = mu

    # 4. 시퀀스/불리언 등 기타 필수 칼럼 처리 (0으로 대체)
    for c in SEQ_COLS + BOOL_COLS + ["sbytes", "dbytes", "Spkts", "Dpkts", "swin", "dwin"]:
        if c not in X.columns:
            X[c] = 0
    # -------------------------------------------------------------------------

    # 이후 기존 전처리 로직 수행
    _cast_numeric(X, NUM_LOGZ + TTL_COLS + SEQ_COLS)
    for c in BOOL_COLS:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0).astype(int)
        else:
            X[c] = 0

    # 범주형 → 정수 인코딩(UNK=0)
    for c in CAT_COLS:
        if c in X.columns:
            vocab = params.get("vocabs", {}).get(c, [])
            X[c] = _encode_cats_to_int(X[c].astype(str), vocab)

    # 수치형: clip→log1p→z
    for c in NUM_LOGZ:
        if c in X.columns:
            X[c] = _log1p_clip_standardize(
                X[c],
                params["p99"].get(c, np.inf),
                params["mu_log"].get(c, 0.0),
                params["sd_log"].get(c, 1.0),
            )

    # TTL: z만
    for c in TTL_COLS:
        if c in X.columns:
            mu = params["mu_ttl"].get(c, 0.0)
            sd = params["sd_ttl"].get(c, 1.0) or 1.0
            X[c] = (X[c].astype("float64") - mu) / sd

    # 파생 변수 생성
    if set(SEQ_COLS).issubset(X.columns):
        X["seq_diff"] = (X["stcpb"].astype("float64") - X["dtcpb"].astype("float64"))
    if {"sbytes","dbytes"}.issubset(X.columns):
        X["bytes_tot"]   = X["sbytes"] + X["dbytes"]
        X["bytes_ratio"] = (X["sbytes"] / (X["bytes_tot"] + 1e-6)).clip(0, 1)
    if {"Spkts","Dpkts"}.issubset(X.columns):
        X["pkts_tot"]   = X["Spkts"] + X["Dpkts"]
        X["pkts_ratio"] = (X["Spkts"] / (X["pkts_tot"] + 1e-6)).clip(0, 1)
    if {"swin","dwin"}.issubset(X.columns):
        X["win_ratio"] = (X["swin"] / (X["swin"] + X["dwin"] + 1e-6)).clip(0, 1)
    if {"sttl","dttl"}.issubset(X.columns):
        X["ttl_diff"] = X["sttl"] - X["dttl"]

    # 시간 파생
    if make_time_features and ("Stime" in X.columns):
        ts = pd.to_datetime(pd.to_numeric(X["Stime"], errors="coerce"), unit="s", utc=True)
        X["hour"] = ts.dt.hour.fillna(0).astype("int16")
        X["dow"]  = ts.dt.dayofweek.fillna(0).astype("int8")

    # 학습 입력에서 제외할 칼럼 드롭
    drop_cols = []
    if drop_time_raw:
        drop_cols += [c for c in ["Stime","Ltime"] if c in X.columns]
    drop_cols += [c for c in ["srcip","sport","dstip","dsport", "f_name"] if c in X.columns]
    drop_cols += [c for c in SEQ_COLS if c in X.columns]
    if not INCLUDE_PORT_FEATURES:
        pass
    X = X.drop(columns=drop_cols, errors="ignore")
    X = X.drop(columns=[c for c in ["bytes_tot","pkts_tot"] if c in X.columns], errors="ignore")
    X = X.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    
    return X, y

def prepare_test_parquets():
    """
    preprocessing.py의 transform_holdout과 같은 역할:
    - UNSW-NB15_4.csv -> test_X/Y/meta.parquet 생성
    """
    root_dir = ROOT_DIR
    data_dir = root_dir / DATA_DIR
    out_dir  = ART
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_concat_csvs(data_dir, TEST_CSVS)
    df["event_id"] = np.arange(len(df), dtype=np.int64) + TEST_BASE
    df = df.sort_values(
        [c for c in ["srcip","Stime","Ltime","event_id"] if c in df.columns],
        kind="mergesort"
    ).reset_index(drop=True)

    meta = _make_meta(df)

    # preproc_params.json은 모델(.pth)과 동일한 폴더에서 읽음
    with open(PREPROC_PARAMS_PATH, "r", encoding="utf-8") as f:
        params = json.load(f)

    Xte, yte = transform_all(df, params, True, True)

    Xte.to_parquet(out_dir / "test_X.parquet",index=False)
    if "Label" in df.columns:
        pd.DataFrame({"event_id": df["event_id"].values, "Label": yte.values})\
          .to_parquet(out_dir / "test_y.parquet", index=False)
    meta.to_parquet(out_dir / "test_meta.parquet")

    print("Test saved:", out_dir.resolve(), "test_X:", Xte.shape)


def _load_test_joined():
    """
    test_X, test_y, test_meta를 한 번 합쳐서 큰 df로 만든 뒤 반환.
    df_all: event_id, [features...], Label, srcip, sport, dstip, dsport, f_name ...
    """
    X    = pd.read_parquet(ART / "test_X.parquet")        # event_id + features
    y    = pd.read_parquet(ART / "test_y.parquet")        # event_id, Label
    meta = pd.read_parquet(ART / "test_meta.parquet")     # index=event_id or col

    if meta.index.name != "event_id":
        meta = meta.set_index("event_id")

    meta_reset = meta.reset_index()

    df = (
        X.merge(y, on="event_id", how="left")
         .merge(meta_reset, on="event_id", how="left")
    )
    X_cols    = X.columns.tolist()          # event_id + feature들
    meta_cols = meta_reset.columns.tolist() # event_id, srcip, dstip, ...
    return df, X_cols, meta_cols


# -------------------------------------------------------------
# 옵션 1 구현: "애매한 공격(soft attack)"을 시나리오에 주입
# -------------------------------------------------------------
def _inject_soft_attacks(
    scen_df: pd.DataFrame,
    df_all: pd.DataFrame,
    X_cols,
    soft_ratio: float = 0.1,
) -> pd.DataFrame:
    pos = scen_df[scen_df["Label"] == 1]
    neg_global = df_all[df_all["Label"] == 0]

    if pos.empty or neg_global.empty:
        print("[soft] 공격 또는 정상 샘플이 부족해 soft attack을 만들지 않습니다.")
        return scen_df

    n_soft = int(len(pos) * soft_ratio)
    n_soft = min(n_soft, len(pos), len(neg_global))
    if n_soft <= 0:
        print("[soft] soft attack 개수가 0입니다.")
        return scen_df

    feature_cols = [c for c in X_cols if c != "event_id"]

    atk_idx = RNG.choice(len(pos), size=n_soft, replace=False)
    norm_idx = RNG.choice(len(neg_global), size=n_soft, replace=False)

    pos_sample = pos.iloc[atk_idx].reset_index(drop=True)
    neg_sample = neg_global.iloc[norm_idx].reset_index(drop=True)

    new_rows = []
    base_eid = int(scen_df["event_id"].max()) + 1

    for i in range(n_soft):
        atk_row = pos_sample.iloc[i].copy()
        nor_row = neg_sample.iloc[i]

        alpha = RNG.uniform(0.3, 0.8)

        mixed_feats = (
            alpha * atk_row[feature_cols].to_numpy(dtype=float)
            + (1.0 - alpha) * nor_row[feature_cols].to_numpy(dtype=float)
        )

        atk_row[feature_cols] = mixed_feats
        atk_row["event_id"] = base_eid + i
        atk_row["Label"] = 1

        new_rows.append(atk_row)

    soft_df = pd.DataFrame(new_rows)
    out_df = pd.concat([scen_df, soft_df], axis=0).reset_index(drop=True)

    print(
        f"[soft] soft attacks added: {len(soft_df)} "
        f"(orig pos={len(pos)}, new total rows={len(out_df)})"
    )

    return out_df


# -------------------------------------------------------------------
# 시나리오 1: 외부 DDoS + 다양한 공격자 IP
# -------------------------------------------------------------------
def _build_ddos_scenario(df_all: pd.DataFrame, X_cols) -> pd.DataFrame:
    pos = df_all[df_all["Label"] == 1].copy()
    neg = df_all[df_all["Label"] == 0].copy()

    if pos.empty:
        raise ValueError("Label=1 (attack) 샘플이 없습니다. ddos 시나리오 생성 불가.")

    atk_counts = (
        pos.groupby("srcip")["event_id"]
        .count()
        .sort_values(ascending=False)
    )

    main_atk_ip = atk_counts.index[0]
    sub_atk_ips = [ip for ip in atk_counts.index if ip != main_atk_ip][:3]

    src_label_stats = df_all.groupby("srcip")["Label"].agg(["sum", "count"])
    benign_ips = src_label_stats[src_label_stats["sum"] == 0] \
                    .sort_values("count", ascending=False) \
                    .head(30).index.tolist()
    if len(benign_ips) < 5:
        tmp = src_label_stats.copy()
        tmp["neg_ratio"] = (tmp["count"] - tmp["sum"]) / (tmp["count"] + 1e-6)
        benign_ips = tmp.sort_values(["neg_ratio", "count"], ascending=False) \
                        .head(30).index.tolist()

    blocks = []

    # A. 평소 트래픽
    target_norm_rows = 8_000
    per_ip_max = 600

    norm_frames = []
    for ip in benign_ips:
        df_ip = neg[neg["srcip"] == ip]
        if df_ip.empty: continue
        n = min(per_ip_max, len(df_ip))
        idx = RNG.choice(len(df_ip), size=n, replace=False)
        norm_frames.append(df_ip.iloc[idx])

    if norm_frames:
        df_norm = pd.concat(norm_frames, axis=0)
        if len(df_norm) > target_norm_rows:
            idx = RNG.choice(len(df_norm), size=target_norm_rows, replace=False)
            df_norm = df_norm.iloc[idx]
        df_norm = df_norm.sample(frac=1.0, random_state=42).reset_index(drop=True)
        blocks.append(df_norm)

    # B. 서브 공격자
    for ip in sub_atk_ips:
        df_ip_pos = pos[pos["srcip"] == ip]
        if len(df_ip_pos) < 20: continue
        n = min(len(df_ip_pos), int(RNG.integers(30, 81)))
        idx = RNG.choice(len(df_ip_pos), size=n, replace=False)
        blocks.append(df_ip_pos.iloc[idx].copy())

        if not neg.empty:
            n_norm = int(n * 0.5)
            idx2 = RNG.choice(len(neg), size=n_norm, replace=False)
            blocks.append(neg.iloc[idx2])

    # C. 메인 공격자
    df_main_pos = pos[pos["srcip"] == main_atk_ip]
    n_heavy = min(3_000, len(df_main_pos))
    if n_heavy < 200: n_heavy = len(df_main_pos)
    if n_heavy > 0:
        idx = RNG.choice(len(df_main_pos), size=n_heavy, replace=False)
        blocks.append(df_main_pos.iloc[idx].copy())

    scen_df = pd.concat(blocks, axis=0).reset_index(drop=True)

    print("[ddos scenario]")
    print(f"  main_atk_ip : {main_atk_ip}, total rows : {len(scen_df)}")

    scen_df = _inject_soft_attacks(scen_df, df_all, X_cols, soft_ratio=0.3)
    return scen_df


# -------------------------------------------------------------------
# 시나리오 2: 느리지만 꾸준한 포트 스캔 + 여러 스캐너
# -------------------------------------------------------------------
def _build_slow_scan_scenario(df_all: pd.DataFrame, X_cols) -> pd.DataFrame:
    pos = df_all[df_all["Label"] == 1].copy()
    neg = df_all[df_all["Label"] == 0].copy()

    if pos.empty:
        raise ValueError("Label=1 (attack) 샘플이 없습니다. slow_scan 시나리오 생성 불가.")

    atk_stats = pos.groupby("srcip").agg(
        pos_cnt=("event_id", "count"),
        dst_uniq=("dstip", "nunique")
    )
    atk_stats = atk_stats.sort_values(["dst_uniq", "pos_cnt"], ascending=False)

    scanner_ips = atk_stats.index.tolist()
    main_scan_ip = scanner_ips[0]
    sub_scan_ips = scanner_ips[1:3]

    src_label_stats = df_all.groupby("srcip")["Label"].agg(["sum", "count"])
    benign_ips = src_label_stats[src_label_stats["sum"] == 0] \
                    .sort_values("count", ascending=False) \
                    .head(30).index.tolist()
    if len(benign_ips) < 5:
        tmp = src_label_stats.copy()
        tmp["neg_ratio"] = (tmp["count"] - tmp["sum"]) / (tmp["count"] + 1e-6)
        benign_ips = tmp.sort_values(["neg_ratio", "count"], ascending=False) \
                        .head(30).index.tolist()

    blocks = []

    # A. 오전 정상
    target_norm_rows = 5_000
    per_ip_max = 400

    norm_frames = []
    for ip in benign_ips:
        df_ip = neg[neg["srcip"] == ip]
        if df_ip.empty: continue
        n = min(per_ip_max, len(df_ip))
        idx = RNG.choice(len(df_ip), size=n, replace=False)
        norm_frames.append(df_ip.iloc[idx])

    if norm_frames:
        df_norm = pd.concat(norm_frames, axis=0)
        if len(df_norm) > target_norm_rows:
            idx = RNG.choice(len(df_norm), size=target_norm_rows, replace=False)
            df_norm = df_norm.iloc[idx]
        df_norm = df_norm.sample(frac=1.0, random_state=99).reset_index(drop=True)
        blocks.append(df_norm)

    # B. 서브 스캐너
    for ip in sub_scan_ips:
        df_ip_pos = pos[pos["srcip"] == ip]
        if len(df_ip_pos) < 20: continue
        n = min(len(df_ip_pos), int(RNG.integers(50, 151)))
        idx = RNG.choice(len(df_ip_pos), size=n, replace=False)
        blocks.append(df_ip_pos.iloc[idx].copy())

        if not neg.empty:
            n_norm = int(n * 0.5)
            idx2 = RNG.choice(len(neg), size=n_norm, replace=False)
            blocks.append(neg.iloc[idx2])

    # C. 메인 스캐너
    df_main_pos = pos[pos["srcip"] == main_scan_ip]
    n_scan = min(2_000, len(df_main_pos))
    if n_scan < 200: n_scan = len(df_main_pos)
    if n_scan > 0:
        idx = RNG.choice(len(df_main_pos), size=n_scan, replace=False)
        blocks.append(df_main_pos.iloc[idx].copy())

    scen_df = pd.concat(blocks, axis=0).reset_index(drop=True)

    print("[slow_scan scenario]")
    print(f"  main_scan_ip : {main_scan_ip}, total rows : {len(scen_df)}")

    scen_df = _inject_soft_attacks(scen_df, df_all, X_cols, soft_ratio=0.3)
    return scen_df


# -------------------------------------------------------------------
# 시나리오 3: 피처/열 결측치가 존재하는 시나리오
# -------------------------------------------------------------------
def build_missing_feature_scenario():
    """
    반환은 하지 않고, 바로 artifacts_parquet에 저장:
      - scenario_missing_X.parquet
      - scenario_missing_y.parquet
      - scenario_missing_meta.parquet
      - scenario_missing_raw_missing.json   👈 (raw 결측률 요약)
    (행 수는 MAX_MISSING_ROWS = 100,000 으로 제한)
    """
    root_dir = ROOT_DIR
    data_dir = root_dir / DATA_DIR
    out_dir  = ART
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 원본 테스트 CSV 로드 + event_id 부여
    df = _load_concat_csvs(data_dir, TEST_CSVS)
    df["event_id"] = np.arange(len(df), dtype=np.int64) + TEST_BASE

    # 🔹 1-1) 결측치 시나리오는 최대 10만 행만 사용
    if len(df) > MAX_MISSING_ROWS:
        df = df.sample(n=MAX_MISSING_ROWS, random_state=20241210)
        df = df.sort_values("event_id").reset_index(drop=True)
    print(f"[scenario_missing] rows limited to {len(df)} (MAX_MISSING_ROWS={MAX_MISSING_ROWS})")

    # f_name: 결측치 시나리오 전용 이름으로 재생성
    indices = np.arange(len(df))
    df["f_name"] = "UNSW-NB15_4_missing_" + pd.Series(indices).astype(str) + ".csv"
    print(f"[scenario_missing] f_name unique values generated: {len(df)} files.")

    # 2) 메타( srcip, sport, dstip, dsport, f_name ) 확보
    meta = _make_meta(df)

    # 3) 일부 열을 "통째로" 드롭해서
    #    transform_all에서 "열 자체가 없는" 상황을 테스트
    DROP_NUM_COLS = ["dur", "Sload"]
    DROP_CAT_COLS = ["proto"]
    drop_cols = [c for c in DROP_NUM_COLS + DROP_CAT_COLS if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        print(f"[missing_scenario] dropped columns: {drop_cols}")

    # 4) 나머지 열에는 값 수준의 NaN도 섞어 주입
    if INJECT_MISSING_VALUES:
        df = _inject_value_missing(df)

    #  4-1) 원본(raw) 기준 결측률 요약 → JSON으로 저장
    raw_missing_rates = df.isna().mean().to_dict()
    raw_missing_json = {
        "scenario": "missing_feature",
        "n_rows": int(len(df)),
        "missing_rate": {col: float(rate) for col, rate in raw_missing_rates.items()},
    }
    raw_json_path = out_dir / "scenario_missing_raw_missing.json"
    with open(raw_json_path, "w", encoding="utf-8") as f:
        json.dump(raw_missing_json, f, indent=2, ensure_ascii=False)
    print(f"[scenario_missing] raw missing summary saved -> {raw_json_path}")

    # 5) 학습 때와 동일한 params 로 transform_all 수행
    with open(PREPROC_PARAMS_PATH, "r", encoding="utf-8") as f:
        params = json.load(f)

    Xmiss, ymiss = transform_all(df, params, True, True)

    # 6) 저장 (이름만 scenario_missing_* 으로)
    X_path = out_dir / "scenario_missing_X.parquet"
    y_path = out_dir / "scenario_missing_y.parquet"
    m_path = out_dir / "scenario_missing_meta.parquet"

    Xmiss.to_parquet(X_path, index=False)
    if "Label" in df.columns:
        pd.DataFrame({"event_id": df["event_id"].values, "Label": ymiss.values})\
          .to_parquet(y_path, index=False)
    meta.to_parquet(m_path)

    print(f"[+] saved missing_feature scenario:")
    print(f"    X   -> {X_path} shape={Xmiss.shape}")
    print(f"    y   -> {y_path} shape={ymiss.shape}")
    print(f"    meta-> {m_path} shape={meta.shape}")

# -------------------------------------------------------------------
# 공통 wrapper (ddos / slow_scan)
# -------------------------------------------------------------------
def build_scenario_df(scenario_type: str):
    """
    반환: (X_scen, y_scen, meta_scen)
    """
    df_all, X_cols, meta_cols = _load_test_joined()

    if "Label" not in df_all.columns:
        raise ValueError("Label 컬럼이 없습니다. y를 만들 수 없습니다.")

    if scenario_type == "ddos":
        scen_df = _build_ddos_scenario(df_all, X_cols)
    elif scenario_type == "slow_scan":
        scen_df = _build_slow_scan_scenario(df_all, X_cols)
    else:
        raise ValueError(f"지원하지 않는 시나리오 타입: {scenario_type}")

    # ========================================================
    # f_name을 개별 접근마다 고유하게 변경하는 로직
    # 형식: UNSW-NB15_40.csv, UNSW-NB15_41.csv ...
    # ========================================================
    if "f_name" in scen_df.columns:
        indices = np.arange(len(scen_df))
        scen_df["f_name"] = "UNSW-NB15_4" + pd.Series(indices).astype(str) + ".csv"
        print(f"[{scenario_type}] f_name unique values generated: {len(scen_df)} files.")

    # X_scen / y_scen / meta_scen 분리
    X_scen = scen_df[X_cols].copy()
    y_scen = scen_df[["event_id", "Label"]].copy()

    meta_scen = scen_df[meta_cols].copy()
    meta_scen = meta_scen.set_index("event_id")

    return X_scen, y_scen, meta_scen


def main():
    # 1) 먼저 test_X / test_y / test_meta 생성
    if not (ART / "test_X.parquet").exists():
        prepare_test_parquets()
    else:
        print("[info] test_X.parquet already exists. Skip test preprocessing.")

    # 2) DDoS / Slow Scan 시나리오 생성
    scenarios = ["ddos", "slow_scan"]

    for name in scenarios:
        X_scen, y_scen, meta_scen = build_scenario_df(name)

        x_path = ART / f"scenario_{name}_X.parquet"
        y_path = ART / f"scenario_{name}_y.parquet"
        m_path = ART / f"scenario_{name}_meta.parquet"

        X_scen.to_parquet(x_path, index=False)
        y_scen.to_parquet(y_path, index=False)
        meta_scen.to_parquet(m_path)

        print(f"[+] saved {name}:")
        print(f"    X   -> {x_path} shape={X_scen.shape}")
        print(f"    y   -> {y_path} shape={y_scen.shape}")
        print(f"    meta-> {m_path} shape={meta_scen.shape}")

    # 3) 결측치/누락열 시나리오 생성
    build_missing_feature_scenario()


if __name__ == "__main__":
    main()

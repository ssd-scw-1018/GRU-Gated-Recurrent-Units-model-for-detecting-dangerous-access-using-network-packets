# test.py
from pathlib import Path
from collections import defaultdict, deque
import numpy as np
import pandas as pd
import torch
import shutil 
import os
import torch.nn as nn

# 시각화용 라이브러리
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# ================== 설정 및 상수 ==================
ART = Path("artifacts_parquet")

# 🔹 fake 파일 루트 디렉토리 (시나리오별 서브폴더 생성 예정)
FAKE_ROOT = Path("fake_files")

# [설정] 가짜 파일을 최대 몇 개까지 유지할 것인가?
MAX_FAKE_FILES = 10

# [설정] 동일 로그 반복 출력 제한 횟수
LOG_REPEAT_LIMIT = 5

# [설정] 시나리오별 최대 로그 출력 줄 수
LOG_MAX_LINES = 100

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def list_scenario_splits():
    """
    ART 디렉토리 안의 *_X.parquet 중에서
    - test_X.parquet 은 제외하고
    - 나머지 파일들의 prefix( *_X 앞 부분 )를 시나리오 이름으로 인식한다.

    예)
      scenario_ddos_X.parquet   -> "scenario_ddos"
      scenario_slow_scan_X.parquet -> "scenario_slow_scan"
      scenario_missing_X.parquet   -> "scenario_missing"
    """
    splits = []
    for p in ART.glob("*_X.parquet"):
        stem = p.stem             # 예: 'scenario_ddos_X', 'test_X'
        if stem == "test_X":
            # test_X.parquet은 “학습/평가용 원본 테스트셋”이라 시나리오 아님
            continue

        # 뒤의 "_X" 떼기
        if stem.endswith("_X"):
            split_name = stem[:-2]  # 'scenario_ddos_X' -> 'scenario_ddos'
        else:
            split_name = stem

        if split_name == "test":
            # 혹시라도 'test_X'를 또 잡아도 방어
            continue

        splits.append(split_name)

    # 정렬(선택 사항)
    splits = sorted(set(splits))
    print(f"[info] 발견된 시나리오 splits: {splits}")
    return splits
import json

def data_check_before_run():
    """
    test_X는 검사하지 않음.
    ART 안에 있는 *_X.parquet 중 test_X를 제외한
    모든 시나리오 데이터에 대해:

      - core_features.json 기반 핵심 피처 존재 여부
      - 핵심 피처 결측률 (가능하면 raw JSON 기준)

    을 검사하고, 기준을 통과한 시나리오 이름 리스트를 반환한다.
    """
    print("\n========== [DATA CHECK: SCENARIO DATA] ==========")

    # 0) 현재 존재하는 시나리오 split 자동 수집
    scenario_splits = list_scenario_splits()
    if not scenario_splits:
        print("[!] 시나리오용 *_X.parquet 파일을 찾지 못했습니다.")
        return []

    # 1) 핵심 피처 목록 불러오기
    core_path = Path("core_features.json")
    if not core_path.exists():
        alt_path = ART / "core_features.json"
        if alt_path.exists():
            core_path = alt_path
        else:
            print("[!] core_features.json 없음 → 중요 피처 기반 체크 불가.")
            print("    feature_importance_auc.py 를 먼저 실행해서 core_features.json을 생성하세요.")
            return []

    print(f" - using core_features.json from: {core_path}")
    core = json.load(open(core_path, "r", encoding="utf-8"))
    core_features = core.get("core_features", [])
    if not core_features:
        print("[!] core_features.json 안에 'core_features' 키가 비어 있습니다.")
        return []

    print(f" - 핵심 피처 {len(core_features)}개 로드됨")
    print(f"   {core_features}")

    CRITICAL_THRESHOLD = 0.05  # 핵심 피처 결측률 허용 최대값 (5%)

    valid_scenarios = []
    invalid_scenarios = []

    # ---------------------------
    # 각 시나리오별 검사
    # ---------------------------
    for split in scenario_splits:
        split_ok = True

        x_path = ART / f"{split}_X.parquet"
        print(f"\n--- Checking scenario: {split} ({x_path.name}) ---")

        if not x_path.exists():
            print(f"[!] {x_path.name} 없음 → 이 시나리오는 스킵됨")
            split_ok = False
            invalid_scenarios.append(split)
            continue

        df = pd.read_parquet(x_path)
        df = df.drop(columns=["event_id"], errors="ignore")

        # 1) 핵심 피처 존재 여부 (모델이 실제로 쓸 수 있는지 확인)
        missing_features = [c for c in core_features if c not in df.columns]
        if missing_features:
            print(f"[X] 핵심 피처 누락 → {missing_features}")
            split_ok = False

        else:
            print(" - 핵심 피처 존재 OK")

            # 2) 결측률 체크
            #    2-1) raw 결측률 JSON이 있으면 그걸 우선 사용
            raw_json_path = ART / f"{split}_raw_missing.json"
            if raw_json_path.exists():
                raw_info = json.load(open(raw_json_path, "r", encoding="utf-8"))
                raw_missing = raw_info.get("missing_rate", {})
                use_raw = True
                print(f"   (raw missing 사용: {raw_json_path.name})")
            else:
                # 없으면 전처리 후 파켓 기준으로라도 체크 (fallback)
                miss_rate_series = df.isna().mean()
                raw_missing = {c: float(r) for c, r in miss_rate_series.items()}
                use_raw = False
                print("   (raw JSON 없음 → parquet 기준 결측률 사용)")

            for f in core_features:
                if use_raw:
                    # raw_missing 에 없으면 "열이 아예 없었다"로 보고 100% 결측으로 취급
                    r = float(raw_missing.get(f, 1.0))
                else:
                    r = float(raw_missing.get(f, 0.0))

                if r > CRITICAL_THRESHOLD:
                    print(f"[!] 핵심 피처 '{f}' 결측률 = {r*100:.2f}% "
                          f"(허용 {CRITICAL_THRESHOLD*100:.1f}% 초과)")
                    split_ok = False
                else:
                    print(f" - {f}: 결측률 {r*100:.2f}% OK")

        if split_ok:
            print(f" --> ✅ scenario '{split}' 사용 가능")
            valid_scenarios.append(split)
        else:
            print(f" --> ⚠ scenario '{split}' 는 기준 미달 (실행 대상에서 제외)")
            invalid_scenarios.append(split)

    # 요약 출력
    print("\n[DATA CHECK SUMMARY]")
    print(f" - 사용 가능 시나리오: {valid_scenarios}")
    print(f" - 제외된 시나리오  : {invalid_scenarios}")

    if not valid_scenarios:
        print("[!] 기준을 통과한 시나리오가 없습니다.")

    # ✅ 이제 bool이 아니라 "쓸 수 있는 시나리오 리스트"를 반환
    return valid_scenarios

# ================= Model 정의 =================
class GRUCls(nn.Module):
    def __init__(self, in_dim, hid=128, num_layers=1, dropout=0.3, n_classes=2):
        super().__init__()
        self.gru = nn.GRU(
            in_dim, hid,
            num_layers=num_layers,
            batch_first=True,
            dropout=(dropout if num_layers > 1 else 0.0)
        )
        self.bn = nn.BatchNorm1d(hid)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hid, n_classes)
        )

    def forward(self, x):
        out, _ = self.gru(x)
        last = self.bn(out[:, -1, :])
        return self.fc(last)

def decide_dual(p, lo, hi):
    p = np.asarray(p)
    return np.where(p < lo, 0, np.where(p >= hi, 2, 1))

# ================== 핵심 보안 엔진 ==================
class SecurityPolicyEngine:
    def __init__(self, model, tau_lo, tau_hi, device,
                 fake_dir: Path, watch_threshold=5):
        """
        fake_dir: 이 인스턴스(시나리오)에서 사용할 fake 파일 전용 디렉토리
                  예: fake_files/scenario_ddos, fake_files/scenario_slow_scan ...
        """
        self.model = model
        self.tau_lo = tau_lo
        self.tau_hi = tau_hi
        self.device = device
        self.watch_threshold = watch_threshold

        self.block_list = set()
        self.watch_counts = defaultdict(int)
        self.fake_file_queue = deque()
        self.dummy_content = os.urandom(1024)  # 1KB 더미 데이터

        # 시나리오별 fake 디렉토리
        self.fake_dir = fake_dir

        self._init_environment()

    def _init_environment(self):
        """해당 시나리오 전용 fake_dir만 정리"""
        if self.fake_dir.exists():
            try:
                shutil.rmtree(self.fake_dir, ignore_errors=True)
            except Exception:
                pass
        self.fake_dir.mkdir(parents=True, exist_ok=True)
            
    def _create_dynamic_fake_file(self, requested_fname):
        """해당 시나리오 전용 디렉토리에 fake 파일 생성"""
        target_path = self.fake_dir / requested_fname

        if target_path.exists():
            return str(target_path)

        while len(self.fake_file_queue) >= MAX_FAKE_FILES:
            oldest = self.fake_file_queue.popleft()
            try:
                os.remove(self.fake_dir / oldest)
            except OSError:
                pass

        try:
            with open(target_path, "wb") as f:
                f.write(self.dummy_content)
            self.fake_file_queue.append(requested_fname)
            return str(target_path)
        except Exception:
            return str(target_path)

    def _predict(self, x_seq):
        arr = np.asarray(x_seq, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, None, :]
        elif arr.ndim == 2:
            arr = arr[:, None, :]
        x_tensor = torch.tensor(arr, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.model(x_tensor)
            prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()[0]
        return float(prob)

    def decide(self, ip, f_name, x_seq):
        # 1. 차단된 IP
        if ip in self.block_list:
            return "BLOCKED", None, None, {
            "watch_count": self.watch_counts.get(ip, 0),
            "is_alert_ip": True
            }

        # 2. 모델 추론
        prob = self._predict(x_seq)
        code = decide_dual(np.array([prob]), self.tau_lo, self.tau_hi)[0]

        # 3. 정책 적용
        if code == 2:  # ALERT
            self.block_list.add(ip)
            fake_path = self._create_dynamic_fake_file(f_name)
            return "BLOCKED", fake_path, prob, {
                "watch_count": self.watch_counts.get(ip, 0),
                "is_alert_ip": True
            }

        elif code == 1:  # WATCH
            self.watch_counts[ip] += 1
            if self.watch_counts[ip] >= self.watch_threshold:
                self.block_list.add(ip)
                fake_path = self._create_dynamic_fake_file(f_name)
                return "BLOCKED", fake_path, prob, {
                    "watch_count": self.watch_counts[ip],
                    "is_alert_ip": True
                }
            else:
                return "WATCH", f"real_files/{f_name}", prob, {
                    "watch_count": self.watch_counts[ip],
                    "is_alert_ip": False
                }

        else:  # NORMAL
            return "NORMAL", f"real_files/{f_name}", prob, {
                "watch_count": self.watch_counts.get(ip, 0),
                "is_alert_ip": False
            }

# ================== 시각화 유틸 ==================
def load_data_for_vis():
    RESULT_CSV = ART / "test_decisions.csv"
    LABEL_DATA = ART / "scenario_y.parquet"
    if not RESULT_CSV.exists():
        print("결과 파일(test_decisions.csv)이 없습니다.")
        return None

    df_pred = pd.read_csv(RESULT_CSV)
    if LABEL_DATA.exists():
        df_label = pd.read_parquet(LABEL_DATA)
        df = pd.merge(df_pred, df_label, on="event_id", how="left")
    else:
        df = df_pred
        print("Warning: scenario_y.parquet가 없어 정답 비교(혼동 행렬)는 건너뜁니다.")
    return df

def plot_ip_based_summary(df, title_suffix=""):
    if df.empty:
        return
    ip_states = {}
    for ip, g in df.groupby("ip"):
        decisions = set(g["decision"])
        if "BLOCKED" in decisions:
            ip_states[ip] = "BLOCKED"
        elif "WATCH" in decisions:
            ip_states[ip] = "WATCH"
        else:
            ip_states[ip] = "NORMAL"

    state_counts = pd.Series(list(ip_states.values())).value_counts()
    colors = {'NORMAL': '#2ecc71', 'WATCH': '#f1c40f', 'BLOCKED': '#e74c3c'}
    col_list = [colors.get(x, '#95a5a6') for x in state_counts.index]

    plt.figure(figsize=(8, 6))
    plt.pie(
        state_counts, labels=state_counts.index, autopct='%1.1f%%',
        startangle=140, colors=col_list
    )
    plt.title(f"IP 기준 보안 상태 분포{title_suffix}")
    plt.tight_layout()
    plt.show()

def plot_attack_scenario(df, target_ip, tau_lo, tau_hi, title_suffix=""):
    subset = df[df['ip'] == target_ip].copy()
    if subset.empty:
        return

    # event_id 기준으로 시간 순 정렬
    subset = subset.sort_values('event_id').reset_index(drop=True)

    # 🔹 x축으로 쓸 접근 순서(0,1,2,...) 생성
    subset['seq_idx'] = np.arange(len(subset))

    plt.figure(figsize=(12, 6))
    # 회색 선: 전체 궤적
    plt.plot(
        subset['seq_idx'], subset['prob'],
        label='Attack Probability', color='gray', alpha=0.5
    )

    states = subset['decision'].unique()
    markers = {'NORMAL': 'o', 'WATCH': 'v', 'BLOCKED': 'X'}
    colors  = {'NORMAL': 'green', 'WATCH': 'orange', 'BLOCKED': 'red'}

    for state in states:
        mask = subset['decision'] == state
        plt.scatter(
            subset.loc[mask, 'seq_idx'], subset.loc[mask, 'prob'],
            label=state, marker=markers.get(state, 'o'),
            c=colors.get(state, 'blue'), s=60
        )

    plt.axhline(
        y=tau_hi, color='r', linestyle='--', alpha=0.5,
        label=f'Alert Threshold ({tau_hi:.3f})'
    )
    plt.axhline(
        y=tau_lo, color='y', linestyle='--', alpha=0.5,
        label=f'Watch Threshold ({tau_lo:.3f})'
    )

    plt.title(f"공격 시나리오 분석 - IP: {target_ip}{title_suffix}")
    plt.xlabel("Event Order in Scenario (seq_idx)")  # 🔹 x축 이름
    plt.ylabel("Malicious Probability")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(df, title_suffix=""):
    if 'Label' not in df.columns:
        return

    y_true = df['Label']

    # BLOCKED 만 탐지로 인정
    y_pred = df['decision'].apply(lambda x: 1 if x == 'BLOCKED' else 0)

    print("\n" + "="*40)
    print(f" [System Performance Report - BLOCKED only]{title_suffix}")
    print("="*40)
    print(classification_report(
        y_true, y_pred,
        target_names=['Normal', 'Blocked']
    ))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Greens',
        xticklabels=['Pred Normal/Watch', 'Pred Block'],
        yticklabels=['True Normal', 'True Attack']
    )
    plt.title(f"보안 정책 적용 후 혼동 행렬 (BLOCKED only){title_suffix}")
    plt.ylabel("실제 값 (True)")
    plt.xlabel("시스템 판단 (Pred)")
    plt.tight_layout()
    plt.show()

def pick_most_interesting_ip(df):
    """
    가장 볼만한 IP를 선택한다.
    기준 = NORMAL / WATCH / BLOCKED 사이의 상태 전이가 가장 많은 IP
    """
    transition_scores = {}

    for ip, g in df.groupby("ip"):
        g = g.sort_values("event_id")
        states = g["decision"].tolist()

        # 연속된 상태가 바뀐 횟수 계산
        transitions = sum(1 for i in range(1, len(states)) if states[i] != states[i-1])

        transition_scores[ip] = transitions

    if not transition_scores:
        return None

    # 변화량이 가장 큰 IP를 선택
    return max(transition_scores, key=transition_scores.get)
def run_visualization(tau_lo, tau_hi):
    print("\n[+] 시각화 및 리포트 생성을 시작합니다...")
    df_all = load_data_for_vis()
    if df_all is None:
        return

    for scen in df_all["scenario"].unique():
        print(f"\n===== [Scenario: {scen}] =====")
        df = df_all[df_all["scenario"] == scen].copy()
        suffix = f" ({scen})"

        # IP 기준 Pie Chart
        plot_ip_based_summary(df, title_suffix=suffix)

        # 공격 시나리오 그래프
        target_ip = pick_most_interesting_ip(df)

        if target_ip is not None:
            print(f"[Graph] '{scen}'에서 변화량이 가장 많은 IP '{target_ip}'를 선택하여 그래프 생성")
            plot_attack_scenario(df, target_ip, tau_lo, tau_hi, title_suffix=suffix)
        else:
            print(f"[Info] '{scen}'에서 적절한 IP를 찾지 못해 그래프를 표시하지 않음")


        # 혼동 행렬
        plot_confusion_matrix(df, title_suffix=suffix)

# ================== 데이터 로더 ==================
def build_events_with_meta(split="test"):
    X = pd.read_parquet(ART / f"{split}_X.parquet")
    meta = pd.read_parquet(ART / f"{split}_meta.parquet")

    ev = X["event_id"].astype("int64").to_numpy()
    feats = X.drop(columns=["event_id"]).to_numpy(np.float32)

    if meta.index.name != "event_id":
        meta = meta.set_index("event_id")

    srcip_arr = meta.loc[ev, "srcip"].astype(str).to_numpy()

    if "f_name" in meta.columns:
        fname_arr = meta.loc[ev, "f_name"].astype(str).to_numpy()
    else:
        fname_arr = np.array(["unknown.dat"] * len(ev), dtype=object)

    X_seq = feats[:, None, :].astype(np.float32)
    print(f"[build_events_with_meta] split={split}, X_seq={X_seq.shape}")
    return X_seq, srcip_arr, fname_arr, ev

# ================== 모델 로드 ==================
def load_model(ckpt_path="gru_dual_threshold_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    config = ckpt["config"]
    model = GRUCls(
        in_dim=config["in_dim"],
        hid=config["hid"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        n_classes=config["n_classes"],
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tau_lo = float(ckpt["thresholds"]["tau_lo"])
    tau_hi = float(ckpt["thresholds"]["tau_hi"])

    print(f"[+] model loaded on {device}, tau_lo={tau_lo:.3f}, tau_hi={tau_hi:.3f}")
    return model, tau_lo, tau_hi, device

# ================== 메인 실행부 ==================
# ================== 메인 실행부 ==================
def run_on_test_split(scenario_splits=None):
    # data_check에서 이미 필터링된 리스트를 넘겨줄 수도 있고,
    # 직접 호출할 때는 None으로 두면 전체 리스트 사용
    if scenario_splits is None:
        scenario_splits = list_scenario_splits()

    if not scenario_splits:
        print("[!] 실행할 시나리오 split이 없습니다.")
        return

    model, tau_lo, tau_hi, device = load_model()
    all_logs = []

    for split_name in scenario_splits:
        print("\n" + "=" * 60)
        print(f"[+] Starting Traffic Analysis on scenario: {split_name}")
        print("=" * 60)
        
        # 🔹 시나리오별 fake 파일 디렉토리 (fake_files/scenario_xxx)
        scenario_fake_dir = FAKE_ROOT / split_name

        engine = SecurityPolicyEngine(
            model, tau_lo, tau_hi, device,
            fake_dir=scenario_fake_dir,
            watch_threshold=5,
        )

        X_seq, seq_ips, seq_fnames, seq_eids = build_events_with_meta(split_name)
        decisions_log = []

        # 로그 중복 출력 방지용 변수
        prev_ip = None
        prev_decision = None
        consecutive_count = 0

        # 🔹 이 시나리오에서 실제 콘솔에 찍힌 줄 수
        printed_lines = 0
        log_truncated = False
        
        for i in range(len(X_seq)):
            ip = seq_ips[i]
            f_name = seq_fnames[i]
            x_seq = X_seq[i]
            decision, served_path, prob, state = engine.decide(ip, f_name, x_seq)

            decisions_log.append({
                "scenario": split_name,
                "event_id": int(seq_eids[i]),
                "ip": ip,
                "f_name": f_name,
                "served_file": served_path,
                "decision": decision,
                "prob": prob,
                "watch_count": state["watch_count"],
                "is_alert_ip": state["is_alert_ip"],
            })

            # -----------------------------
            # 로그 출력 로직 (중복 요약 + 최대 줄 수 제한)
            # -----------------------------
            if ip != prev_ip or decision != prev_decision:
                # 상태가 바뀌기 전에, 이전에 쌓인 반복이 많으면 요약 출력
                if consecutive_count > LOG_REPEAT_LIMIT and printed_lines < LOG_MAX_LINES:
                    print(
                        f"   ... [Skipped {consecutive_count - LOG_REPEAT_LIMIT} "
                        f"identical events for {prev_ip} ({prev_decision})] ..."
                    )
                    printed_lines += 1
                prev_ip = ip
                prev_decision = decision
                consecutive_count = 1
            else:
                consecutive_count += 1

            # 이미 이 시나리오에서 로그가 꽉 찼으면 더는 안 찍음
            if printed_lines >= LOG_MAX_LINES:
                log_truncated = True
                continue

            should_print = False
            if decision == "BLOCKED":
                if consecutive_count <= LOG_REPEAT_LIMIT:
                    should_print = True
            elif i < 5 or (i % 500 == 0):
                should_print = True

            if should_print and printed_lines < LOG_MAX_LINES:
                prob_val = prob if prob is not None else 0.0
                if "fake_files" in str(served_path):
                    action_msg = f"DECEPTION! ({f_name} -> {served_path})"
                else:
                    action_msg = f"Access Granted ({f_name})"

                repeat_tag = (
                    f"(Repeat {consecutive_count})"
                    if consecutive_count > 1 and decision == "BLOCKED"
                    else ""
                )
                print(
                    f"[{split_name}][{i}] {ip} -> {decision} "
                    f"(prob={prob_val:.4f}) | {action_msg} {repeat_tag}"
                )
                printed_lines += 1

        # 마지막 구간에 남아있던 중복도 요약
        if consecutive_count > LOG_REPEAT_LIMIT and printed_lines < LOG_MAX_LINES:
            print(
                f"   ... [Skipped {consecutive_count - LOG_REPEAT_LIMIT} "
                f"identical events for {prev_ip} ({prev_decision})] ..."
            )
            printed_lines += 1

        if log_truncated:
            print(f"[Info] Log outputs for '{split_name}' truncated after {LOG_MAX_LINES} lines.\n")

        print(f"\n[+] Scenario '{split_name}' finished")
        print(f"    - Blocked IPs count: {len(engine.block_list)}")
        print(f"    - Current Fake Files: {len(engine.fake_file_queue)}/{MAX_FAKE_FILES}")
        all_logs.extend(decisions_log)

    # 전체 로그 저장
    df_dec = pd.DataFrame(all_logs)
    df_dec.to_csv(ART / "test_decisions.csv", index=False)
    
    # 정답 y 모으기 (선택된 시나리오들만 통합)
    y_list = []
    for split_name in scenario_splits:
        y_path = ART / f"{split_name}_y.parquet"
        if y_path.exists():
            y_df = pd.read_parquet(y_path)
            y_list.append(y_df)

    if y_list:
        df_y_all = pd.concat(y_list, ignore_index=True)
        df_y_all = df_y_all.drop_duplicates("event_id", keep="last")
        df_y_all.to_parquet(ART / "scenario_y.parquet", index=False)

    # 시각화/리포트
    run_visualization(tau_lo, tau_hi)


if __name__ == "__main__":
    valid_scenarios = data_check_before_run()

    if not valid_scenarios:
        print("\n[!] 기준을 통과한 시나리오가 없어 실행을 중단합니다.")
        exit(1)

    # 문제 있는 시나리오는 자동으로 제외하고,
    # valid_scenarios 만 가지고 분석/그래프 수행
    run_on_test_split(valid_scenarios)

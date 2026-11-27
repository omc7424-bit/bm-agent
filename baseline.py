"""② 흐름 복원·정상 기준

- 부품별 통과 순서(이동 기록)로 스테이션 간 연결(앞 → 뒤)을 집계해 흐름 구조를 만든다.
- 앞 기간(기본 70%)으로 스테이션별 정상 기준을 계산한다.
  처리량 평균, 도착 간격의 중앙값·상위 95% 값, 대기 부품 수 중앙값, starving 비율.
- 공장 전체가 동시에 비는 구간(계획 정지 추정)과 라인 전체가 비는 구간(라인 정지)을 찾는다.
  이 구간들은 정지 의심에서 제외한다.

※ 품질 판정(Response)은 여기서 쓰지 않는다. 정상 기준과 정지 판단은 통과 기록으로만 만든다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from metrics import FlowBins, trailing_sum
from stream import TICKS_PER_UNIT

# 같은 공정을 나눠 맡는 병렬 스테이션 (2_architecture.md 데이터 확인 결과)
PARALLEL_GROUPS = [
    ["L0_S2", "L0_S3"], ["L0_S4", "L0_S5"], ["L0_S6", "L0_S7"], ["L0_S9", "L0_S10", "L0_S11"],
    ["L0_S14", "L0_S15"], ["L0_S21", "L0_S22", "L0_S23"], ["L3_S35", "L3_S36"],
]


@dataclass
class BaselineParams:
    split_ratio: float = 0.7          # 앞 70% = 정상 기준, 뒤 30% = 재생·검증
    min_stop_units: float = 0.2       # 정지 의심 최소 지속 (데이터 시간 단위)
    act_multiplier: float = 3.0       # 조치 권고: 상위 95% 값의 이 배수와 '관찰된 정지 길이 중앙값' 중 큰 값 이상
    planned_stop_units: float = 0.5   # 공장 전체가 이 길이 이상 비면 계획 정지로 본다
    # 같은 라인 가동 여부 = 간격 동안 같은 라인 다른 스테이션의 도착 수 ÷ 평소 기대 도착 수
    line_running_ratio: float = 0.5   # 이 이상이면 '라인은 가동 중' → 정지 의심 판정
    line_stop_ratio: float = 0.15     # 이 미만이면 라인 정지로 보고 정지 의심에서 제외 (사이는 판단 보류)


@dataclass
class Baseline:
    params: BaselineParams
    split_bin: int
    R_factory: int
    R_line: dict
    factory_active: np.ndarray        # (N,) 공장 가동 여부
    peer_active: np.ndarray           # (S, N) 같은 라인의 다른 스테이션이 가동 중인지
    line_active: np.ndarray           # (L, N)
    line_ids: list
    peer_counts: np.ndarray           # (S, N) 같은 라인 다른 스테이션의 tick별 도착 수
    peer_rate: np.ndarray             # 앞 기간 가동 tick당 평균 peer 도착 수
    rate_mean: np.ndarray             # tick당 평균 도착 수
    gap_p50: np.ndarray               # 도착 간격 중앙값 (tick, 가동 시간 기준)
    gap_p95: np.ndarray               # 도착 간격 상위 95% 값
    gap_reliable: np.ndarray          # 기준을 만들 만큼 기록이 있었는지
    q_med: np.ndarray
    q_p95: np.ndarray
    starve_base: np.ndarray
    thr_obs: np.ndarray               # 정지 의심(관찰 필요) 판정 간격 (tick)
    thr_act: np.ndarray               # 조치 권고 판정 간격 (tick)
    stop_dur_median: float            # 앞 기간에서 관찰된 정지 의심 구간 길이 중앙값 (tick)
    stop_dur_p75: float
    observed_stops: int
    successors: list                  # 스테이션별 [(다음 스테이션, 비중)]
    predecessors: list
    planned_stops: list = field(default_factory=list)   # [(시작 bin, 끝 bin)]
    line_stops: list = field(default_factory=list)      # [(라인, 시작 bin, 끝 bin)]
    parallel_of: dict = field(default_factory=dict)     # 스테이션 → 같은 병렬 그룹의 인덱스들

    def summary(self, bins: FlowBins) -> dict:
        st = bins.stations
        return {
            "split_time": float(bins.time_of(self.split_bin)),
            "R_factory_units": self.R_factory / TICKS_PER_UNIT,
            "stop_dur_median": round(self.stop_dur_median / TICKS_PER_UNIT, 2),
            "stop_dur_p75": round(self.stop_dur_p75 / TICKS_PER_UNIT, 2),
            "observed_stops": self.observed_stops,
            "stations": [
                {"station": st[i], "rate_mean": round(float(self.rate_mean[i]) * TICKS_PER_UNIT, 2),
                 "gap_p50": round(float(self.gap_p50[i]) / TICKS_PER_UNIT, 2),
                 "gap_p95": round(float(self.gap_p95[i]) / TICKS_PER_UNIT, 2),
                 "reliable": bool(self.gap_reliable[i]),
                 "q_med": round(float(self.q_med[i]), 1), "starve_base": round(float(self.starve_base[i]), 3),
                 "successors": [[st[n], round(w, 3)] for n, w in self.successors[i]]}
                for i in range(len(st))
            ],
        }


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """True 연속 구간 [(시작, 끝 포함)]."""
    m = np.concatenate([[False], mask.astype(bool), [False]])
    d = np.diff(m.astype(np.int8))
    starts = np.nonzero(d == 1)[0]
    ends = np.nonzero(d == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist()))


def _adaptive_window(arrivals: np.ndarray, lo: int = 5, hi: int = 200) -> int:
    """가동 여부를 볼 창 길이: 평소 도착 간격의 99% 값 × 2 (tick)."""
    idx = np.nonzero(arrivals > 0)[0]
    if len(idx) < 3:
        return hi
    return int(np.clip(np.ceil(2 * np.percentile(np.diff(idx), 99)), lo, hi))


def compute_baseline(bins: FlowBins, params: BaselineParams | None = None) -> Baseline:
    p = params or BaselineParams()
    S, N = bins.A.shape
    split = int(N * p.split_ratio)
    min_stop = max(int(round(p.min_stop_units * TICKS_PER_UNIT)), 2)

    # 공장 / 라인 가동 여부 (최근 R tick 안에 도착이 있었는지)
    total = bins.A.sum(0)
    R_f = _adaptive_window(total[:split])
    factory_active = trailing_sum(total, R_f) > 0

    line_ids = sorted(set(bins.lines.tolist()))
    line_active = np.zeros((len(line_ids), N), bool)
    peer_active = np.zeros((S, N), bool)
    peer_counts = np.zeros((S, N), np.int32)
    R_line = {}
    for li, L in enumerate(line_ids):
        members = np.nonzero(bins.lines == L)[0]
        lc = bins.A[members].sum(0)
        peer_counts[members] = lc[None, :] - bins.A[members]
        R_l = _adaptive_window(lc[:split])
        R_line[int(L)] = R_l
        lt = trailing_sum(lc, R_l)
        line_active[li] = lt > 0
        own = trailing_sum(bins.A[members], R_l)
        peer_active[members] = (lt[None, :] - own) > 0

    fa_front = factory_active[:split]
    n_fa = max(int(fa_front.sum()), 1)
    eff = np.cumsum(factory_active)                      # 가동 시간만 세는 시계

    rate_mean = bins.A[:, :split][:, fa_front].sum(1) / n_fa
    peer_rate = peer_counts[:, :split][:, fa_front].sum(1) / n_fa
    q_front = bins.Q[:, :split][:, fa_front] if fa_front.any() else bins.Q[:, :split]
    q_med = np.median(q_front, axis=1) if q_front.shape[1] else np.zeros(S)
    q_p95 = np.percentile(q_front, 95, axis=1) if q_front.shape[1] else np.zeros(S)
    starve_base = ((bins.A[:, :split] == 0) & fa_front[None, :]).sum(1) / n_fa

    gap_p50 = np.full(S, np.nan)
    gap_p95 = np.full(S, np.nan)
    gap_lists, cov_lists = [], []
    for s in range(S):
        a = np.nonzero(bins.A[s, :split] > 0)[0]
        peer_clock = np.cumsum(np.where(factory_active[:split], peer_counts[s, :split], 0))
        g = np.diff(eff[a]) if len(a) > 1 else np.array([], np.int64)
        c = np.diff(peer_clock[a]) if len(a) > 1 else np.array([], np.int64)
        keep = g > 0
        g, c = g[keep], c[keep]
        gap_lists.append(g)
        # 간격 동안 같은 라인 도착 수 ÷ 평소 기대 도착 수 (같은 라인에 다른 스테이션이 없으면 1로 봄)
        cov_lists.append(c / (peer_rate[s] * g) if peer_rate[s] > 0 else np.ones(len(g)))
        if len(g) >= 5:
            gap_p50[s] = np.median(g)
            gap_p95[s] = np.percentile(g, 95)
    reliable = ~np.isnan(gap_p95)
    fallback = np.nanmedian(gap_p95) if reliable.any() else min_stop
    gap_p95 = np.where(reliable, gap_p95, fallback)
    gap_p50 = np.where(reliable, gap_p50, np.nanmedian(gap_p50) if reliable.any() else min_stop / 2)

    thr_obs = np.maximum(np.floor(gap_p95) + 1, min_stop).astype(np.int64)

    # 앞 기간에서 관찰된 정지 의심 구간 길이 → 조치 권고 기준, 최적화의 조치 시간 기준
    durs = []
    for s in range(S):
        g, ratio = gap_lists[s], cov_lists[s]
        if len(g):
            sel = (g >= thr_obs[s]) & (ratio >= p.line_running_ratio)
            durs.append(g[sel])
    durs = np.concatenate(durs) if durs else np.array([])
    if len(durs):
        d_med, d_p75 = float(np.median(durs)), float(np.percentile(durs, 75))
    else:
        d_med, d_p75 = float(2 * min_stop), float(4 * min_stop)
    thr_act = np.maximum(np.ceil(gap_p95 * p.act_multiplier), np.ceil(d_med)).astype(np.int64)
    thr_act = np.maximum(thr_act, thr_obs + 1)

    # 흐름 구조
    out = bins.edges.sum(1)
    inn = bins.edges.sum(0)
    successors, predecessors = [], []
    for s in range(S):
        sh = bins.edges[s] / max(out[s], 1)
        succ = [(int(n), float(sh[n])) for n in np.argsort(-sh)[:3] if sh[n] >= 0.1]
        successors.append(succ)
        shp = bins.edges[:, s] / max(inn[s], 1)
        predecessors.append([(int(n), float(shp[n])) for n in np.argsort(-shp)[:3] if shp[n] >= 0.1])

    planned = [(a, b) for a, b in _runs(~factory_active)
               if b - a + 1 >= int(p.planned_stop_units * TICKS_PER_UNIT)]
    line_stops = []
    for li, L in enumerate(line_ids):
        for a, b in _runs(~line_active[li] & factory_active):
            if b - a + 1 >= min_stop:
                line_stops.append((int(L), a, b))

    idx = {s: i for i, s in enumerate(bins.stations)}
    parallel_of = {}
    for grp in PARALLEL_GROUPS:
        members = [idx[g] for g in grp if g in idx]
        if len(members) >= 2:
            for m in members:
                parallel_of[m] = [x for x in members if x != m]

    return Baseline(
        params=p, split_bin=split, R_factory=R_f, R_line=R_line,
        factory_active=factory_active, peer_active=peer_active, line_active=line_active, line_ids=line_ids,
        peer_counts=peer_counts, peer_rate=peer_rate,
        rate_mean=rate_mean, gap_p50=gap_p50, gap_p95=gap_p95, gap_reliable=reliable,
        q_med=q_med, q_p95=q_p95, starve_base=starve_base, thr_obs=thr_obs, thr_act=thr_act,
        stop_dur_median=d_med, stop_dur_p75=d_p75, observed_stops=int(len(durs)),
        successors=successors, predecessors=predecessors,
        planned_stops=planned, line_stops=line_stops, parallel_of=parallel_of)

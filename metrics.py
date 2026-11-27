"""③ 실시간 지표 계산

통과 기록을 tick(0.01 데이터 시간 단위) 격자로 집계한다.
  A  : 스테이션별 도착 부품 수 (처리량)
  Q  : 대기 부품 수 = 앞 스테이션은 통과했지만 이 스테이션에 아직 도착하지 않은 부품 수
  TS/TC : 스테이션 간 이동 시간 합계 / 건수 (도착 tick 기준)
  TR/DF : train 부품 도착 수 / 그중 불량(Response=1) 수  ※ 품질 참고용. 탐지에는 쓰지 않는다.
  edges : 스테이션 간 이동 건수 (흐름 구조)

집계는 batch 단위로 더해 가므로 원본 전체를 메모리에 올리지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stream import TICKS_PER_UNIT, FlowSource, station_line, to_tick


@dataclass
class FlowBins:
    stations: list[str]
    lines: np.ndarray          # 스테이션별 라인 번호
    t0_tick: int               # bin 0의 절대 tick
    n_bins: int
    A: np.ndarray
    Q: np.ndarray
    TS: np.ndarray
    TC: np.ndarray
    TR: np.ndarray
    DF: np.ndarray
    edges: np.ndarray
    reversed_transitions: int  # 다음 스테이션 진입이 앞 스테이션 이탈보다 이른 기록 수 (데이터 점검용)
    source_key: str = ""

    @property
    def S(self) -> int:
        return len(self.stations)

    def time_of(self, b) -> np.ndarray | float:
        return (np.asarray(b) + self.t0_tick) / TICKS_PER_UNIT

    def bin_of(self, t: float) -> int:
        return int(round(t * TICKS_PER_UNIT)) - self.t0_tick


def build_bins(src: FlowSource, log=print) -> FlowBins:
    stations = src.stations
    S = len(stations)
    t0 = int(round(src.t_min * TICKS_PER_UNIT))
    N = int(round(src.t_max * TICKS_PER_UNIT)) - t0 + 1
    size = S * N
    A = np.zeros(size, np.int64)
    TR = np.zeros(size, np.int64)
    DF = np.zeros(size, np.int64)
    lookup = src.response_lookup

    rows = 0
    for P, Sx, TI, is_test in src.long_batches():
        b = np.clip(to_tick(TI) - t0, 0, N - 1)
        flat = Sx.astype(np.int64) * N + b
        A += np.bincount(flat, minlength=size)
        train = is_test == 0
        TR += np.bincount(flat[train], minlength=size)
        resp = np.full(len(P), -1, np.int8)
        ok = P < len(lookup)
        resp[ok] = lookup[P[ok]]
        DF += np.bincount(flat[train & (resp == 1)], minlength=size)
        rows += len(P)
        log(f"도착 집계 {rows:,}행")

    Qd = np.zeros(S * (N + 1), np.int64)
    TS = np.zeros(size, np.float64)
    TC = np.zeros(size, np.int64)
    edges = np.zeros(S * S, np.int64)
    reversed_n = 0
    rows = 0
    for F, T, TFO, TTI in src.trans_batches():
        bo = np.clip(to_tick(TFO) - t0, 0, N - 1)
        bi = np.clip(to_tick(TTI) - t0, 0, N - 1)
        rev = bi < bo
        reversed_n += int(rev.sum())
        bo = np.minimum(bo, bi)
        T64 = T.astype(np.int64)
        Qd += np.bincount(T64 * (N + 1) + bo, minlength=S * (N + 1))
        Qd -= np.bincount(T64 * (N + 1) + bi, minlength=S * (N + 1))
        flat = T64 * N + bi
        TS += np.bincount(flat, weights=(bi - bo).astype(np.float64), minlength=size)
        TC += np.bincount(flat, minlength=size)
        edges += np.bincount(F.astype(np.int64) * S + T64, minlength=S * S)
        rows += len(F)
        log(f"이동 기록 집계 {rows:,}행")

    Q = np.cumsum(Qd.reshape(S, N + 1), axis=1)[:, :N]
    return FlowBins(
        stations=list(stations),
        lines=np.array([station_line(s) for s in stations], np.int16),
        t0_tick=t0, n_bins=N,
        A=A.reshape(S, N).astype(np.int32), Q=Q.astype(np.int32),
        TS=TS.reshape(S, N).astype(np.float32), TC=TC.reshape(S, N).astype(np.int32),
        TR=TR.reshape(S, N).astype(np.int32), DF=DF.reshape(S, N).astype(np.int32),
        edges=edges.reshape(S, S), reversed_transitions=reversed_n, source_key=src.cache_key)


def save_bins(bins: FlowBins, path: Path) -> None:
    np.savez_compressed(path, stations=np.array(bins.stations), lines=bins.lines,
                        t0_tick=bins.t0_tick, n_bins=bins.n_bins, A=bins.A, Q=bins.Q, TS=bins.TS,
                        TC=bins.TC, TR=bins.TR, DF=bins.DF, edges=bins.edges,
                        reversed_transitions=bins.reversed_transitions, source_key=bins.source_key)


def load_bins(path: Path) -> FlowBins:
    z = np.load(path, allow_pickle=False)
    return FlowBins(stations=[str(s) for s in z["stations"]], lines=z["lines"],
                    t0_tick=int(z["t0_tick"]), n_bins=int(z["n_bins"]), A=z["A"], Q=z["Q"],
                    TS=z["TS"], TC=z["TC"], TR=z["TR"], DF=z["DF"], edges=z["edges"],
                    reversed_transitions=int(z["reversed_transitions"]), source_key=str(z["source_key"]))


# ---------------------------------------------------------------------------
# 슬라이딩 윈도우 / 추세용 계산
# ---------------------------------------------------------------------------
def trailing_sum(x: np.ndarray, R: int) -> np.ndarray:
    """마지막 축 기준으로 최근 R개 값의 합 (현재 포함). 과거 값만 쓰므로 실시간 계산과 같다."""
    x = np.asarray(x)
    cs = np.concatenate([np.zeros(x.shape[:-1] + (1,), np.int64), np.cumsum(x, axis=-1, dtype=np.int64)], axis=-1)
    idx = np.arange(x.shape[-1]) + 1
    lo = np.maximum(idx - R, 0)
    return cs[..., idx] - cs[..., lo]


def window_stats(bins: FlowBins, end_bin: int, W: int, running: np.ndarray | None = None) -> dict:
    """최근 W tick 동안 스테이션별 처리량·평균 대기·이동 시간·starving 비율."""
    lo = max(end_bin - W + 1, 0)
    sl = slice(lo, end_bin + 1)
    A = bins.A[:, sl]
    run = running[sl] if running is not None else np.ones(A.shape[1], bool)
    n_run = max(int(run.sum()), 1)
    tc = bins.TC[:, sl].sum(1)
    return {
        "throughput": A.sum(1),
        "queue": bins.Q[:, sl].mean(1),
        "transit": np.where(tc > 0, bins.TS[:, sl].sum(1) / np.maximum(tc, 1) / TICKS_PER_UNIT, np.nan),
        "starving": ((A == 0) & run).sum(1) / n_run,
    }


def bucket_series(bins: FlowBins, bucket: int) -> dict:
    """추세 그래프용으로 bucket tick씩 묶은 시계열 (스테이션 × 구간)."""
    S, N = bins.A.shape
    nb = int(np.ceil(N / bucket))
    pad = nb * bucket - N

    def agg(x, how):
        xp = np.pad(x.astype(np.float64), ((0, 0), (0, pad)))
        r = xp.reshape(S, nb, bucket)
        return r.sum(2) if how == "sum" else r.mean(2)

    idx = np.arange(N)
    last = np.where(bins.A > 0, idx[None, :], -1)
    last = np.maximum.accumulate(last, axis=1)
    ends = np.minimum((np.arange(nb) + 1) * bucket - 1, N - 1)
    gap_now = np.where(last[:, ends] >= 0, (ends[None, :] - last[:, ends]) / TICKS_PER_UNIT, np.nan)
    tc = agg(bins.TC, "sum")
    return {
        "bucket": bucket,
        "t_end": bins.time_of(ends),
        "throughput": agg(bins.A, "sum").astype(np.float32),
        "gap": gap_now.astype(np.float32),                 # 마지막 도착 이후 경과 간격
        "queue": agg(bins.Q, "mean").astype(np.float32),
        "queue_now": bins.Q[:, ends].astype(np.float32),   # 구간 끝 시각의 대기 부품 수
        "transit": np.where(tc > 0, agg(bins.TS, "sum") / np.maximum(tc, 1) / TICKS_PER_UNIT, np.nan).astype(np.float32),
    }


def edge_series(trans_batches, bins: FlowBins, edge_from: np.ndarray, edge_to: np.ndarray, bucket: int) -> np.ndarray:
    """흐름 그래프의 연결선(앞 → 뒤)별로, 구간마다 실제로 이동한 부품 수 (다음 스테이션 도착 시각 기준)."""
    S, N = bins.S, bins.n_bins
    nb = int(np.ceil(N / bucket))
    E = len(edge_from)
    lookup = np.full(S * S, -1, np.int64)
    lookup[edge_from.astype(np.int64) * S + edge_to.astype(np.int64)] = np.arange(E)
    out = np.zeros(E * nb, np.int64)
    for F, T, _TFO, TTI in trans_batches():
        e = lookup[F.astype(np.int64) * S + T.astype(np.int64)]
        keep = e >= 0
        k = np.clip(to_tick(TTI[keep]) - bins.t0_tick, 0, N - 1) // bucket
        out += np.bincount(e[keep] * nb + k, minlength=E * nb)
    return out.reshape(E, nb).astype(np.float32)

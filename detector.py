"""④ 병목·정지 의심 탐지

재생 기간을 tick 순서대로 따라가며 판정한다 (각 tick에서는 그 시각까지의 기록만 사용).

정지 의심
  - 같은 라인은 가동 중인데 해당 스테이션의 도착 간격이 정상 기준(상위 95% 값)을 넘고
    최소 지속 시간 이상 이어지며, 앞에 대기 부품이 있을 때.
  - 공장 전체가 비는 구간(계획 정지)은 간격을 세지 않는다.
  - 라인 전체가 비는 구간은 라인 정지로 따로 기록하고 정지 의심에서 제외한다.
  - 같은 라인 기록이 부분적으로만 있으면 '판단 보류'(기록 누락 가능, 검토 필요).
병목
  - 스테이션 앞 대기 증가 + 바로 뒤 스테이션 starving 증가가 함께 나타날 때.
  - Active Period Method(쉬지 않고 일한 시간이 가장 긴 스테이션)로 교차 확인한다.

이벤트는 새로 생기거나 등급이 오르거나 영향 범위가 넓어질 때만 기록(snapshot)한다.
※ 품질 판정(Response)은 쓰지 않는다. 고장 기록이 없으므로 '고장'이 아니라 '정지 의심'이다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from baseline import Baseline
from metrics import FlowBins, trailing_sum
from stream import TICKS_PER_UNIT

GRADES = ["정상", "관찰 필요", "조치 권고", "판단 보류"]
GRADE_RANK = {"정상": 0, "판단 보류": 1, "관찰 필요": 2, "조치 권고": 3}
TYPE_STOP = "정지 의심"
TYPE_BN = "병목"


@dataclass
class DetectorParams:
    eval_every: int = 10          # 병목 판정 주기 (tick)
    window: int = 50              # 병목 판정 창 (tick)
    bn_q_obs: float = 1.5         # 앞 대기가 평소의 몇 배 이상이면 병목 '관찰 필요'
    bn_q_act: float = 2.5         # 몇 배 이상이면 '조치 권고'
    bn_starve_delta: float = 0.2  # 뒤 스테이션 starving 비율이 평소보다 이만큼 늘어야 함
    bn_self_working: float = 0.05 # 병목 자신은 일하는 중: 자기 starving 증가가 이 값 이하
    bn_min_persist: int = 2       # 병목 조건이 연속 몇 번 판정돼야 이벤트로 만들지
    apm_window: int = 3           # APM에서 '일하는 중'으로 볼 최근 도착 창 (tick)
    validate_block: int = 100     # APM 비교 구간 (tick)


def _u(t: float) -> float:
    return round(float(t) / TICKS_PER_UNIT, 2)


class Detector:
    def __init__(self, bins: FlowBins, base: Baseline, params: DetectorParams | None = None):
        self.b = bins
        self.base = base
        self.p = params or DetectorParams()
        S = bins.S
        self.gap = np.zeros(S, np.int64)       # 마지막 도착 이후 경과 (가동 시간 기준 tick)
        self.pcnt = np.zeros(S, np.int64)      # 그동안 같은 라인 다른 스테이션에 도착한 부품 수
        self.open = {}                         # (station, type) → event
        self.events: list[dict] = []
        self.alerts: list[dict] = []           # 알림 대상 변화 (new / upgrade / spread)
        self.repeat = {}
        self.apm_records = []                  # (bin, 탐지 병목, APM 병목)
        self.bn_streak = np.zeros(S, np.int64)  # 병목 조건 연속 판정 횟수
        # APM: '일하는 중' = 최근 apm_window tick 안에 도착이 있음. 쉬지 않고 일한 시간이 가장 긴 스테이션이 병목.
        #      앞 스테이션이 없는 라인 입구(투입량이 곧 활동)와 처리 비중 1% 미만 스테이션은 비교에서 뺀다.
        #      ※ 통과 기록에는 막힘(blocking)이 남지 않아, 공급이 끊기지 않는 앞쪽 스테이션이 APM 1위가 되기 쉽다.
        self.active_run = np.zeros(S, np.int64)
        self.station_active = trailing_sum(bins.A, self.p.apm_window) > 0
        share = base.rate_mean / max(base.rate_mean.sum(), 1e-9)
        self.apm_eligible = np.array([len(base.predecessors[s]) > 0 and share[s] >= 0.01 for s in range(S)])

    # ------------------------------------------------------------------
    def _snapshot(self, ev: dict, tick: int, kind: str) -> None:
        snap = {k: ev[k] for k in ("grade", "delayed_parts", "affected_stations", "evidence", "repeat_count")}
        snap["tick"] = int(tick)
        snap["time"] = _u(tick + self.b.t0_tick)
        snap["kind"] = kind
        ev["snapshots"].append(snap)
        if ev["grade"] != "정상":
            self.alerts.append({"event_id": ev["id"], "tick": int(tick), "time": snap["time"], "kind": kind,
                                "snapshot": len(ev["snapshots"]) - 1})

    def _new_event(self, s: int, typ: str, tick: int, start_tick: int, grade: str, delayed: int,
                   affected: list, evidence: str) -> dict:
        key = (s, typ)
        self.repeat[key] = self.repeat.get(key, 0) + 1
        ev = {"id": len(self.events), "station": self.b.stations[s], "station_idx": s, "type": typ,
              "grade": grade, "start_tick": int(start_tick), "start_time": _u(start_tick + self.b.t0_tick),
              "end_tick": None, "end_time": None, "duration": None, "delayed_parts": int(delayed),
              "affected_stations": affected, "repeat_count": self.repeat[key], "evidence": evidence,
              "peak_grade": grade, "snapshots": []}
        self.events.append(ev)
        self.open[key] = ev
        self._snapshot(ev, tick, "new")
        return ev

    def _update(self, ev: dict, tick: int, grade: str, delayed: int, affected: list, evidence: str) -> None:
        ev["delayed_parts"] = max(ev["delayed_parts"], int(delayed))
        ev["evidence"] = evidence
        kind = None
        if GRADE_RANK[grade] > GRADE_RANK[ev["grade"]]:      # 등급은 열려 있는 동안 내려가지 않는다
            kind = "upgrade"
            ev["grade"] = grade
        elif set(affected) - set(ev["affected_stations"]):
            kind = "spread"
        ev["affected_stations"] = sorted(set(ev["affected_stations"]) | set(affected))
        if GRADE_RANK[ev["grade"]] > GRADE_RANK[ev["peak_grade"]]:
            ev["peak_grade"] = ev["grade"]
        if kind:
            self._snapshot(ev, tick, kind)

    def _close(self, key, tick: int, duration_ticks: int | None = None) -> None:
        ev = self.open.pop(key)
        ev["end_tick"] = int(tick)
        ev["end_time"] = _u(tick + self.b.t0_tick)
        d = duration_ticks if duration_ticks is not None else tick - ev["start_tick"]
        ev["duration"] = _u(d)

    def _starving_successors(self, s: int, tick: int) -> tuple[list, list]:
        lo = max(tick - self.p.window + 1, 0)
        run = self.base.factory_active[lo:tick + 1]
        n = max(int(run.sum()), 1)
        out, txt = [], []
        for nxt, _w in self.base.successors[s]:
            st = float(((self.b.A[nxt, lo:tick + 1] == 0) & run).sum() / n)
            if st - self.base.starve_base[nxt] >= self.p.bn_starve_delta:
                out.append(self.b.stations[nxt])
                txt.append(f"뒤 스테이션 {self.b.stations[nxt]} starving {round(st * 100)}%")
        return out, txt

    # ------------------------------------------------------------------
    def _eval_stop(self, s: int, tick: int) -> None:
        g = int(self.gap[s])
        key = (s, TYPE_STOP)
        ev = self.open.get(key)
        # 같은 라인 가동 정도 = 간격 동안 같은 라인 다른 스테이션 도착 수 ÷ 평소 기대 도착 수
        pr = self.base.peer_rate[s]
        cov = self.pcnt[s] / (pr * g) if pr > 0 else 1.0
        P = self.base.params
        if cov < P.line_stop_ratio:
            return                                  # 라인 정지 구간 → 정지 의심 아님
        q = int(self.b.Q[s, tick])
        ratio = round(g / max(self.base.gap_p95[s], 1), 1)
        if cov < P.line_running_ratio:
            grade = "판단 보류"
            evidence = f"같은 라인 다른 스테이션 도착이 평소 기대의 {round(cov * 100)}%뿐, 라인 가동 여부 불분명·기록 누락 가능"
            affected = []
        else:
            if q <= 0 and ev is None:
                return                              # 기다리는 부품이 없으면 공급 없음(정지 의심 아님)
            grade = "조치 권고" if g >= self.base.thr_act[s] else "관찰 필요"
            affected, txt = self._starving_successors(s, tick)
            evidence = ", ".join([f"도착 간격 평소 상위 95% 값의 {ratio}배", f"앞 대기 {q}개",
                                  f"같은 라인 가동 중(평소 기대 도착의 {round(cov * 100)}%)"] + txt)
        if ev is None:
            self._new_event(s, TYPE_STOP, tick, tick - g + 1, grade, q, affected, evidence)
        else:
            self._update(ev, tick, grade, q, affected, evidence)

    def _eval_bottleneck(self, tick: int, ws: dict) -> int:
        S = self.b.S
        q_ratio = ws["queue"] / np.maximum(self.base.q_med, 1.0)
        # 병목 자신은 쉬지 않고 일하는 중이어야 한다 (자기 starving이 평소와 비슷)
        working = ws["starving"] - self.base.starve_base <= self.p.bn_self_working
        best, best_score = -1, 0.0
        cand = set(np.nonzero(q_ratio >= self.p.bn_q_obs * 0.8)[0].tolist())
        self.bn_streak[[s for s in range(S) if s not in cand]] = 0
        for s in sorted(cand):
            key = (s, TYPE_BN)
            ev = self.open.get(key)
            if not self.base.peer_active[s, tick] or not working[s]:
                cond = False
            else:
                affected, txt = self._starving_successors(s, tick)
                cond = q_ratio[s] >= self.p.bn_q_obs and bool(affected)
            self.bn_streak[s] = self.bn_streak[s] + 1 if cond else 0
            if not cond:
                if ev is not None:
                    self._close(key, tick)
                continue
            score = float(q_ratio[s])
            if score > best_score:
                best, best_score = int(s), score
            if ev is None and self.bn_streak[s] < self.p.bn_min_persist:
                continue                            # 한 번 반짝인 것은 이벤트로 만들지 않는다
            grade = "조치 권고" if q_ratio[s] >= self.p.bn_q_act else "관찰 필요"
            delayed = int(round(max(ws["queue"][s] - self.base.q_med[s], 0)))
            parts = [f"앞 대기 {round(float(q_ratio[s]), 1)}배 증가"] + txt
            par = self.base.parallel_of.get(int(s))
            if par:
                grp = [int(s)] + par
                now_share = ws["throughput"][s] / max(ws["throughput"][grp].sum(), 1)
                base_share = self.base.rate_mean[s] / max(self.base.rate_mean[grp].sum(), 1e-9)
                if base_share > 0 and now_share < 0.5 * base_share:
                    parts.append(f"병렬 스테이션 대비 처리 비중 평소의 {round(now_share / base_share, 1)}배")
            apm_top = self._apm_rank()[:3]
            parts.append("APM 교차 확인 일치" if apm_top and apm_top[0] == s else
                         ("APM 상위 3개 안" if s in apm_top else "APM 교차 확인 불일치"))
            evidence = ", ".join(parts)
            if ev is None:
                start = tick - (self.bn_streak[s] - 1) * self.p.eval_every
                self._new_event(int(s), TYPE_BN, tick, start, grade, delayed, affected, evidence)
            else:
                self._update(ev, tick, grade, delayed, affected, evidence)
        # 열린 병목 중 이번 후보에 없던 것은 닫는다
        for key in [k for k in self.open if k[1] == TYPE_BN]:
            if key[0] not in cand:
                self._close(key, tick)
        return best

    def _apm_rank(self) -> list[int]:
        """APM 순위: 쉬지 않고 일한 시간이 긴 순서 (라인 입구 제외)."""
        run = np.where(self.apm_eligible, self.active_run, 0)
        if run.max() <= 0:
            return []
        order = np.argsort(-run, kind="stable")
        return [int(x) for x in order[:3] if run[x] > 0]

    # ------------------------------------------------------------------
    def run(self, start_bin: int, end_bin: int, log=None) -> None:
        from metrics import window_stats
        A = self.b.A
        fa_all = self.base.factory_active
        peer = self.base.peer_counts
        thr_obs = self.base.thr_obs
        thr_act = self.base.thr_act
        E = self.p.eval_every
        total = end_bin - start_bin
        # 재생 시작 전 기록으로 '마지막 도착 이후 경과'를 이어받는다 (재생 시작 시각에 간격이 0이 되지 않게)
        if start_bin > 0:
            eff = np.cumsum(fa_all[:start_bin])
            for s in range(self.b.S):
                a = np.nonzero(A[s, :start_bin] > 0)[0]
                if len(a):
                    last = a[-1]
                    peer_clock = np.cumsum(np.where(fa_all[:start_bin], peer[s, :start_bin], 0))
                    self.gap[s] = eff[-1] - eff[last]
                    self.pcnt[s] = peer_clock[-1] - peer_clock[last]
        for i, t in enumerate(range(start_bin, end_bin)):
            arr = A[:, t] > 0
            fa = bool(fa_all[t])
            # 도착한 스테이션: 열린 정지 의심 종료, 간격 초기화
            for s in np.nonzero(arr)[0]:
                key = (int(s), TYPE_STOP)
                if key in self.open:
                    self._close(key, t, duration_ticks=int(self.gap[s]))
            self.gap[arr] = 0
            self.pcnt[arr] = 0
            if fa:
                idle = ~arr
                self.gap[idle] += 1
                self.pcnt[idle] += peer[idle, t]
                g = self.gap
                due = (g >= thr_obs) & ((g == thr_obs) | (g == thr_act) | (g % E == 0))
                for s in np.nonzero(due)[0]:
                    self._eval_stop(int(s), t)
                # APM 활동: 최근 apm_window tick 안에 도착이 있으면 '일하는 중' (공장 휴무 tick은 건너뜀)
                self.active_run = (self.active_run + 1) * self.station_active[:, t]
            if fa and (t - start_bin) % E == 0 and t >= self.p.window:
                ws = window_stats(self.b, t, self.p.window, fa_all)
                ours = self._eval_bottleneck(t, ws)
                rank = self._apm_rank()
                self.apm_records.append((t, ours, rank[0] if rank else -1,
                                         rank[1] if len(rank) > 1 else -1, rank[2] if len(rank) > 2 else -1))
            if log and i % 5000 == 0:
                log(f"탐지 진행 {i:,}/{total:,} tick, 이벤트 {len(self.events)}건")

    # ------------------------------------------------------------------
    def validate_apm(self) -> dict:
        """구간별로 '탐지한 병목'과 'APM 병목'(가장 오래 쉬지 않고 일한 스테이션)을 비교한다."""
        if not self.apm_records:
            return {"blocks": 0, "compared": 0, "match_rate": None, "top3_rate": None}
        rec = np.array(self.apm_records)
        blk = rec[:, 0] // self.p.validate_block
        compared = matched = top3 = 0
        blocks = np.unique(blk)
        for bk in blocks:
            r = rec[blk == bk]
            ours = r[r[:, 1] >= 0, 1]
            apm1 = r[r[:, 2] >= 0, 2]
            if not len(ours) or not len(apm1):
                continue
            compared += 1
            o = np.bincount(ours).argmax()                 # 구간에서 가장 자주 병목으로 판정한 스테이션
            cnt = np.bincount(r[:, 2:5][r[:, 2:5] >= 0])   # APM 상위 3 등장 횟수
            a1 = np.bincount(apm1).argmax()                # 구간 APM 1위
            a3 = [x for x in np.argsort(-cnt)[:3] if cnt[x] > 0]
            matched += int(o == a1)
            top3 += int(o in a3)
        # 판정 시점 단위 비교: 병목을 판정한 시점에 그 스테이션이 APM 1위 / 상위 3 안인지
        flagged = rec[rec[:, 1] >= 0]
        pt1 = float((flagged[:, 1] == flagged[:, 2]).mean()) if len(flagged) else None
        pt3 = float((flagged[:, 1:2] == flagged[:, 2:5]).any(1).mean()) if len(flagged) else None
        return {"blocks": int(len(blocks)), "compared": compared,
                "match_rate": round(matched / compared * 100, 1) if compared else None,
                "top3_rate": round(top3 / compared * 100, 1) if compared else None,
                "point_match_rate": round(pt1 * 100, 1) if pt1 is not None else None,
                "point_top3_rate": round(pt3 * 100, 1) if pt3 is not None else None,
                "flag_evals": int(len(flagged)), "evals": int(len(rec)),
                "block_units": _u(self.p.validate_block)}

    def finalize(self, end_bin: int) -> None:
        """재생 끝에서도 열려 있는 이벤트는 '진행 중'으로 남긴다 (끝 시각 없음)."""
        for ev in self.open.values():
            ev["duration"] = _u(end_bin - ev["start_tick"])
            ev["status_at_end"] = "진행 중"


def grade_at(ev: dict, tick: int) -> dict | None:
    """tick 시점에 보이는 이벤트 상태 (snapshot 기준). 아직 발생 전이면 None."""
    snap = None
    for sn in ev["snapshots"]:
        if sn["tick"] <= tick:
            snap = sn
        else:
            break
    return snap


def is_open_at(ev: dict, tick: int) -> bool:
    if not ev["snapshots"] or ev["snapshots"][0]["tick"] > tick:
        return False
    return ev["end_tick"] is None or ev["end_tick"] > tick

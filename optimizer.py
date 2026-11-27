"""⑤ 최적화 (BM 대응 인원·순서)

자원 제약 일정 모델 (Google OR-Tools CP-SAT)
  결정변수 : 각 이벤트의 대응 시작 시각, 투입 여부
  제약     : 같은 시각에 투입된 인원 합 ≤ 전체 인원, 이벤트별 필요 인원 충족
  목적함수 : 누적 손실 = Σ 지연 부품 수 × (대응 완료 시각 − 발생 시각)  → 최소화
비교 기준 : 같은 인원으로 발생 순서대로 대응(FIFO)한 경우의 누적 손실
인원 수별 비교 : 전체 인원을 여러 값으로 바꿔 풀고, 추가 인원 효과가 줄어드는 지점을 찾는다.

※ 전체 인원·등급별 필요 인원은 데이터가 아니라 사용자 설정값이다.
  조치 시간은 앞 기간에서 관찰된 정지 의심 구간 길이(중앙값 / 상위 75% 값)를 쓴다.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

from stream import TICKS_PER_UNIT

STAFFED_GRADES = ("관찰 필요", "조치 권고")   # 판단 보류는 인원 투입 대신 '검토 필요'


@dataclass
class Rules:
    total_staff: int = 20
    staff_obs: int = 1            # 관찰 필요 필요 인원 (설정값)
    staff_act: int = 3            # 조치 권고 필요 인원 (설정값)
    dur_obs: int = 10             # 관찰 필요 조치 시간 (tick)
    dur_act: int = 20             # 조치 권고 조치 시간 (tick)

    def demand(self, grade: str) -> int:
        d = self.staff_act if grade == "조치 권고" else self.staff_obs
        return max(1, min(d, self.total_staff))

    def duration(self, grade: str) -> int:
        return max(1, self.dur_act if grade == "조치 권고" else self.dur_obs)


def make_jobs(events: list[dict], rules: Rules, release_now: int | None = None, grade_key: str = "grade") -> list[dict]:
    jobs = []
    for ev in events:
        g = ev[grade_key]
        if g not in STAFFED_GRADES:
            continue
        r = release_now if release_now is not None else ev["start_tick"]
        jobs.append({"id": ev["id"], "station": ev["station"], "grade": g,
                     "weight": max(int(ev["delayed_parts"]), 1), "release": int(r),
                     "demand": rules.demand(g), "dur": rules.duration(g), "start_tick": ev["start_tick"]})
    return jobs


def _loss(jobs, ends) -> float:
    return sum(j["weight"] * (ends[j["id"]] - j["release"]) for j in jobs) / TICKS_PER_UNIT


def _result(jobs, starts, rules, status, t0_tick=0) -> dict:
    ends = {j["id"]: starts[j["id"]] + j["dur"] for j in jobs}
    order = sorted(jobs, key=lambda j: (starts[j["id"]], -j["weight"]))
    assign = [{"event_id": j["id"], "station": j["station"], "grade": j["grade"], "staff": j["demand"],
               "order": k + 1, "start": round((starts[j["id"]] + t0_tick) / TICKS_PER_UNIT, 2),
               "end": round((ends[j["id"]] + t0_tick) / TICKS_PER_UNIT, 2),
               "delayed_parts": j["weight"]} for k, j in enumerate(order)]
    return {"assignments": assign, "total_loss": round(_loss(jobs, ends), 2), "status": status,
            "total_staff": rules.total_staff}


def _list_schedule(jobs: list[dict], rules: Rules, key) -> dict:
    """정해진 우선순위대로, 인원이 비는 가장 이른 시각에 하나씩 배치한다 (시작 tick 사전)."""
    placed = []   # (start, end, demand)
    starts = {}
    for j in sorted(jobs, key=key):
        cands = sorted({j["release"]} | {e for _, e, _ in placed if e > j["release"]})
        for t in cands:
            pts = {t} | {s for s, _, _ in placed if t < s < t + j["dur"]}
            if all(sum(d for s, e, d in placed if s <= p < e) + j["demand"] <= rules.total_staff for p in pts):
                starts[j["id"]] = t
                placed.append((t, t + j["dur"], j["demand"]))
                break
    return starts


def fifo_schedule(jobs: list[dict], rules: Rules, t0_tick: int = 0) -> dict:
    """발생 순서대로, 인원이 비는 가장 이른 시각에 대응한다 (비교 기준)."""
    starts = _list_schedule(jobs, rules, key=lambda j: (j["release"], j["start_tick"], j["id"]))
    return _result(jobs, starts, rules, "FIFO", t0_tick)


def wspt_schedule(jobs: list[dict], rules: Rules, t0_tick: int = 0) -> dict:
    """지연 부품 수 ÷ (인원 × 조치 시간)이 큰 순서 (CP-SAT 시작점용 휴리스틱)."""
    starts = _list_schedule(jobs, rules, key=lambda j: (-j["weight"] / (j["demand"] * j["dur"]), j["release"], j["id"]))
    return _result(jobs, starts, rules, "WSPT", t0_tick)


def optimize(jobs: list[dict], rules: Rules, time_limit: float = 5.0, t0_tick: int = 0, workers: int = 4) -> dict:
    """CP-SAT으로 누적 손실 최소 일정. 발생 순·WSPT 중 좋은 해를 시작점(hint)으로 넣고,
    결과가 그보다 나쁘면(시간 제한) 휴리스틱 해를 쓴다 → 발생 순보다 나빠지지 않는다."""
    if not jobs:
        return {"assignments": [], "total_loss": 0.0, "status": "이벤트 없음", "total_staff": rules.total_staff}
    if sum(j["demand"] for j in jobs) <= rules.total_staff and len({j["release"] for j in jobs}) == 1:
        # 모두 동시에 투입 가능 → 바로 시작이 최적. 순서는 지연 부품이 많은 순
        r = jobs[0]["release"]
        return _result(jobs, {j["id"]: r for j in jobs}, rules, "OPTIMAL(인원 충분)", t0_tick)

    from ortools.sat.python import cp_model

    heur = min((fifo_schedule(jobs, rules), wspt_schedule(jobs, rules)), key=lambda r: r["total_loss"])
    fifo_start = {a["event_id"]: int(round(a["start"] * TICKS_PER_UNIT)) for a in heur["assignments"]}
    horizon = max(j["release"] for j in jobs) + sum(j["dur"] for j in jobs) + 1

    m = cp_model.CpModel()
    starts, ends, intervals, demands, obj = {}, {}, [], [], []
    for j in jobs:
        i = j["id"]
        pres = m.NewBoolVar(f"p{i}")
        s = m.NewIntVar(j["release"], horizon, f"s{i}")
        e = m.NewIntVar(j["release"] + j["dur"], horizon + j["dur"], f"e{i}")
        iv = m.NewOptionalIntervalVar(s, j["dur"], e, pres, f"iv{i}")
        m.Add(e == s + j["dur"])                           # 투입하지 않아도 완료 시각은 시작 + 조치 시간
        m.Add(s == horizon).OnlyEnforceIf(pres.Not())     # 투입하지 않으면 끝까지 손실이 쌓인다
        intervals.append(iv)
        demands.append(j["demand"])
        starts[i], ends[i] = s, e
        obj.append(j["weight"] * (e - j["release"]))
        m.AddHint(s, fifo_start[i])
        m.AddHint(pres, 1)
    m.AddCumulative(intervals, demands, rules.total_staff)
    m.Minimize(sum(obj))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = workers
    st = solver.Solve(m)
    heur_t = min((fifo_schedule(jobs, rules, t0_tick), wspt_schedule(jobs, rules, t0_tick)),
                 key=lambda r: r["total_loss"])
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        heur_t["status"] = f"해 없음 → {heur_t['status']} 사용"
        return heur_t
    sol = {i: solver.Value(starts[i]) for i in starts}
    res = _result(jobs, sol, rules, "OPTIMAL" if st == cp_model.OPTIMAL else "FEASIBLE(시간 제한)", t0_tick)
    if heur_t["total_loss"] < res["total_loss"]:      # 시간 제한으로 휴리스틱보다 나쁘면 휴리스틱 해 유지
        heur_t["status"] = f"{heur_t['status']}(시간 제한 내 CP-SAT보다 좋음)"
        return heur_t
    return res


def compare(jobs: list[dict], rules: Rules, time_limit: float = 10.0, t0_tick: int = 0) -> dict:
    opt = optimize(jobs, rules, time_limit, t0_tick)
    base = fifo_schedule(jobs, rules, t0_tick)
    bl = base["total_loss"]
    # 기존 방식 손실이 0이면(비교할 대응 없음) 감소율을 계산하지 않는다
    red = round((bl - opt["total_loss"]) / bl * 100, 1) if bl > 0 else None
    return {"optimized": opt, "fifo": base, "total_loss": opt["total_loss"], "baseline_loss": bl,
            "reduction_pct": red, "reduction_note": None if bl > 0 else "비교할 대응 없음",
            "jobs": len(jobs), "rules": asdict(rules)}


def staff_curve(jobs: list[dict], rules: Rules, staff_values: list[int], time_limit: float = 5.0) -> dict:
    """전체 인원별 누적 손실 곡선과 적정 BM 인원(추가 인원 효과가 줄어드는 지점)."""
    rows = []
    for n in sorted(set(int(v) for v in staff_values if int(v) > 0)):
        r = Rules(**{**asdict(rules), "total_staff": n})
        jb = [{**j, "demand": r.demand(j["grade"])} for j in jobs]
        o = optimize(jb, r, time_limit)
        f = fifo_schedule(jb, r)
        rows.append({"total_staff": n, "optimized_loss": o["total_loss"], "fifo_loss": f["total_loss"]})
    # 적정 인원 = 아래 중 먼저 만족하는 인원
    #   ① 그 인원의 손실이 비교 범위 최소 손실과 2% 이내 (더 늘려도 거의 줄지 않음)
    #   ② 다음 증원의 1명당 손실 감소가 가장 큰 감소의 30% 미만
    knee = None
    reason = ""
    if not jobs or not rows or len({r["optimized_loss"] for r in rows}) == 1:
        # 이벤트가 없거나 모든 인원에서 누적 지연이 같으면 적정 인원을 정하지 않는다
        return {"rows": rows, "knee": None, "knee_reason": "비교할 이벤트 없음 - 적정 인원 판단 불가"}
    if rows:
        losses = [r["optimized_loss"] for r in rows]
        lmin = min(losses)
        gains = [(a["optimized_loss"] - b["optimized_loss"]) / max(b["total_staff"] - a["total_staff"], 1)
                 for a, b in zip(rows, rows[1:])]
        top = max(gains) if gains else 0
        for k, r in enumerate(rows[:-1]):          # 마지막 값은 '더 늘려 봤을 때'를 알 수 없으므로 판정하지 않음
            if r["optimized_loss"] <= lmin * 1.02:
                knee = r["total_staff"]
                reason = "이 인원부터 손실이 비교 범위 최소값과 2% 이내 (더 늘려도 거의 줄지 않음)"
                break
            if top > 0 and k < len(gains) and gains[k] < 0.3 * top:
                knee = r["total_staff"]
                reason = "다음 증원부터 1명당 손실 감소가 최대 감소의 30% 미만"
                break
        if knee is None:
            knee = rows[-1]["total_staff"]
            reason = "비교한 범위 끝까지 손실이 계속 줄어듦 (더 큰 인원 비교 필요)"
    return {"rows": rows, "knee": knee, "knee_reason": reason}

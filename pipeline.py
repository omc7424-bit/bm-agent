"""연산 파이프라인: 코드가 먼저 계산하고 결과를 results/ 폴더에 저장한다. 웹(app.py)은 이 결과만 읽어 보여준다.

  stream(데이터 읽기) → metrics(구간 집계) → baseline(정상 기준) → detector(탐지)
  → optimizer(BM 인원·순서) → llm(문장) → results/<sample|full>/

명령줄 실행:
    python pipeline.py --source sample
    python pipeline.py --source full
    python pipeline.py --source full --staff 20 --staff-obs 1 --staff-act 3 --compare 15 20 25
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

import stream
from baseline import BaselineParams, compute_baseline
from detector import Detector, DetectorParams, grade_at, is_open_at
from llm import Interpreter
from metrics import FlowBins, build_bins, bucket_series, edge_series, load_bins, save_bins
from optimizer import Rules, compare, make_jobs, optimize, staff_curve
from stream import TICKS_PER_UNIT

RESULTS_DIR = stream.BASE_DIR / "results"
# 웹 배포(GitHub 웹 업로드는 폴더가 잘 안 올라감)용: results 폴더 대신 results.zip 한 파일만 있으면 처음에 풀어서 쓴다
if not (RESULTS_DIR / "full" / "summary.json").exists() and (stream.BASE_DIR / "results.zip").exists():
    import zipfile
    with zipfile.ZipFile(stream.BASE_DIR / "results.zip") as _z:
        _z.extractall(stream.BASE_DIR)
BINS_CACHE = stream.CACHE_DIR / "flow_bins.npz"
MAX_EVAL_JOBS = 300
REPEAT_THRESHOLD = 3

DEFAULT_SETTINGS = {"total_staff": 20, "staff_obs": 1, "staff_act": 3, "compare_staff": [15, 20, 25],
                    "minutes_per_tick": stream.DEFAULT_MIN_PER_TICK}   # 표시용 환산 비율 (계산에는 안 씀)


def result_dir(kind: str) -> Path:
    return RESULTS_DIR / kind


def results_ready(kind: str) -> bool:
    d = result_dir(kind)
    return all((d / f).exists() for f in ("summary.json", "events.json", "alerts.json", "plans.json", "series.npz"))


def _dump(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=_np_default), encoding="utf-8")


def _np_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def _params_for(kind: str, bins: FlowBins) -> tuple[BaselineParams, DetectorParams]:
    bp, dp = BaselineParams(), DetectorParams()
    if kind == "sample":
        # 샘플은 5,000행뿐이라 기록이 듬성듬성하다. 화면 확인용으로 창을 데이터 길이에 맞춰 줄인다.
        dp.window = max(5, min(dp.window, bins.n_bins // 20))
        dp.eval_every = max(1, min(dp.eval_every, dp.window // 5))
        dp.validate_block = max(10, min(dp.validate_block, bins.n_bins // 10))
    return bp, dp


# ---------------------------------------------------------------------------
def load_or_build_bins(kind: str, log=print) -> FlowBins:
    if kind == "sample":
        return build_bins(stream.load_sample(), log=lambda *_: None)
    src = stream.load_full()
    if BINS_CACHE.exists():
        b = load_bins(BINS_CACHE)
        if b.source_key == src.cache_key:
            log("구간 집계 캐시 사용")
            return b
    log("parquet에서 구간 집계 계산 (처음 한 번)")
    b = build_bins(src, log=log)
    save_bins(b, BINS_CACHE)
    return b


def _event_view(ev: dict, snap: dict, tick: int) -> dict:
    """특정 시각의 이벤트 상태 (1_prompt.md events 형식)."""
    end = ev["end_tick"] if ev["end_tick"] is not None and ev["end_tick"] <= tick else tick
    return {"id": ev["id"], "station": ev["station"], "type": ev["type"], "grade": snap["grade"],
            "start_tick": ev["start_tick"], "start_time": ev["start_time"],
            "duration": round((end - ev["start_tick"]) / TICKS_PER_UNIT, 2),
            "delayed_parts": snap["delayed_parts"], "affected_stations": snap["affected_stations"],
            "repeat_count": snap["repeat_count"], "evidence": snap["evidence"]}


def open_events_at(events: list[dict], tick: int) -> list[dict]:
    out = []
    for ev in events:
        if is_open_at(ev, tick):
            out.append(_event_view(ev, grade_at(ev, tick), tick))
    return out


def snapshot_schedule(events: list[dict], tick: int, rules: Rules, t0_tick: int, time_limit=0.05, workers=1) -> dict:
    """tick 시점에 열려 있는 이벤트들에 대한 BM 투입 인원·순서 (지금 바로 대응한다고 보고 계산)."""
    views = open_events_at(events, tick)
    jobs = make_jobs(views, rules, release_now=tick)
    res = optimize(jobs, rules, time_limit=time_limit, t0_tick=t0_tick, workers=workers)
    res["tick"] = int(tick)
    return res


def plan_at(plans: dict, tick: int) -> dict:
    """tick 이전 가장 최근 결정 시점의 BM 투입 계획 (pipeline이 미리 계산해 둔 것)."""
    keys = [int(k) for k in plans]
    if not keys:
        return {"assignments": [], "status": "계획 없음", "tick": None}
    import bisect
    keys.sort()
    i = bisect.bisect_right(keys, tick) - 1
    if i < 0:
        return {"assignments": [], "status": "계획 없음", "tick": None}
    return plans[str(keys[i])]


def repeat_table(events: list[dict], upto_tick: int | None = None) -> list[dict]:
    cnt = {}
    for ev in events:
        if upto_tick is not None and ev["snapshots"][0]["tick"] > upto_tick:
            continue
        c = cnt.setdefault(ev["station"], {"station": ev["station"], "count": 0, "stop": 0, "bottleneck": 0,
                                           "action": 0})
        c["count"] += 1
        c["stop" if ev["type"] == "정지 의심" else "bottleneck"] += 1
        if ev["peak_grade"] == "조치 권고":
            c["action"] += 1
    rows = sorted(cnt.values(), key=lambda r: (-r["stop"], -r["count"]))
    for r in rows:
        r["review"] = r["stop"] >= REPEAT_THRESHOLD
    return rows


def optimization_stage(events, alerts, t0_tick, end_tick, settings, interp: Interpreter, log=print) -> dict:
    """설정값(인원)에 따라 다시 계산되는 부분: 알림별 투입 인원·순서, 기간 비교, 인원 수별 곡선, 문장."""
    rules = Rules(total_staff=int(settings["total_staff"]), staff_obs=int(settings["staff_obs"]),
                  staff_act=int(settings["staff_act"]), dur_obs=int(settings["dur_obs"]),
                  dur_act=int(settings["dur_act"]))
    by_id = {ev["id"]: ev for ev in events}

    # 1) 알림 시각(결정 시점)마다 투입 인원·순서 + 알림 문장. 같은 시각의 알림은 계획 하나를 같이 쓴다.
    sched_cache = {}
    t_start = time.time()
    for i, al in enumerate(alerts):
        t = al["tick"]
        if t not in sched_cache:
            sched_cache[t] = snapshot_schedule(events, t, rules, t0_tick)
        sch = sched_cache[t]
        a = next((x for x in sch["assignments"] if x["event_id"] == al["event_id"]), None)
        ev = by_id[al["event_id"]]
        view = _event_view(ev, ev["snapshots"][al["snapshot"]], t)
        view["timestamp"] = al["time"]
        view["assignment"] = {"staff": a["staff"], "order": a["order"], "start": a["start"], "end": a["end"]} if a else None
        use_llm = view["grade"] == "조치 권고"
        if use_llm:
            text, source = interp.alert(view, float(settings.get("minutes_per_tick", stream.DEFAULT_MIN_PER_TICK)))
        else:
            from llm import template_alert
            text, source = template_alert(view, float(settings.get("minutes_per_tick", stream.DEFAULT_MIN_PER_TICK))), "template"
        al.update({"assignment": view["assignment"], "text": text, "text_source": source,
                   "open_count": len(sch["assignments"])})
        if i % 500 == 0:
            log(f"알림 {i + 1}/{len(alerts)} 처리 ({round(time.time() - t_start)}초)")

    # 2) 재생 기간 전체: 발생 순 대응 vs 최적화 대응
    jobs_all = make_jobs([{**ev, "delayed_parts": ev["delayed_parts"]} for ev in events], rules, grade_key="peak_grade")
    note = ""
    if len(jobs_all) > MAX_EVAL_JOBS:
        jobs_all = sorted(jobs_all, key=lambda j: j["release"])[-MAX_EVAL_JOBS:]
        note = f"이벤트가 많아 최근 {MAX_EVAL_JOBS}건으로 평가"
    log(f"기간 전체 최적화 ({len(jobs_all)}건)")
    cmp = compare(jobs_all, rules, time_limit=10.0, t0_tick=t0_tick)
    cmp["note"] = note
    curve = staff_curve(jobs_all, rules, settings["compare_staff"], time_limit=5.0)

    # 3) 재생 끝 시각 기준 현재 대응
    final_sched = snapshot_schedule(events, end_tick, rules, t0_tick, time_limit=2.0, workers=4)
    reps = repeat_table(events)
    summary_payload = {
        "timestamp": round((end_tick + t0_tick) / TICKS_PER_UNIT, 2),
        "optimization": {"total_staff": rules.total_staff, "assignments": final_sched["assignments"][:10],
                         "total_loss": cmp["total_loss"], "baseline_loss": cmp["baseline_loss"],
                         "reduction_pct": cmp["reduction_pct"]},
        "repeat_top": [r for r in reps if r["review"]][:5],
    }
    s_text, s_src = interp.summary(summary_payload)
    st_text, st_src = interp.staffing(curve)
    sched_cache[end_tick] = final_sched
    return {"plans": {str(k): v for k, v in sched_cache.items()},
            "rules": asdict(rules), "comparison": cmp, "curve": curve, "final_schedule": final_sched,
            "summary_text": s_text, "summary_source": s_src, "staffing_text": st_text, "staffing_source": st_src,
            "llm_label": interp.label, "llm_calls": interp.calls, "llm_errors": interp.errors[-5:],
            "settings": settings}


# ---------------------------------------------------------------------------
def run(kind: str = "sample", settings: dict | None = None, log=print) -> Path:
    t_start = time.time()
    settings = {**DEFAULT_SETTINGS, **(settings or {})}
    out = result_dir(kind)
    out.mkdir(parents=True, exist_ok=True)

    log(f"[{kind}] 데이터 읽기·구간 집계")
    bins = load_or_build_bins(kind, log=log)
    bp, dp = _params_for(kind, bins)
    log("정상 기준 계산")
    base = compute_baseline(bins, bp)
    settings.setdefault("dur_obs", int(round(base.stop_dur_median)))
    settings.setdefault("dur_act", int(round(base.stop_dur_p75)))

    log("병목·정지 의심 탐지")
    det = Detector(bins, base, dp)
    det.run(base.split_bin, bins.n_bins, log=log)
    det.finalize(bins.n_bins)
    validation = det.validate_apm()
    end_tick = bins.n_bins - 1

    interp = Interpreter()
    opt = optimization_stage(det.events, det.alerts, bins.t0_tick, end_tick, settings, interp, log=log)
    plans = opt.pop("plans")

    # 추세 그래프·흐름 지도용 시계열 + 정상 기준선
    log("흐름 지도용 시계열 계산")
    bucket = max(1, int(np.ceil(bins.n_bins / 20000)))
    ser = bucket_series(bins, bucket)
    front = ser["t_end"] <= bins.time_of(base.split_bin)
    thr_p95 = np.percentile(ser["throughput"][:, front], 95, axis=1) if front.any() else np.zeros(bins.S)
    # 흐름 그래프(baseline이 복원한 앞 → 뒤 연결)와 연결선별 이동 부품 수
    pairs = [(s, n, w) for s in range(bins.S) for n, w in base.successors[s]]
    e_from = np.array([p[0] for p in pairs], np.int16)
    e_to = np.array([p[1] for p in pairs], np.int16)
    e_share = np.array([p[2] for p in pairs], np.float32)
    src = stream.load_sample() if kind == "sample" else stream.load_full()
    e_flow = edge_series(src.trans_batches, bins, e_from, e_to, bucket)
    np.savez_compressed(
        out / "series.npz", stations=np.array(bins.stations), lines=bins.lines, t_end=ser["t_end"],
        throughput=ser["throughput"], gap=ser["gap"], queue=ser["queue"], transit=ser["transit"],
        queue_now=ser["queue_now"], edge_from=e_from, edge_to=e_to, edge_share=e_share, edge_flow=e_flow,
        base_thr_mean=base.rate_mean * bucket, base_thr_p95=thr_p95,
        base_gap_p95=base.gap_p95 / TICKS_PER_UNIT, base_gap_p50=base.gap_p50 / TICKS_PER_UNIT,
        base_q_med=base.q_med, base_q_p95=base.q_p95, edges=bins.edges,
        TR=bins.TR.reshape(bins.S, -1).sum(1), DF=bins.DF.reshape(bins.S, -1).sum(1))

    t = lambda b: round(float(bins.time_of(b)), 2)
    replay_from = base.split_bin
    if kind == "sample":
        data_info = dict(stream.load_sample().info)
    else:
        data_info = json.loads(stream.META_JSON.read_text(encoding="utf-8"))
    data_info.update({"stations": bins.S, "n_bins": bins.n_bins, "t0_tick": bins.t0_tick})
    summary = {
        "kind": kind, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "elapsed_sec": round(time.time() - t_start, 1),
        "data": data_info,
        "time": {"t_min": t(0), "t_max": t(bins.n_bins - 1), "split_time": t(base.split_bin),
                 "replay_from_tick": replay_from, "replay_to_tick": end_tick, "bucket_ticks": bucket},
        "baseline": base.summary(bins),
        "params": {"baseline": asdict(bp), "detector": asdict(dp)},
        "validation": validation,
        "planned_stops": [{"start": t(a), "end": t(b), "length": round((b - a + 1) / TICKS_PER_UNIT, 2),
                           "in_replay": bool(b >= replay_from)} for a, b in base.planned_stops],
        "line_stops": [{"line": L, "start": t(a), "end": t(b), "start_tick": int(a), "end_tick": int(b),
                        "length": round((b - a + 1) / TICKS_PER_UNIT, 2)}
                       for L, a, b in base.line_stops if b >= replay_from],
        "data_checks": {
            "reversed_transitions": bins.reversed_transitions,
            "unreliable_baseline_stations": [bins.stations[i] for i in np.nonzero(~base.gap_reliable)[0]],
            "empty_periods": len(base.planned_stops),
        },
        "counts": {"events": len(det.events), "alerts": len(det.alerts),
                   "stop": sum(e["type"] == "정지 의심" for e in det.events),
                   "bottleneck": sum(e["type"] == "병목" for e in det.events)},
        "repeat": repeat_table(det.events),
        **opt,
    }
    _dump(out / "events.json", det.events)
    _dump(out / "alerts.json", det.alerts)
    _dump(out / "plans.json", plans)
    _dump(out / "summary.json", summary)
    log(f"[{kind}] 완료: 이벤트 {len(det.events)}건, 알림 {len(det.alerts)}건, {summary['elapsed_sec']}초 → {out}")
    return out


def recompute_optimization(kind: str, settings: dict, log=print) -> dict:
    """웹에서 인원 설정을 바꿨을 때: 저장된 탐지 결과로 최적화·문장만 다시 계산해 저장한다."""
    out = result_dir(kind)
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    events = json.loads((out / "events.json").read_text(encoding="utf-8"))
    alerts = json.loads((out / "alerts.json").read_text(encoding="utf-8"))
    st = {**summary["settings"], **settings}
    opt = optimization_stage(events, alerts, summary["data"]["t0_tick"], summary["time"]["replay_to_tick"],
                             st, Interpreter(), log=log)
    plans = opt.pop("plans")
    summary.update(opt)
    summary["recomputed"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _dump(out / "alerts.json", alerts)
    _dump(out / "plans.json", plans)
    _dump(out / "summary.json", summary)
    log(f"[{kind}] 최적화 재계산 완료")
    return summary


def load_results(kind: str) -> dict:
    out = result_dir(kind)
    z = np.load(out / "series.npz", allow_pickle=False)
    return {
        "summary": json.loads((out / "summary.json").read_text(encoding="utf-8")),
        "events": json.loads((out / "events.json").read_text(encoding="utf-8")),
        "alerts": json.loads((out / "alerts.json").read_text(encoding="utf-8")),
        "plans": json.loads((out / "plans.json").read_text(encoding="utf-8")),
        "series": {k: z[k] for k in z.files},
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="탐지·최적화 연산 후 results/에 저장")
    ap.add_argument("--source", choices=["sample", "full"], default="sample")
    ap.add_argument("--staff", type=int, default=DEFAULT_SETTINGS["total_staff"])
    ap.add_argument("--staff-obs", type=int, default=DEFAULT_SETTINGS["staff_obs"])
    ap.add_argument("--staff-act", type=int, default=DEFAULT_SETTINGS["staff_act"])
    ap.add_argument("--compare", type=int, nargs="+", default=DEFAULT_SETTINGS["compare_staff"])
    ap.add_argument("--recompute", action="store_true", help="탐지는 그대로 두고 최적화·문장만 다시 계산")
    a = ap.parse_args(argv)
    st = {"total_staff": a.staff, "staff_obs": a.staff_obs, "staff_act": a.staff_act, "compare_staff": a.compare}
    log = lambda m: print(m, flush=True)
    if a.recompute:
        recompute_optimization(a.source, st, log=log)
    else:
        run(a.source, st, log=log)


if __name__ == "__main__":
    main()

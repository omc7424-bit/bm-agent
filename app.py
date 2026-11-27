"""⑦ 화면 (Streamlit)

이 화면은 계산을 직접 하지 않는다. pipeline.py가 계산해 results/에 저장한 결과
(metrics·detector·optimizer·llm의 출력)를 읽어 모니터링 시점에 맞춰 보여준다.
설정을 바꾸면 '설정 반영' 버튼으로 코드(최적화)를 다시 실행해 결과를 저장한 뒤 반영한다.

구성
- 상단 한 줄 상태 바
- 왼쪽 2/3: 탭 (모니터링 / BM 최적화 / 이상 발생 기록 / 보고서)
- 오른쪽 1/3: Agent 브리핑 패널 (메시지가 시간순으로 쌓임, 패널 안에서만 스크롤)
모니터링 중에는 st.fragment(run_every=1초)로 상태 바·선택된 탭·브리핑 패널을 다시 그린다. 일시정지하면 멈춘다.

실행: streamlit run app.py
"""
from __future__ import annotations

import io
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import pipeline
import stream
from detector import grade_at, is_open_at
from llm import LABEL_TEMPLATE, grade_label, template_alert, template_summary, type_label
from report import build_report, flow_layout, line_row_y, station_grades_at

# 어두운 화면용 색 (Word 보고서는 흰 바탕이라 report.GRADE_COLOR를 그대로 쓴다)
GRADE_COLOR = {"정상": "#4E8A64", "관찰 필요": "#E0A548", "조치 권고": "#E5534B", "판단 보류": "#A79FD6"}
BG = "#0F1419"          # 바탕
ACCENT = "#5B8DEF"      # 강조(선택·정비 중·추세선)
TXT, TXT2, TXT3 = "#E5E7EB", "#9CA3AF", "#6B7280"   # 글자: 기본 / 보조 / 옅은
GRID, EDGE = "#262E38", "#3A4350"
from stream import (TICKS_PER_UNIT, clock_full, clock_text, clock_ticks, default_clock_base, duration_text, station_key,
                    to_datetime)

st.set_page_config(page_title="물류흐름 모니터링·BM 개선 Agent", layout="wide")
# 1초 갱신 때 Streamlit이 '갱신 중' 요소를 반투명(33%)으로 흐리게 했다 되돌리는 기본 효과를 끈다 (화면 깜빡임 방지)
st.markdown("""<style>
[data-stale='true']{opacity:1 !important;transition:none !important;}
[data-testid='stElementContainer'],[data-testid='stVerticalBlock']{transition:none !important;}
.block-container{padding-top:1.2rem;padding-bottom:1rem;}
h1,h2,h3{letter-spacing:-0.01em}
/* 상단 지표 카드 */
.kpis{display:flex;gap:10px;flex-wrap:wrap;margin:2px 0 6px 0}
.kpi{background:#161C24;border:1px solid #262E38;border-radius:10px;padding:8px 14px;min-width:120px;flex:1}
.kpi .l{font-size:12px;color:#9CA3AF;display:flex;align-items:center;gap:4px}
.kpi .v{font-size:22px;font-weight:700;color:#F3F4F6;line-height:1.25;white-space:nowrap}
.kpi .s{font-size:11px;color:#6B7280}
.kpi.main{flex:1.6}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%}
.bar{height:3px;background:#262E38;border-radius:2px;margin:0 0 4px 0}
.bar>div{height:3px;background:#5B8DEF;border-radius:2px}
.meta{font-size:11.5px;color:#6B7280}
.sec{font-size:14px;font-weight:700;color:#E5E7EB;margin:4px 0 2px 0}
/* 브리핑 메시지 */
.msg{display:flex;justify-content:space-between;align-items:baseline;gap:6px}
.msg .h{font-size:13.5px;font-weight:600;color:#E5E7EB}
.msg .t{font-size:11px;color:#6B7280;white-space:nowrap}
.msg-sub{font-size:12.5px;color:#9CA3AF;margin-top:2px}
.badge{display:inline-flex;align-items:center;font-size:12px;font-weight:600;color:#D1D5DB;margin-right:6px}
.badge::before{content:'';display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--c);margin-right:5px}
.info{cursor:help;color:#6B7280;font-size:12px}
</style>""", unsafe_allow_html=True)
ss = st.session_state
LOG_DIR = stream.CACHE_DIR
SOURCES = {"실제 전체 데이터(Bosch)": "full", "테스트용 샘플(5,000행)": "sample"}
REFRESH_SEC = 1.0          # 모니터링 중 화면 갱신 주기 (초)
SPEED_OPTIONS = {"1분": 1, "10분": 10, "1시간": 60, "6시간": 360, "1일": 1440, "3일": 4320}  # 화면 1초당 진행(환산 분)
MAP_WINDOW = 10            # 흐름 지도 '최근 구간' = 시계열 구간 10개
STOP_COLOR = "#E5534B"
TABS = ["모니터링", "BM 최적화", "이상 발생 기록", "보고서"]
GRADE_ICON = {"조치 권고": "🔴", "관찰 필요": "🟡", "판단 보류": "🟣", "정상": "⚪"}
PANEL_MSGS = 15            # 브리핑 패널에 그리는 '확인할 알림' 최대 수 (급한 순)

HELP = {
    "now": "모니터링 중인 현재 시점입니다.",
    "progress": "모니터링 구간(기준 학습 구간 이후) 중 지금까지 진행한 비율입니다.",
    "obs": "관찰 필요: 정상 범위를 벗어났지만 아직 짧거나 약한 이상입니다. 1건당 출동 인원 설정값만큼 투입합니다.",
    "act": "조치 권고: 오래 이어지거나 영향이 큰 이상입니다. 1건당 출동 인원 설정값만큼 투입합니다.",
    "hold": "확인 필요: 같은 라인 기록이 부분적으로만 있어 정지인지 판단할 수 없는 구간입니다. 정비 인원 투입 전에 기록을 확인합니다.",
    "map": "노드 = 설비(스테이션), 선 = 부품 이동. 노드 색은 현재 등급, 크기는 지금 앞에서 기다리는 부품 수, 선 두께는 최근 이동 부품 수입니다.",
    "trend": "선택한 설비의 처리량(구간당 도착 부품 수), 통과 간격(마지막 도착 이후 경과), 대기 부품 수입니다. 점선은 기준 학습 구간의 정상 범위입니다.",
    "crew": "담당자가 승인한 정비만 반영합니다. 승인하면 그 인원이 출동해 조치 시간 동안 '정비 중'이 되고, 끝나면 가용 인원으로 돌아옵니다. "
            "가용 인원보다 많은 인원이 필요한 정비는 승인할 수 없습니다.",
    "now_plan":"지금 진행 중인 이상에 대해 최적화(CP-SAT)가 정한 정비 인원 투입 순서와 인원입니다. 담당자 승인 후 발송합니다.",
    "records": "모니터링 구간에서 감지한 설비 정지 의심·병목 이상의 기록입니다. 최신 기록이 위에 옵니다.",
    "repeat": "설비 정지 의심이 3회 이상 반복된 설비는 BM 개선(예방보전 전환, 점검 주기 재검토) 대상으로 표시합니다.",
}


def q(text: str) -> str:
    """마우스를 올리면 설명이 나오는 작은 ⓘ 표시 (HTML)."""
    t = text.replace("'", "&#39;")
    return f"<span class='info' title='{t}'>ⓘ</span>"


def sec(title: str, help_key: str | None = None, extra: str = ""):
    """섹션 제목 한 줄 (설명은 ⓘ 안에)."""
    st.markdown(f"<div class='sec'>{title} {q(HELP[help_key]) if help_key else ''}"
                f"<span class='meta' style='font-weight:400;margin-left:6px'>{extra}</span></div>",
                unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 결과 읽기 (계산은 pipeline이 이미 끝냄)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False, max_entries=2)
def _load(kind: str, mtime: float) -> dict:
    """결과를 한 번만 읽어 공유한다 (매번 복사하지 않음). 읽기 전용으로만 쓴다."""
    res = pipeline.load_results(kind)
    ser = res["series"]
    res["scale"] = {
        "queue": float(max(np.nanpercentile(ser["queue_now"], 99), 1.0)),
        "edge": float(max(np.percentile(ser["edge_flow"], 99) * MAP_WINDOW, 1.0)) if ser["edge_flow"].size else 1.0,
    }
    res["by_id"] = {e["id"]: e for e in res["events"]}
    # 브리핑 메시지 목록: 이상 발생·등급 변경(알림) + 최적 배치 1순위가 바뀐 시점
    feed = [{"kind": "alert", "tick": a["tick"], "idx": i} for i, a in enumerate(res["alerts"])]
    prev = None
    for k in sorted(res["plans"], key=int):
        p = res["plans"][k]
        asg = p.get("assignments") or []
        first = asg[0]["station"] if asg else None
        if asg and first != prev:
            feed.append({"kind": "plan", "tick": int(k), "key": k})
        prev = first
    feed.sort(key=lambda f: (f["tick"], 0 if f["kind"] == "alert" else 1))
    res["feed"] = feed
    res["feed_ticks"] = np.array([f["tick"] for f in feed])
    return res


def results_mtime(kind: str) -> float:
    return max((pipeline.result_dir(kind) / f).stat().st_mtime for f in ("summary.json", "alerts.json", "plans.json"))


def load(kind: str) -> dict | None:
    if not pipeline.results_ready(kind):
        return None
    return _load(kind, results_mtime(kind))


def current_plan(res: dict, tick: int) -> dict:
    """tick 직전 결정 시점의 BM 투입 계획 (pipeline이 미리 계산). 이미 끝난 이벤트는 뺀다."""
    plan = pipeline.plan_at(res["plans"], tick)
    asg = [a for a in plan["assignments"] if is_open_at(res["by_id"][a["event_id"]], tick)]
    return {**plan, "assignments": asg}


def empty_period_at(sm: dict, tick: int) -> tuple[int, int, dict] | None:
    """tick이 공장 전체 데이터가 없는 구간(계획 정지) 안이면 (시작 tick, 끝 다음 tick, 구간)을 돌려준다."""
    t0_ = sm["data"]["t0_tick"]
    for p in sm["planned_stops"]:
        s_ = int(round(p["start"] * TICKS_PER_UNIT)) - t0_
        e_ = int(round(p["end"] * TICKS_PER_UNIT)) - t0_
        if s_ <= tick <= e_:
            return s_, e_ + 1, p
    return None


def first_active_tick(sm: dict, tick: int) -> int:
    """데이터가 없는 구간이면 공장이 다시 도는 시점으로 옮긴다."""
    hit = empty_period_at(sm, tick)
    return min(hit[1], sm["time"]["replay_to_tick"]) if hit else tick


def _bg(cmd: list[str], log_name: str):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    f = open(LOG_DIR / log_name, "w", encoding="utf-8")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return subprocess.Popen([sys.executable, "-u"] + cmd, cwd=str(stream.BASE_DIR), stdout=f,
                            stderr=subprocess.STDOUT, creationflags=flags)


def _tail(name: str, n: int = 6) -> str:
    p = LOG_DIR / name
    if not p.exists():
        return ""
    try:
        return "\n".join(p.read_text(encoding="utf-8", errors="ignore").splitlines()[-n:])
    except OSError:
        return ""


def _proc_running(key: str) -> bool:
    p = ss.get(key)
    return p is not None and p.poll() is None


# ---------------------------------------------------------------------------
# 사이드바
# ---------------------------------------------------------------------------
@st.dialog("사용 설명서", width="large")
def show_guide():
    """사용설명서.md를 그대로 보여 준다 (설명서는 문서 하나만 고치면 화면에도 반영된다)."""
    here = Path(__file__).resolve().parent
    path = next((p for p in (here / "사용설명서.md", here.parent / "사용설명서.md") if p.exists()), None)
    if path is None:
        st.info("사용설명서.md를 찾지 못했습니다. 제출 폴더 맨 위에 있습니다.")
        return
    st.markdown(path.read_text(encoding="utf-8").split("\n", 1)[1])


with st.sidebar:
    if st.button("📖 사용 설명서", use_container_width=True):
        show_guide()
    st.header("설정")
    default_src = 0 if (stream.full_data_ready() or pipeline.results_ready("full")) else 1
    src_label = st.radio("분석 데이터", list(SOURCES), index=default_src,
                         help="실제 전체 데이터는 data 폴더의 Bosch zip을 변환한 결과를, 테스트용 샘플은 5,000행 샘플 CSV를 씁니다.")
    kind = SOURCES[src_label]
    if ss.get("kind") != kind:
        ss.kind = kind
        ss.approvals = {}
        ss.dispatch = {}
        ss.tick = None
        ss.tick_f = None
        ss.playing = False

    if kind == "full":
        if pipeline.results_ready("full") and not _proc_running("pipe_proc"):
            pass                                  # 계산 결과가 있으면 원본 데이터 없이도 바로 연다 (제출본은 결과만 포함)
        elif not stream.full_data_ready():
            zp = stream.zips_present()
            st.caption("zip 파일: " + ", ".join(f"{k} {'있음' if v else '없음'}" for k, v in zp.items()))
            prog = stream.read_progress()
            if _proc_running("conv_proc"):
                st.info(f"데이터 변환 중: {prog.get('stage') if prog else ''} {prog.get('percent') if prog else ''}%")
                st.button("진행 상태 새로고침")
            elif all(zp.values()):
                if prog and not prog.get("done"):
                    st.warning("이전 데이터 변환이 끝나지 않았습니다. 다시 시작하면 처음부터 변환합니다(약 17분).")
                st.button("데이터 변환 시작 (백그라운드)", type="primary",
                          on_click=lambda: ss.__setitem__("conv_proc", _bg(["stream.py"], "convert_log.txt")))
        elif _proc_running("pipe_proc"):
            st.info("계산 중 (이상 탐지·최적 배치)")
            st.code(_tail("pipeline_full_log.txt"), language=None)
            st.button("진행 상태 새로고침")
        elif not pipeline.results_ready("full"):
            st.button("계산 실행 (약 4분)", type="primary",
                      on_click=lambda: ss.__setitem__("pipe_proc", _bg(
                          ["pipeline.py", "--source", "full"], "pipeline_full_log.txt")))
    elif not pipeline.results_ready("sample"):
        with st.spinner("샘플 데이터 계산 중..."):
            pipeline.run("sample", log=lambda *_: None)

    res = load(kind) if not _proc_running("pipe_proc") else None
    speed_min = 1
    if res is not None:
        sm = res["summary"]
        r0 = sm["rules"]
        st.divider()

        def _toggle_play(r_from=sm["time"]["replay_from_tick"], r_to=sm["time"]["replay_to_tick"]):
            ss.playing = not ss.get("playing", False)
            if ss.playing and ss.get("tick") is not None and ss.tick >= r_to:
                ss.tick = first_active_tick(sm, r_from)
            ss.tick_f = float(ss.tick) if ss.get("tick") is not None else None
            ss.last_wall = None

        def _jump(t, reset=False):
            if reset:
                t = first_active_tick(sm, t)    # 처음부터: 데이터 없는 구간이면 공장이 다시 도는 시점부터
            ss.tick = t
            ss.tick_f = float(t)
            ss.playing = False
            if reset:                       # 처음부터: 승인·출동 기록도 비운다
                ss.approvals = {}
                ss.dispatch = {}

        st.button("⏸ 일시정지" if ss.get("playing") else "▶ 모니터링 시작", use_container_width=True, on_click=_toggle_play,
                  type="primary")
        c1, c2 = st.columns(2)
        c1.button("⏮ 처음부터", use_container_width=True, on_click=_jump, args=(sm["time"]["replay_from_tick"], True),
                  help="처음 시점으로 돌아가고 승인·정비 출동 기록을 비웁니다.")
        c2.button("⏭ 마지막 시점", use_container_width=True, on_click=_jump, args=(sm["time"]["replay_to_tick"],))
        speed_label = st.select_slider("모니터링 속도 (화면 1초당 진행 시간)", list(SPEED_OPTIONS), value="1분",
                                       help="화면에서 1초가 지날 때 환산 시각이 얼마나 진행할지 정합니다.")
        speed_min = SPEED_OPTIONS[speed_label]

        st.divider()
        total_staff = st.number_input("투입 가능 정비 인원 (명)", 1, 200, int(r0["total_staff"]),
                                      help="동시에 투입할 수 있는 BM 정비 인원입니다. 데이터가 아니라 사용자 설정값입니다.")
        with st.expander("상세 설정", expanded=False):
            staff_obs = st.number_input("1건당 출동 인원: 관찰 필요 (명)", 1, 50, int(r0["staff_obs"]))
            staff_act = st.number_input("1건당 출동 인원: 조치 권고 (명)", 1, 50, int(r0["staff_act"]))
            cmp_txt = st.text_input("비교할 정비 인원 (명, 쉼표로 구분)",
                                    ", ".join(str(x) for x in sm["settings"]["compare_staff"]))
            st.number_input("시간 환산 기준 (0.01 단위 = ○분)", min_value=0.1, max_value=1440.0,
                            value=float(sm["settings"].get("minutes_per_tick", stream.DEFAULT_MIN_PER_TICK)),
                            step=1.0, key="mpt", help="원본 데이터의 시간값은 익명화되어 있어 화면 시각으로 바꿀 비율입니다. "
                                                      "표시만 바뀌고 이상 탐지·최적 배치 계산은 원본 시간값으로 합니다.")
            _mpt = float(ss.get("mpt", stream.DEFAULT_MIN_PER_TICK))
            st.date_input("첫 기록 날짜", value=default_clock_base(_mpt).date(), key="base_date",
                          help="원본 시간값 0을 어느 날짜로 볼지 정합니다. 기본값은 데이터의 주간 휴무가 "
                               "토·일요일이 되도록 맞춘 날짜(2024-01-02 화요일 20:00)입니다. 표시만 바뀝니다.")
        try:
            cmp_vals = sorted({int(x) for x in cmp_txt.replace(" ", "").split(",") if x})
        except ValueError:
            cmp_vals = sm["settings"]["compare_staff"]
            st.warning("비교할 정비 인원은 숫자와 쉼표로 입력하세요.")
        new_settings = {"total_staff": int(total_staff), "staff_obs": int(staff_obs), "staff_act": int(staff_act),
                        "compare_staff": cmp_vals}
        changed = any(new_settings[k] != sm["settings"].get(k) for k in new_settings)
        if st.button("설정 반영 (최적 배치 다시 계산)", disabled=not changed, use_container_width=True):
            if kind == "sample":
                with st.spinner("최적 배치 다시 계산 중..."):
                    pipeline.recompute_optimization(kind, new_settings, log=lambda *_: None)
                st.cache_resource.clear()
                st.rerun()
            else:
                ss.playing = False
                ss.pipe_proc = _bg(["pipeline.py", "--source", "full", "--recompute",
                                    "--staff", str(total_staff), "--staff-obs", str(staff_obs),
                                    "--staff-act", str(staff_act), "--compare"] + [str(v) for v in cmp_vals],
                                   "pipeline_full_log.txt")
                st.rerun()
        if changed:
            st.caption("설정이 바뀌었습니다. '설정 반영'을 눌러야 화면에 반영됩니다(실제 전체 데이터는 약 4분).")
    st.caption("※ 정비 인원과 1건당 출동 인원은 데이터가 아닌 사용자 설정값입니다.")

# ---------------------------------------------------------------------------
# 본문 준비
# ---------------------------------------------------------------------------
st.markdown("<div style='font-size:19px;font-weight:700;color:#F3F4F6;margin-bottom:2px'>물류흐름 모니터링 · BM 개선 Agent"
            "</div>",
            unsafe_allow_html=True)
if res is None:
    if kind == "full":
        st.info("실제 전체 데이터의 계산 결과가 아직 없습니다. 왼쪽에서 데이터 변환 → 계산 실행 순서로 진행하세요. "
                "그동안 '테스트용 샘플'로 화면을 확인할 수 있습니다.")
    st.stop()

sm, events, alerts, ser = res["summary"], res["events"], res["alerts"], res["series"]
by_id = res["by_id"]
t0 = sm["data"]["t0_tick"]
r_from, r_to = sm["time"]["replay_from_tick"], sm["time"]["replay_to_tick"]
stations = [str(s) for s in ser["stations"]]
pos = flow_layout(stations)
if ss.get("tick") is None:
    ss.tick = first_active_tick(sm, r_from)       # 시작 시점이 데이터 없는 구간이면 공장이 다시 도는 시점부터
    ss.tick_f = float(ss.tick)
if ss.get("tick_f") is None:
    ss.tick_f = float(ss.tick)
ss.setdefault("approvals", {})
ss.setdefault("dispatch", {})     # 승인으로 출동한 정비: event_id → {station, staff, start, end} (원본 시간값)


def crew_status(now: float) -> dict:
    """승인한 정비 기준 정비 인원 현황. 승인 시점부터 조치 시간 동안 정비 중, 끝나면 가용으로 복귀."""
    total = int(sm["rules"]["total_staff"])
    rows = []
    for eid, d in ss.dispatch.items():
        state = "정비 중" if d["start"] <= now < d["end"] else ("정비 완료" if now >= d["end"] else "출동 예정")
        rows.append({**d, "event_id": eid, "state": state})
    busy = sum(r["staff"] for r in rows if r["state"] == "정비 중")
    return {"total": total, "busy": busy, "avail": max(total - busy, 0), "rows": rows}


def on_site(now: float) -> dict:
    """지금 정비 중인 설비별 인원: 설비 → {staff, end}. 같은 설비에 여러 건이면 인원은 합치고 복귀는 가장 늦은 시각."""
    out = {}
    for r in crew_status(now)["rows"]:
        if r["state"] == "정비 중":
            o = out.setdefault(r["station"], {"staff": 0, "end": r["end"]})
            o["staff"] += int(r["staff"])
            o["end"] = max(o["end"], r["end"])
    return out


MPT = float(ss.get("mpt", stream.DEFAULT_MIN_PER_TICK))   # 표시용 환산 비율 (0.01 단위 = MPT분)
# 원본 시간값 0의 일시 (표시 전용). 날짜를 바꾸면 기본 시각(하루 중 시:분)은 유지하고 날짜만 옮긴다.
_base = default_clock_base(MPT)
if ss.get("base_date") and ss.base_date != _base.date():
    from datetime import datetime as _dt
    _base = _dt.combine(ss.base_date, _base.time())
stream.CLOCK_BASE = _base


def ck(t: float) -> str:
    """원본 시간값 → 시각 '05-18(일) 05:24' (표시 전용)."""
    return clock_text(t, MPT)


def dur(d: float) -> str:
    return duration_text(d, MPT)


def clock_axis(fig, lo: float, hi: float, **kw):
    """가로축 위치는 원본 시간값 그대로, 눈금 글자만 환산 시각으로."""
    vals, texts = clock_ticks(lo, hi, MPT)
    fig.update_xaxes(tickmode="array", tickvals=vals, ticktext=texts, **kw)


def t_of(tick: int) -> float:
    return round((tick + t0) / TICKS_PER_UNIT, 2)


# ---------------------------------------------------------------------------
# 그림 (값은 모두 metrics·detector·optimizer 계산 결과에서 가져옴)
# ---------------------------------------------------------------------------
def flow_map(tick: int, now: float, blink: bool) -> go.Figure:
    t_end = ser["t_end"]
    bi = int(np.clip(np.searchsorted(t_end, now - 1e-9), 0, len(t_end) - 1))
    lo = max(bi - MAP_WINDOW + 1, 0)
    win_units = round(MAP_WINDOW * sm["time"]["bucket_ticks"] / TICKS_PER_UNIT, 2)
    thr = ser["throughput"][:, lo:bi + 1].sum(1)
    gap = ser["gap"][:, bi]
    qn = ser["queue_now"][:, bi]
    flow = ser["edge_flow"][:, lo:bi + 1].sum(1) if ser["edge_flow"].size else np.array([])
    sg = station_grades_at(events, tick)
    qs, es = res["scale"]["queue"], res["scale"]["edge"]
    crew_at = on_site(now)

    fig = go.Figure()
    # 같은 열(병렬 설비)에서 같은 곳으로 가는 짧은 표시는 글자를 맨 아래 하나만 단다 (글자 겹침 방지)
    stub_label = {}
    for f, t in zip(ser["edge_from"], ser["edge_to"]):
        a_, b_ = stations[int(f)], stations[int(t)]
        if station_key(a_)[0] != station_key(b_)[0] and abs(pos[b_][0] - pos[a_][0]) > 4:
            key = (station_key(a_)[0], b_, pos[a_][0])
            if key not in stub_label or pos[a_][1] < pos[stub_label[key]][1]:
                stub_label[key] = a_
    for k, (f, t) in enumerate(zip(ser["edge_from"], ser["edge_to"])):
        a, b = stations[int(f)], stations[int(t)]
        (x0, y0), (x1, y1) = pos[a], pos[b]
        n = float(flow[k]) if len(flow) else 0.0
        w = 0.5 + 5 * np.sqrt(min(n / es, 1.0))
        hover_e = (f"{a} → {b}<br>최근 {dur(win_units)} 이동 부품 {int(n)}개<br>"
                   f"평소 이동 비중 {round(float(ser['edge_share'][k]) * 100)}%")
        if station_key(a)[0] != station_key(b)[0] and abs(x1 - x0) > 4:
            # 라인을 건너 멀리 가는 흐름은 지도를 가로지르지 않게 짧은 화살표 + '→ 도착 설비' 표시로 그린다
            dx, dy = x1 - x0, (y1 - y0) * 3.0          # 세로는 화면에서 더 늘어나 보이므로 보정
            norm = max(np.hypot(dx, dy), 1e-9)
            sx, sy = x0 + 0.9 * dx / norm, y0 + 0.9 * (dy / 3.0) / norm
            fig.add_trace(go.Scatter(
                x=[x0, sx], y=[y0, sy], mode="lines+markers+text",
                line=dict(width=w, color="#4A5563" if n > 0 else "#252C36"),
                marker=dict(symbol="arrow", angleref="previous", size=[0, 5 + w], color="#5B6675"),
                text=["", f"→ {b.replace('_', ' ')}" if stub_label.get((station_key(a)[0], b, x0)) == a else ""],
                textposition="bottom center",
                textfont=dict(size=8, color=TXT3),
                hoverinfo="text", hovertext=hover_e, showlegend=False))
            continue
        xm, ym = x0 + (x1 - x0) * 0.7, y0 + (y1 - y0) * 0.7
        fig.add_trace(go.Scatter(
            x=[x0, xm, x1], y=[y0, ym, y1], mode="lines+markers",
            line=dict(width=w, color="#4A5563" if n > 0 else "#252C36"),
            marker=dict(symbol="arrow", angleref="previous", size=[0, 5 + w, 0], color="#5B6675"),
            hoverinfo="text", hovertext=f"{a} → {b}<br>최근 {dur(win_units)} 이동 부품 {int(n)}개<br>"
                                        f"평소 이동 비중 {round(float(ser['edge_share'][k]) * 100)}%",
            showlegend=False))

    xs, ys, colors, sizes, lw, lc, hover = [], [], [], [], [], [], []
    for i, s in enumerate(stations):
        info = sg.get(s, {"grade": "정상", "types": set()})
        is_bn, is_stop = "병목" in info["types"], "정지 의심" in info["types"]
        xs.append(pos[s][0])
        ys.append(pos[s][1])
        # 정상 설비는 작고 옅게, 이상 설비만 크게 (크기는 대기 부품 수)
        base_sz = 8 + 22 * np.sqrt(min(max(float(qn[i]), 0) / qs, 1.0))
        hold = info["grade"] == "판단 보류"
        if hold:
            # 확인 필요(기록 불확실)는 눈에 덜 띄게: 속이 빈 옅은 보라 테두리만, 크기도 키우지 않는다
            colors.append(BG)
            sizes.append(max(base_sz, 11))
            lw.append(2)
            lc.append(GRADE_COLOR["판단 보류"])
        else:
            colors.append(GRADE_COLOR[info["grade"]])
            sizes.append(base_sz if info["grade"] == "정상" else max(base_sz, 16) + 6)
            if is_stop:
                lw.append(3)
                lc.append(STOP_COLOR)      # 깜빡이지 않는 빨간 테두리
            elif is_bn:
                lw.append(3)
                lc.append(TXT)
            else:
                lw.append(1)
                lc.append(BG)
        p95 = float(ser["base_gap_p95"][i])
        g = "-" if np.isnan(gap[i]) else f"{dur(float(gap[i]))}"
        types = ", ".join(type_label(x) for x in sorted(info["types"]))
        hover.append(
            f"<b>{s}</b><br>현재 등급: {grade_label(info['grade'])}" + (f" ({types})" if types else "")
            + f"<br>처리량: 최근 {dur(win_units)} {int(thr[i])}개"
            + f"<br>도착 간격: 마지막 도착 후 {g} (정상 범위 상한 {dur(p95)})"
            + f"<br>대기 부품 수: {int(qn[i])}개"
            + (f"<br>정비 중: {crew_at[s]['staff']}명 (복귀 {ck(crew_at[s]['end'])})" if s in crew_at else ""))
    # 설비 번호는 이상 설비만 진하게
    labels = [s.split("_")[1] for s in stations]
    lab_col = [TXT if sg.get(s, {}).get("grade", "정상") in ("조치 권고", "관찰 필요") else "#4B5563"
               for s in stations]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers+text", text=labels,
                             textposition="top center", textfont=dict(size=9, color=lab_col), customdata=stations,
                             hovertext=hover, hoverinfo="text", showlegend=False,
                             marker=dict(size=sizes, color=colors, line=dict(width=lw, color=lc), opacity=1)))
    # 정비 중인 설비: 설비 아래에 '🔧 인원' 표시
    if crew_at:
        cs_ = [s for s in stations if s in crew_at]
        # 설비 원과 같은 크기의 투명 표식을 깔아 글자가 원 바로 아래에 오게 한다 (원에 가려지지 않게)
        fig.add_trace(go.Scatter(x=[pos[s][0] for s in cs_], y=[pos[s][1] for s in cs_], mode="markers+text",
                                 marker=dict(size=[sizes[stations.index(s)] + 4 for s in cs_], opacity=0),
                                 text=[f"<b>🔧{crew_at[s]['staff']}명</b>" for s in cs_], textposition="bottom center",
                                 textfont=dict(size=11, color=ACCENT), hoverinfo="skip", showlegend=False))
    for name, col, line in (("조치 권고", GRADE_COLOR["조치 권고"], None), ("관찰 필요", GRADE_COLOR["관찰 필요"], None),
                            ("확인 필요", BG, GRADE_COLOR["판단 보류"]), ("정상", GRADE_COLOR["정상"], None),
                            ("설비 정지 의심", BG, STOP_COLOR), ("병목", BG, TXT)):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=name,
                                 marker=dict(size=9, color=col, line=dict(width=2 if line else 0, color=line or col))))
    lines = sorted({station_key(s)[0] for s in stations})
    stopped_lines = {l["line"] for l in sm["line_stops"] if l["start_tick"] <= tick <= l["end_tick"]}
    for L in lines:
        txt = f"<b>L{L}</b>" + ("<br><span style='color:#A79FD6;font-size:10px'>라인 정지</span>" if L in stopped_lines else "")
        fig.add_annotation(x=-1.0, y=line_row_y(L), text=txt, showarrow=False, xanchor="right",
                           font=dict(size=12, color=TXT2))
    # 데이터가 없는 구간(계획 정지)은 모니터링 중 자동으로 건너뛰므로 지도에 따로 표시하지 않는다
    xmax = max(p[0] for p in pos.values())
    fig.update_layout(
        uirevision="flowmap",            # 갱신해도 확대·선택 상태 유지
        height=420, margin=dict(l=5, r=5, t=6, b=5), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        hovermode="closest", font=dict(color=TXT),
        legend=dict(orientation="h", y=-0.01, yanchor="top", x=0, font=dict(size=10, color=TXT2)),
        xaxis=dict(visible=False, range=[-5.0, xmax + 0.8]),
        yaxis=dict(visible=False, range=[line_row_y(max(lines)) - 0.8, line_row_y(min(lines)) + 0.8]))
    return fig



# ---------------------------------------------------------------------------
# 탭 내용
# ---------------------------------------------------------------------------
def tab_monitor(tick: int, now: float, blink: bool):
    sec("실시간 물류 흐름 지도", "map")
    sel = st.plotly_chart(flow_map(tick, now, blink), key="flowmap", on_select="rerun", selection_mode="points")
    # 지도 바로 아래 한 줄: 어느 설비에 몇 명이 투입 중인지 (전체·가용 수는 상단 카드에 있음)
    crew_at = on_site(now)
    if crew_at:
        items = " · ".join(f"<b>{s}</b>에 {v['staff']}명 투입 중 (복귀 {ck(v['end'])})"
                           for s, v in sorted(crew_at.items(), key=lambda kv: kv[1]["end"]))
        st.markdown(f"<div style='margin:-6px 0 10px;font-size:12.5px;color:#9CA3AF'>🔧 {items}</div>",
                    unsafe_allow_html=True)
    try:
        pts = sel["selection"]["points"] if sel else []
    except (KeyError, TypeError):
        pts = []
    if pts:
        cd = pts[0].get("customdata")
        clicked = cd[0] if isinstance(cd, list) else cd
        if clicked in stations and clicked != ss.get("last_click"):
            ss.last_click = clicked
            ss.station_select = clicked
            ss.station_auto = False                 # 직접 고른 설비는 계속 보여준다

    # 자동 선택: 지금 가장 심한 이상 설비 (조치 권고 → 관찰 필요 → 확인 필요, 같으면 지연 부품이 많은 순).
    # 이상이 없으면 지금 부품이 가장 많이 흐르는 설비.
    sg_now = {}
    for e in events:
        if is_open_at(e, tick):
            snap = grade_at(e, tick)
            r = {"조치 권고": 0, "관찰 필요": 1, "판단 보류": 2}.get(snap["grade"], 3)
            key = (r, -snap["delayed_parts"])
            if e["station"] not in sg_now or key < sg_now[e["station"]][0]:
                sg_now[e["station"]] = (key, snap, e)
    t_end = ser["t_end"]
    bi = int(np.clip(np.searchsorted(t_end, now - 1e-9), 0, len(t_end) - 1))
    if ss.get("station_auto", True) or ss.get("station_select") not in stations:
        if sg_now:
            ss.station_select = min(sg_now.items(), key=lambda kv: kv[1][0])[0]
        else:
            recent = ser["throughput"][:, max(bi - MAP_WINDOW + 1, 0):bi + 1].sum(1)
            ss.station_select = stations[int(np.argmax(recent))]

    def _manual():
        ss.station_auto = False

    c1, c2, c3 = st.columns([1.2, 1.6, 1])
    with c1:
        sec("설비별 흐름 추이", "trend")
    s_sel = c2.selectbox("설비", stations, key="station_select", label_visibility="collapsed", on_change=_manual,
                         help="지도에서 설비를 클릭해도 바뀝니다.")
    if c3.button("이상 설비 자동", disabled=ss.get("station_auto", True), use_container_width=True,
                 help="지금 가장 심한 이상 설비를 자동으로 따라가도록 되돌립니다."):
        ss.station_auto = True
        st.rerun(scope="fragment")
    i = stations.index(s_sel)
    cur = sg_now.get(s_sel)
    if cur:
        snap, ev = cur[1], cur[2]
        st.markdown(f"<div class='msg-sub'><span class='badge' style='--c:{GRADE_COLOR[snap['grade']]}'>"
                    f"{grade_label(snap['grade'])}</span>{type_label(ev['type'])} · {ck(ev['start_time'])}부터 · "
                    f"지연 부품 {snap['delayed_parts']}개"
                    + (" · 자동 선택" if ss.get("station_auto", True) else "") + "</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='msg-sub'><span class='badge' style='--c:{GRADE_COLOR['정상']}'>정상</span>"
                    + ("지금 이상 설비가 없어 부품이 가장 많이 흐르는 설비를 보여줍니다." if ss.get("station_auto", True)
                       else "지금 진행 중인 이상이 없습니다.") + "</div>", unsafe_allow_html=True)

    # 보이는 구간: 최근 3일 (기준 학습 구간의 기록도 포함해 시작 직후에도 평소 흐름과 비교된다)
    days3 = 3 * 1440 / (TICKS_PER_UNIT * MPT)
    lo_t = sm["time"]["t_min"] if kind == "sample" else now - days3
    vis = (t_end >= lo_t) & (t_end <= now)
    x = t_end[vis]

    # 일감 없는 설비: 부품도 안 들어오고 기다리는 부품도 없으면 빈 그래프 대신 한 줄로 알린다
    thr_v, q_v = ser["throughput"][i][vis], ser["queue"][i][vis]
    if len(x) and float(np.nansum(thr_v)) == 0 and float(np.nanmax(q_v)) <= 0 \
            and (not cur or cur[1]["grade"] == "판단 보류"):
        g_now = ser["gap"][i][bi]
        since = f"{dur(float(g_now))} 동안 " if not np.isnan(g_now) else "최근 "
        if cur:      # 확인 필요: 멈춘 것인지 기록이 빠진 것인지 알 수 없음
            st.info(f"{s_sel}에는 {since}부품 기록이 없고, 기다리는 부품도 없습니다. 같은 라인의 기록도 평소보다 적어 "
                    "설비가 멈춘 것인지 기록이 빠진 것인지 현장 확인이 필요합니다. 기록이 없어 그래프는 생략합니다.")
        else:
            st.info(f"{s_sel}에는 {since}부품이 들어오지 않았고, 기다리는 부품도 없습니다. "
                    "고장이 아니라 일감이 없는 상태라 그래프를 생략합니다. 지도에서 색이 있는 설비를 눌러 보세요.")
        return

    # 통과 간격 단위: 값 크기에 맞춰 분 / 시간 / 일
    gap_min = ser["gap"][i][vis] * TICKS_PER_UNIT * MPT
    gmax = float(np.nanmax(gap_min)) if len(gap_min) and not np.all(np.isnan(gap_min)) else 0.0
    unit, div = ("일", 1440.0) if gmax >= 2880 else (("시간", 60.0) if gmax >= 180 else ("분", 1.0))
    to_unit = TICKS_PER_UNIT * MPT / div
    tf = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                       subplot_titles=(f"처리량 (개 / {dur(sm['time']['bucket_ticks'] / TICKS_PER_UNIT)})",
                                       f"통과 간격 ({unit}, 마지막 도착 이후)", "대기 부품 수 (개)"))
    hov = [ck(v) for v in x]
    gap_v = ser["gap"][i][vis] * to_unit
    rows = (
        # (행, 값, 마우스 오버 형식, 기준값, 기준 이름)
        (1, thr_v, "처리량 %{y:.0f}개", float(ser["base_thr_mean"][i]), "정상 평균"),
        (2, gap_v, "간격 %{y:.1f}" + unit, float(ser["base_gap_p95"][i] * to_unit), "정상 범위 상한"),
        (3, q_v, "대기 %{y:.0f}개", float(ser["base_q_p95"][i]), "정상 범위 상한"),
    )
    for row, y, fmt, ref, ref_name in rows:
        tf.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=ACCENT, width=1.6),
                                hovertext=hov, hovertemplate="%{hovertext}<br>" + fmt + "<extra></extra>"), row, 1)
        ymax = float(np.nanmax(y)) if len(y) and not np.all(np.isnan(y)) else 0.0
        # 기준선은 그래프마다 하나만. 실제 값보다 지나치게 크면(축이 눌려 값이 안 보이므로) 그리지 않는다.
        show_ref = ref > 0 and (ymax == 0 and row == 1 or ref <= max(ymax, 1e-9) * 2.5)
        top = max(ymax, ref if show_ref else 0) * 1.15 or 1
        tf.update_yaxes(range=[0, top], row=row, col=1)
        if show_ref:
            tf.add_hline(y=ref, line_dash="dot", line_color=TXT3, line_width=1, row=row, col=1,
                         annotation_text=ref_name, annotation_position="top left",
                         annotation_font=dict(size=9, color=TXT2))
    for e in events:
        if e["station"] == s_sel and e["type"] == "정지 의심" and e["snapshots"][0]["tick"] <= tick \
                and (e["end_time"] is None or e["end_time"] >= lo_t):
            end = e["end_time"] if e["end_tick"] is not None and e["end_tick"] <= tick else now
            tf.add_vrect(x0=max(e["start_time"], lo_t), x1=end, fillcolor="#E5534B", opacity=0.16, line_width=0)
    tf.update_layout(height=430, showlegend=False, margin=dict(l=5, r=5, t=25, b=5),
                     plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color=TXT2))
    tf.update_yaxes(gridcolor=GRID, zeroline=False, tickfont=dict(size=10, color=TXT3))
    tf.update_xaxes(showgrid=False, linecolor=GRID, tickfont=dict(size=10, color=TXT2))
    tf.update_annotations(font_size=11)
    if len(x):
        clock_axis(tf, float(x.min()), float(x.max()))
    st.plotly_chart(tf, key="trend")


def crew_bar(cs: dict):
    """설비별 투입 인원 행 (설비 · 막대 · 'n명 투입 중 · 복귀 시각'). 투입 중 = 남색, 출동 예정 = 연한 남색. 정비 완료는 뺀다."""
    agg = {}
    for x in cs["rows"]:
        if x["state"] in ("정비 중", "출동 예정"):
            a = agg.setdefault(x["station"], {"정비 중": 0, "출동 예정": 0, "end": x["end"]})
            a[x["state"]] += int(x["staff"])
            a["end"] = max(a["end"], x["end"])
    if not agg:
        st.caption("지금 정비 중인 설비가 없습니다. 승인한 정비는 모두 끝났습니다.")
        return
    names = sorted(agg, key=lambda s: (-(agg[s]["정비 중"] + agg[s]["출동 예정"]), agg[s]["end"]))
    # 막대 전체 길이 = 가장 많이 투입된 설비와 '조치 권고 1건 출동 인원' 중 큰 값 (1명이 화면을 꽉 채우지 않게)
    scale = max(max(agg[s]["정비 중"] + agg[s]["출동 예정"] for s in names), int(sm["rules"]["staff_act"]), 1)
    rows_html = []
    for s in names:
        a = agg[s]
        segs = "".join(
            f"<div class='cb-seg' style='width:{a[k] / scale * 100:.1f}%;background:{col}'></div>"
            for k, col in (("정비 중", ACCENT), ("출동 예정", "#2E4A7D")) if a[k])
        txt = " · ".join(t for t in (f"<b>{a['정비 중']}명 투입 중</b>" if a["정비 중"] else "",
                                     f"{a['출동 예정']}명 출동 예정" if a["출동 예정"] else "") if t)
        rows_html.append(
            f"<div class='cb-row'><div class='cb-st'>{s}</div><div class='cb-track'>{segs}</div>"
            f"<div class='cb-txt'>{txt}<span class='cb-end'>복귀 {ck(a['end'])}</span></div></div>")
    st.markdown(
        "<style>.cb-row{display:flex;align-items:center;gap:12px;padding:7px 0;border-bottom:1px solid #1F2630}"
        ".cb-st{width:72px;font-weight:600;font-size:14px;color:#E5E7EB}"
        ".cb-track{flex:1;height:18px;background:#1F2630;border-radius:4px;display:flex;overflow:hidden}"
        ".cb-seg{height:100%}"
        ".cb-txt{width:250px;font-size:13px;color:#D1D5DB}"
        ".cb-end{color:#6B7280;margin-left:8px;font-size:12px}</style>"
        + "".join(rows_html), unsafe_allow_html=True)


def tab_bm(tick: int, now: float):
    plan = current_plan(res, tick)
    r = sm["rules"]
    setting_txt = (f"사용자 설정: 투입 가능 정비 인원 {r['total_staff']}명, 1건당 출동 인원 관찰 필요 {r['staff_obs']}명 / "
                   f"조치 권고 {r['staff_act']}명. 1건당 조치 시간(기준 학습 구간에서 관찰한 정지 지속 시간): 관찰 필요 "
                   f"{dur(r['dur_obs'] / TICKS_PER_UNIT)}, 조치 권고 {dur(r['dur_act'] / TICKS_PER_UNIT)}.")
    # 설비별 투입 인원 (전체·가용 인원 수는 상단 '정비 인원' 카드에 있으므로 여기서는 설비별만)
    cs = crew_status(now)
    sec("설비별 투입 인원", "crew")
    if cs["rows"]:
        crew_bar(cs)
    else:
        st.caption("아직 승인한 정비가 없습니다. 오른쪽 Agent 브리핑에서 '승인'을 누르면 정비 인원이 출동합니다.")

    def plan_state(eid):
        d = ss.dispatch.get(eid)
        if d:
            return "정비 중" if d["start"] <= now < d["end"] else ("정비 완료" if now >= d["end"] else "출동 예정")
        return ss.approvals.get(eid, "승인 대기")

    st.markdown(f"<div class='sec'>지금 투입할 정비 인원 {q(HELP['now_plan'] + ' ' + setting_txt)}"
                f"<span class='meta' style='font-weight:400;margin-left:6px'>가용 {cs['avail']}명</span></div>",
                unsafe_allow_html=True)
    if plan["assignments"]:
        # 순위는 지금 남은 건만 1부터 다시 매긴다 (끝난 건이 빠져 번호가 건너뛰지 않게).
        # 투입 시작·완료 시각이 표에 있으므로 같은 내용의 일정 막대그래프는 두지 않는다.
        asg = plan["assignments"][:20]
        st.dataframe(pd.DataFrame([{
            "순위": k + 1, "설비": a["station"], "등급": grade_label(a["grade"]), "인원(명)": a["staff"],
            "투입 시작": ck(a["start"]), "완료 예정": ck(a["end"]), "지연 부품(개)": a["delayed_parts"],
            "상태": plan_state(a["event_id"])} for k, a in enumerate(asg)]),
            hide_index=True, height=min(36 * (len(asg) + 1) + 2, 300))
    else:
        st.caption("지금 정비 인원 투입이 필요한 설비가 없습니다.")



def tab_records(tick: int, now: float):
    plan = current_plan(res, tick)
    asg = {a["event_id"]: a for a in plan["assignments"]}
    sec("이상 발생 기록", "records", "최신 200건")
    rows = []
    for e in sorted([e for e in events if e["snapshots"][0]["tick"] <= tick], key=lambda e: -e["snapshots"][0]["tick"])[:200]:
        sn = grade_at(e, tick)
        closed = e["end_tick"] is not None and e["end_tick"] <= tick
        d = e["duration"] if closed else round((tick - e["start_tick"]) / TICKS_PER_UNIT, 2)
        a = asg.get(e["id"])
        rows.append({"설비": e["station"], "유형": type_label(e["type"]), "등급": grade_label(sn["grade"]),
                     "상태": "종료" if closed else "진행 중",
                     "처리": ss.approvals.get(e["id"], "-").replace("승인 후 발송됨", "승인"),
                     "발생 시점": ck(e["start_time"]), "지속 시간": dur(d),
                     "지연 부품(개)": sn["delayed_parts"], "영향 설비": "·".join(sn["affected_stations"]) or "-",
                     "반복(회)": sn["repeat_count"],
                     "정비 인원·순서": f"{a['staff']}명 · {a['order']}번" if a else ("-" if closed else "확인 필요"),
                     "판단 근거": sn["evidence"], "원본 시간값": e["start_time"]})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, height=min(36 * (len(rows) + 1), 330))
    else:
        st.caption("아직 이상 발생 기록이 없습니다.")

    sec("반복 정지 설비 (BM 개선 대상)", "repeat")
    reps = pipeline.repeat_table(events, upto_tick=tick)
    if reps:
        st.dataframe(pd.DataFrame([{"설비": x["station"], "설비 정지 의심(회)": x["stop"], "병목(회)": x["bottleneck"],
                                    "조치 권고(회)": x["action"],
                                    "BM 개선 대상": "✅ 대상" if x["review"] else ""} for x in reps[:12]]),
                     hide_index=True, height=min(36 * (min(len(reps), 12) + 1), 300))
    else:
        st.caption("아직 기록이 없습니다.")


def tab_report(tick: int, now: float):
    sec("Word 보고서", extra=f"{ck(now)} 기준")
    if st.button("보고서 만들기", type="primary"):
        with st.spinner("보고서 작성 중..."):
            ss.report = build_report(res, tick, current_plan(res, tick), ss.get("approvals", {}), mpt=MPT,
                                     crew=crew_status(now))
            ss.report_name = f"BM_보고서_{kind}_{ck(now).replace(':', '').replace(' ', '_')}.docx"
            ss.report_at = ck(now)
    if ss.get("report"):
        st.download_button("Word 다운로드", ss.report, file_name=ss.report_name,
                           mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        from docx import Document
        doc = Document(io.BytesIO(ss.report))
        with st.container(height=420, border=True):
            st.caption(f"미리보기 ({ss.get('report_at')} 기준, 본문 글자와 표 개수)")
            for p in doc.paragraphs[:60]:
                if p.text.strip():
                    bold = any(r.bold for r in p.runs)
                    st.markdown(f"**{p.text}**" if bold else p.text)
            st.caption(f"표 {len(doc.tables)}개 · 그림 {len(doc.inline_shapes)}개 포함")


# ---------------------------------------------------------------------------
# Agent 브리핑 패널
# ---------------------------------------------------------------------------
def approve(eid: int, a: dict | None, now: float) -> str | None:
    """승인 = 정비 인원 출동. 가용 인원이 모자라면 승인하지 않고 이유를 돌려준다."""
    need = int(a["staff"]) if a else 0
    cs = crew_status(now)
    if need > cs["avail"]:
        return f"가용 인원 {cs['avail']}명 < 필요 {need}명 — 정비가 끝나 인원이 복귀하면 다시 승인하세요."
    ss.approvals[eid] = "승인 후 발송됨"
    if a:
        length = float(a["end"]) - float(a["start"])         # 최적화가 정한 조치 시간
        start = max(now, float(a["start"]))                   # 승인한 시점(또는 계획 시작) 부터 출동
        ss.dispatch[eid] = {"station": by_id[eid]["station"], "staff": need, "grade": by_id[eid]["grade"],
                            "start": round(start, 2), "end": round(start + length, 2)}
    return None


def render_alert_msg(f: dict, al: dict, buttons: bool = True, now: float = 0.0):
    ev = by_id[al["event_id"]]
    snap = ev["snapshots"][al["snapshot"]]
    g = snap["grade"]
    status = ss.approvals.get(ev["id"], "승인 대기")
    kind_txt = {"new": "새 이상", "upgrade": "등급 상승", "spread": "영향 확산"}[al["kind"]]
    col = GRADE_COLOR.get(g, "#999")
    a = al.get("assignment")
    # 한 줄 요약: 지연 부품 · 정비 인원 · 순서 (상세는 '자세히' 안에)
    sub = [f"지연 부품 {snap['delayed_parts']}개"]
    if a:
        sub.append(f"정비 {a['staff']}명 · {a['order']}순위")
    elif g == "판단 보류":
        sub.append("기록 확인 필요")
    with st.container(border=True):
        st.markdown(f"<div class='msg'><div class='h'><span class='badge' style='--c:{col}'>{grade_label(g)}</span>"
                    f"{ev['station']} {type_label(ev['type'])}</div><div class='t'>{ck(al['time'])}</div></div>"
                    f"<div class='msg-sub'>{kind_txt} · {' · '.join(sub)}</div>", unsafe_allow_html=True)
        with st.expander("자세히", expanded=False):
            if al.get("text_source", "template") == "llm":
                st.text(al["text"])
            else:
                view = pipeline._event_view(ev, snap, al["tick"])
                view["assignment"] = a
                st.text(template_alert(view, MPT))
            st.caption(f"원본 시간값 {al['time']}")
        if status == "승인 대기" and buttons:
            b1, b2, _ = st.columns([1, 1, 1.2])
            if b1.button("승인", key=f"ok_{al['event_id']}_{al['snapshot']}", type="primary" if g == "조치 권고" else "secondary", use_container_width=True,
                         help="승인하면 정비 인원이 출동합니다. 조치 시간이 끝나면 가용 인원으로 돌아옵니다."):
                msg = approve(ev["id"], a, now)
                if msg:
                    ss.setdefault("approve_warn", {})[ev["id"]] = msg
                else:
                    ss.get("approve_warn", {}).pop(ev["id"], None)
                    # 승인한 알림은 패널에서 사라지므로 짧게 결과만 알려 준다
                    st.toast(f"{ev['station']} 승인" + (f" · 정비 {a['staff']}명 출동" if a else ""))
                st.rerun(scope="fragment")
            if b2.button("보류", key=f"hold_{al['event_id']}_{al['snapshot']}", use_container_width=True):
                ss.approvals[ev["id"]] = "보류"
                st.toast(f"{ev['station']} 보류")
                st.rerun(scope="fragment")
            warn = ss.get("approve_warn", {}).get(ev["id"])
            if warn:
                st.markdown(f"<div class='meta' style='color:#F87171'>{warn}</div>", unsafe_allow_html=True)
        elif status != "승인 대기":
            d = ss.dispatch.get(ev["id"])
            if d and d["start"] <= now < d["end"]:
                txt = f"🔧 정비 중 · {d['staff']}명 · {ck(d['end'])} 복귀"
            elif d and now >= d["end"]:
                txt = f"✔ 정비 완료 · {d['staff']}명 복귀"
            elif d:
                txt = f"출동 예정 · {d['staff']}명 · {ck(d['start'])}"
            else:
                txt = f"✔ {status}"
            st.markdown(f"<div class='meta'>{txt}</div>", unsafe_allow_html=True)


def render_plan_msg(f: dict):
    p = res["plans"][f["key"]]
    asg = p["assignments"]
    first = asg[0]
    with st.container(border=True):
        st.markdown(f"<div class='msg'><div class='h'><span class='badge' style='--c:#5B8DEF'>배치</span>"
                    f"1순위 {first['station']} · 정비 {first['staff']}명</div><div class='t'>{ck(t_of(f['tick']))}</div></div>"
                    f"<div class='msg-sub'>정비 인원 배치 갱신 · 진행 중 이상 {len(asg)}건</div>", unsafe_allow_html=True)
        with st.expander("자세히", expanded=False):
            st.text("\n".join(f"{x['order']}순위 {x['station']} ({grade_label(x['grade'])}) 정비 {x['staff']}명, "
                              f"{ck(x['start'])} – {ck(x['end'])}" for x in asg[:5])
                    + ("\n…" if len(asg) > 5 else "") + "\n(담당자 승인 후 발송)")


def panel(tick: int, now: float):
    """담당자가 지금 판단할 것만 보여 준다.
    - 아직 진행 중이고 승인·보류하지 않은 이상: 이상마다 최신 알림 1개 (승인/보류 버튼 포함)
    - 승인·보류한 이상과 이미 끝난 이상의 알림은 사라진다 (기록은 '이상 발생 기록' 탭)
    - 정비 인원 배치 갱신은 가장 최근 1개만"""
    n = int(np.searchsorted(res["feed_ticks"], tick, side="right"))
    feed = res["feed"][:n]
    latest, last_plan = {}, None
    for k, f in enumerate(feed):
        if f["kind"] == "alert":
            latest[alerts[f["idx"]]["event_id"]] = k
        else:
            last_plan = k
    todo = [k for eid, k in latest.items()
            if is_open_at(by_id[eid], tick) and ss.approvals.get(eid, "승인 대기") == "승인 대기"]
    # 조치 권고 → 관찰 필요 → 확인 필요 순, 같은 등급이면 최신 순. 화면에는 앞에서부터 PANEL_MSGS개만.
    rank = {"조치 권고": 0, "관찰 필요": 1, "판단 보류": 2}
    todo.sort(key=lambda k: (rank.get(grade_at(by_id[alerts[feed[k]["idx"]]["event_id"]], tick)["grade"], 3), -k))
    shown = todo[:PANEL_MSGS]
    sec("Agent 브리핑", extra=f"확인할 알림 {len(todo)}건")
    box = st.container(height=600, border=False)
    with box:
        if last_plan is not None:
            render_plan_msg(feed[last_plan])
        if not feed:
            with st.container(border=True):
                st.markdown(f"<div class='msg-sub'>{ck(now)}부터 모니터링을 시작합니다. 이상이 생기면 여기에 바로 알려 드립니다.</div>",
                            unsafe_allow_html=True)
        elif not todo and tick < r_to:
            st.markdown("<div class='meta'>지금 확인할 알림이 없습니다. 승인·보류한 알림과 지난 기록은 '이상 발생 기록' 탭에 있습니다.</div>",
                        unsafe_allow_html=True)
        for k in shown:
            render_alert_msg(feed[k], alerts[feed[k]["idx"]], buttons=True, now=now)
        if len(todo) > len(shown):
            st.markdown(f"<div class='meta'>그 밖의 알림 {len(todo) - len(shown)}건은 '이상 발생 기록' 탭에 있습니다.</div>",
                        unsafe_allow_html=True)
        if tick >= r_to:
            cmp = sm["comparison"]
            with st.container(border=True):
                st.markdown(f"<div class='msg'><div class='h'><span class='badge' style='--c:#9CA3AF'>요약</span>"
                            f"모니터링 구간 종료</div><div class='t'>{ck(t_of(r_to))}</div></div>", unsafe_allow_html=True)
                summary = sm.get("summary_text", "") if sm.get("summary_source") == "llm" else template_summary({
                    "optimization": {**{k: cmp[k] for k in ("total_loss", "baseline_loss", "reduction_pct")},
                                     "assignments": sm["final_schedule"]["assignments"]},
                    "repeat_top": [r for r in pipeline.repeat_table(events) if r["review"]][:5]})
                st.markdown(f"<div class='msg-sub'>{summary}</div>", unsafe_allow_html=True)
                with st.expander("이상 건수", expanded=False):
                    st.caption(f"이상 {sm['counts']['events']:,}건 (설비 정지 의심 {sm['counts']['stop']:,}건, "
                               f"병목 {sm['counts']['bottleneck']:,}건)")


# ---------------------------------------------------------------------------
# 화면 (모니터링 중에는 1초마다 다시 그림, 일시정지하면 멈춤)
# ---------------------------------------------------------------------------
@st.fragment(run_every=REFRESH_SEC if ss.get("playing") else None)
def live_view():
    if ss.get("playing"):
        wall = time.time()
        # 실제로 흐른 화면 시간만큼 진행 (화면 그리기가 늦어도 속도 유지, 탭을 오래 비운 경우만 10초로 제한)
        dt = 0.0 if ss.get("last_wall") is None else min(max(wall - ss.last_wall, 0.0), 10.0)
        ss.last_wall = wall
        ss.tick_f = min(float(ss.get("tick_f", ss.tick)) + dt * speed_min / MPT, float(r_to))
        ss.tick = int(ss.tick_f)
        # 공장 전체 데이터가 없는 구간(주말 등)은 보여줄 것이 없으므로 끝 시점으로 바로 넘어간다
        hit = empty_period_at(sm, ss.tick)
        if hit:
            ss.tick = min(hit[1], r_to)
            ss.tick_f = float(ss.tick)
        ss.frame = ss.get("frame", 0) + 1
        if ss.tick >= r_to:
            ss.playing = False
            st.rerun()
    tick = int(ss.tick)
    now = t_of(tick)
    blink = bool(ss.get("playing")) and ss.get("frame", 0) % 2 == 0

    # 상단 상태 바 (한 줄)
    cnt = {g: 0 for g in ("관찰 필요", "조치 권고", "판단 보류")}
    for e in events:
        if is_open_at(e, tick):
            g = grade_at(e, tick)["grade"]
            if g in cnt:
                cnt[g] += 1
    prog = (tick - r_from) / max(r_to - r_from, 1)
    state = "모니터링 중" if ss.get("playing") else ("모니터링 구간 종료" if tick >= r_to else "일시정지")
    tm = sm["time"]
    # 긴 안내는 ⓘ 안으로 (마우스를 올리면 보임)
    detail = (f"원본 시간값 {now}. 기준 학습 구간 {clock_full(tm['t_min'], MPT)} – {clock_full(tm['split_time'], MPT)}, "
              f"모니터링 구간 {clock_full(tm['split_time'], MPT)} – {clock_full(tm['t_max'], MPT)}. "
              f"마지막 계산 시각 {sm.get('recomputed', sm['created'])}. "
              + ("문장 작성: 정해진 문장 틀(LLM 미연결)." if sm.get("llm_label", LABEL_TEMPLATE) == LABEL_TEMPLATE else ""))

    def card(label, value, help_txt, color=None, sub="", main=False, faded=False):
        dot = f"<span class='dot' style='background:{color}'></span>" if color else ""
        vcol = ""                               # 숫자는 검정, 등급 색은 점·왼쪽 띠에만
        border = f"border-left:4px solid {color};" if color else ""
        op = "opacity:0.55;" if faded else ""
        return (f"<div class='kpi{' main' if main else ''}' style='{border}'><div class='l'>{dot}{label} {q(help_txt)}</div>"
                f"<div class='v' style='{vcol}{op}'>{value}</div><div class='s'>{sub}</div></div>")

    cs = crew_status(now)

    def cnt_card(label, g, help_key):
        # 등급 색은 항상 표시 (0건이면 숫자만 연하게)
        n = cnt[g]
        return card(label, f"{n}건", HELP[help_key], GRADE_COLOR[g], "진행 중" if n else "없음", faded=n == 0)

    st.markdown(
        "<div class='kpis'>"
        + card("현재 시점", ck(now), HELP["now"] + " " + detail,
               sub=f"{to_datetime(now, MPT):%Y}년 · {state} · 1초 = {dur(speed_min / MPT / TICKS_PER_UNIT)}", main=True)
        + card("모니터링 진행률", f"{prog * 100:.1f}%", HELP["progress"], sub=f"끝 {ck(tm['t_max'])}")
        + cnt_card("조치 권고", "조치 권고", "act")
        + cnt_card("관찰 필요", "관찰 필요", "obs")
        + cnt_card("확인 필요", "판단 보류", "hold")
        + card("정비 인원", f"{cs['busy']}명 정비 중", HELP["crew"], ACCENT,
               sub=f"가용 {cs['avail']}명 / 전체 {cs['total']}명")
        + f"</div><div class='bar'><div style='width:{prog * 100:.2f}%'></div></div>", unsafe_allow_html=True)

    left, right = st.columns([2, 1], gap="medium")
    with left:
        if ss.get("main_tab") not in (None, *TABS):   # 없어진 탭이 기억돼 있으면 첫 탭으로
            ss.main_tab = TABS[0]
        tabs = st.tabs(TABS, key="main_tab", on_change="rerun")
        active = ss.get("main_tab") or TABS[0]
        # 선택된 탭만 그린다 (1초 갱신 때 부하를 줄이기 위해)
        with tabs[0]:
            if active == TABS[0]:
                tab_monitor(tick, now, blink)
        with tabs[1]:
            if active == TABS[1]:
                tab_bm(tick, now)
        with tabs[2]:
            if active == TABS[2]:
                tab_records(tick, now)
        with tabs[3]:
            if active == TABS[3]:
                tab_report(tick, now)
    with right:
        panel(tick, now)


live_view()

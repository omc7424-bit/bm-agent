"""⑧ 보고서 (python-docx)

서식: 맑은 고딕, 검은 글씨, 제목 스타일·자동 번호 없이 굵은 글씨로만 구분.
구성: Agent 브리핑(지금 투입할 정비 인원) → 이상 발생 기록 → 실시간 물류 흐름 지도·설비별 흐름 추이
      → BM 인원 최적 배치 → 정비 인원별 지연 비교·적정 인원 → 반복 정지 설비(BM 개선 대상) → 신뢰도 점검
모든 숫자는 pipeline이 계산해 저장한 결과에서 가져온다.
"""
from __future__ import annotations

import io

import numpy as np

from detector import GRADE_RANK
from llm import LABEL_TEMPLATE, grade_label, template_staffing, template_summary, type_label
from stream import (DEFAULT_MIN_PER_TICK, TICKS_PER_UNIT, clock_full, clock_note, clock_text, clock_ticks, duration_text,
                    station_key)

FONT = "맑은 고딕"
# 등급 색 (화면·보고서 공통): 채도를 낮춘 부드러운 색. 정상은 아주 옅은 초록, 급한 것(조치 권고)만 또렷하게
GRADE_COLOR = {"정상": "#CFE5D6", "관찰 필요": "#E0A548", "조치 권고": "#D2504C", "판단 보류": "#9F97C8"}
ROW_GAP = 1.6


def line_row_y(line: int) -> float:
    return -line * ROW_GAP


def flow_layout(stations: list[str]) -> dict:
    """흐름 지도 좌표.
    - 라인마다 한 줄 (L0 → L1 → L2 → L3, 위에서 아래)
    - 줄 안에서는 공정 순서(스테이션 번호)대로 왼쪽 → 오른쪽
    - 병렬 스테이션(같은 공정을 나눠 맡는 스테이션)은 같은 열에 위아래로 배치
    """
    from baseline import PARALLEL_GROUPS
    group_of = {s: tuple(g) for g in PARALLEL_GROUPS for s in g}
    present = set(stations)
    pos = {}
    for line in sorted({station_key(s)[0] for s in stations}):
        members = sorted([s for s in stations if station_key(s)[0] == line], key=station_key)
        col, seen = -1, {}
        for s in members:
            g = group_of.get(s)
            if g and g in seen:
                c = seen[g]
            else:
                col += 1
                c = col
                if g:
                    seen[g] = c
            if g:
                grp = [x for x in g if x in present]
                k, m = grp.index(s), len(grp)
                off = ((m - 1) / 2 - k) * 0.45
            else:
                off = 0.0
            pos[s] = (c, line_row_y(line) + off)
    return pos


def station_grades_at(events: list[dict], tick: int) -> dict:
    """tick 시점 스테이션별 최고 등급과 유형 (열린 이벤트 기준)."""
    from detector import grade_at, is_open_at
    out = {}
    for ev in events:
        if is_open_at(ev, tick):
            g = grade_at(ev, tick)["grade"]
            cur = out.get(ev["station"])
            if cur is None or GRADE_RANK[g] > GRADE_RANK[cur["grade"]]:
                out[ev["station"]] = {"grade": g, "types": set()}
            out[ev["station"]]["types"] = out[ev["station"]].get("types", set()) | {ev["type"]}
    return out


# ---------------------------------------------------------------------------
def _setup_doc():
    from docx import Document
    from docx.oxml.ns import qn
    from docx.shared import Pt, RGBColor
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = FONT
    st.font.size = Pt(10)
    st.font.color.rgb = RGBColor(0, 0, 0)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    return doc


def _bold(doc, text, size=None):
    from docx.shared import Pt
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = True
    if size:
        r.font.size = Pt(size)
    return p


def _table(doc, header, rows, max_rows=40):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]
        c.text = ""
        c.paragraphs[0].add_run(str(h)).bold = True
    for r in rows[:max_rows]:
        cells = t.add_row().cells
        for i, v in enumerate(r):
            cells[i].text = "" if v is None else str(v)
    if len(rows) > max_rows:
        doc.add_paragraph(f"(전체 {len(rows)}건 중 {max_rows}건 표시)")
    return t


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    for f in ("Malgun Gothic", "맑은 고딕", "AppleGothic", "NanumGothic"):
        if f in names:
            plt.rcParams["font.family"] = f
            break
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def _img(doc, fig, width_cm=16):
    from docx.shared import Cm
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    doc.add_picture(buf, width=Cm(width_cm))
    import matplotlib.pyplot as plt
    plt.close(fig)


# ---------------------------------------------------------------------------
def build_report(res: dict, tick: int, current_schedule: dict, approvals: dict | None = None,
                 mpt: float = DEFAULT_MIN_PER_TICK, crew: dict | None = None) -> bytes:
    """res: pipeline.load_results() 결과, tick: 보고 기준 시각(bin), current_schedule: 그 시각의 BM 투입 계획.
    mpt: 표시용 환산 비율(0.01 단위 = mpt분). 숫자 계산은 원래 데이터 시각 그대로이고 표시만 환산한다."""
    from detector import grade_at
    plt = _plt()
    approvals = approvals or {}
    ck = lambda t: clock_text(t, mpt)
    dr = lambda d: duration_text(d, mpt)

    def clock_xaxis(ax, lo, hi):
        vals, texts = clock_ticks(lo, hi, mpt)
        ax.set_xticks(vals, [t.replace("<br>", "\n") for t in texts], fontsize=7)

    sm, events, ser = res["summary"], res["events"], res["series"]
    t0 = sm["data"]["t0_tick"]
    now = round((tick + t0) / TICKS_PER_UNIT, 2)
    shown = [e for e in events if e["snapshots"][0]["tick"] <= tick]
    stations = [str(s) for s in ser["stations"]]
    label = sm.get("llm_label", LABEL_TEMPLATE)

    doc = _setup_doc()
    _bold(doc, "실시간 물류흐름 모니터링 및 BM 개선 보고서", 14)
    doc.add_paragraph(f"현재 시점 {clock_full(now, mpt)} (원본 시간값 {now}) · 분석 데이터: "
                      f"{'테스트용 샘플(5,000행)' if sm['kind'] == 'sample' else '실제 전체 데이터(Bosch)'}"
                      f" · 문장 작성: {'정해진 문장 틀' if label == LABEL_TEMPLATE else label}")
    doc.add_paragraph(f"※ {clock_note(mpt)}. 표의 시각 옆 괄호는 원본 시간값입니다.")
    doc.add_paragraph("※ 투입 가능 정비 인원과 1건당 출동 인원은 데이터가 아닌 사용자 설정값입니다. "
                      "고장 기록이 없어 '설비 정지 의심'으로 표현하며, 원인은 현장 확인이 필요합니다. "
                      "모든 BM 대응 지시는 담당자 승인 후 발송합니다.")

    # 1. 요약 (첫 페이지)
    _bold(doc, "1. 지금 투입할 정비 인원")
    if crew:
        doc.add_paragraph(f"정비 인원 현황(담당자 승인 기준): 정비 중 {crew['busy']}명, 가용 {crew['avail']}명 / 전체 {crew['total']}명.")
        if crew["rows"]:
            _table(doc, ["상태", "설비", "정비 인원(명)", "출동", "복귀"],
                   [[x["state"], x["station"], x["staff"], ck(x["start"]), ck(x["end"])] for x in crew["rows"]], 20)
    asg = current_schedule.get("assignments", [])
    if asg:
        rows = []
        by_id = {e["id"]: e for e in events}
        for a in asg:
            ev = by_id.get(a["event_id"], {})
            rows.append([a["order"], a["station"], type_label(ev.get("type", "")), grade_label(a["grade"]), a["staff"],
                         f"{ck(a['start'])}\n({a['start']})", f"{ck(a['end'])}\n({a['end']})",
                         a["delayed_parts"], approvals.get(a["event_id"], "승인 대기")])
        _table(doc, ["순서", "설비", "유형", "등급", "정비 인원(명)", "투입 시작", "완료 예정", "지연 부품(개)", "승인 상태"], rows)
    else:
        doc.add_paragraph("지금 정비 인원 투입이 필요한 설비가 없습니다.")
    hold = [e for e in shown if e["end_tick"] is None or e["end_tick"] > tick]
    hold = [e for e in hold if grade_at(e, tick)["grade"] == "판단 보류"]
    if hold:
        doc.add_paragraph("확인 필요(기록 누락 가능 구간): " + ", ".join(sorted({e["station"] for e in hold})))
    at_end = tick >= sm["time"]["replay_to_tick"]
    if at_end and sm.get("summary_source") == "llm":
        summary_text = sm.get("summary_text", "")
    else:
        summary_text = template_summary({
            "optimization": {**{k: sm["comparison"][k] for k in ("total_loss", "baseline_loss", "reduction_pct")},
                             "assignments": asg},
            "repeat_top": [r for r in _repeat_upto(events, tick) if r["review"]][:5]})
    _bold(doc, "Agent 브리핑")
    doc.add_paragraph(summary_text)

    # 2. 이상 발생 기록
    doc.add_page_break()
    _bold(doc, "2. 이상 발생 기록 (최신순)")
    rows = []
    for e in sorted(shown, key=lambda e: -e["snapshots"][0]["tick"]):
        sn = grade_at(e, tick)
        closed = e["end_tick"] is not None and e["end_tick"] <= tick
        d = e["duration"] if closed else round((tick - e["start_tick"]) / TICKS_PER_UNIT, 2)
        rows.append([e["station"], type_label(e["type"]), grade_label(sn["grade"]), f"{ck(e['start_time'])}\n({e['start_time']})",
                     f"{dr(d)}\n({d})", sn["delayed_parts"],
                     "·".join(sn["affected_stations"]) or "-", sn["repeat_count"], sn["evidence"]])
    _table(doc, ["설비", "유형", "등급", "발생 시점(환산)\n(원본 시간값)", "지속 시간(환산)\n(원본)", "지연 부품(개)",
                 "영향 설비", "반복(회)", "판단 근거"], rows)

    # 3. 흐름 구조도·추세
    doc.add_page_break()
    _bold(doc, "3. 실시간 물류 흐름 지도")
    doc.add_paragraph("노드 = 설비, 노드 색 = 현재 등급(회색 정상, 노랑 관찰 필요, 빨강 조치 권고, 보라 확인 필요), 검은 테두리 = 병목, 선 = 부품 이동")
    sg = station_grades_at(events, tick)
    pos = flow_layout(stations)
    fig, ax = plt.subplots(figsize=(11, 4))
    for f, t in zip(ser["edge_from"], ser["edge_to"]):
        (x0, y0), (x1, y1) = pos[stations[int(f)]], pos[stations[int(t)]]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color="#bbbbbb", lw=0.6))
    for s in stations:
        g = sg.get(s, {}).get("grade", "정상")
        x, y = pos[s]
        ax.scatter([x], [y], s=120, color=GRADE_COLOR[g], edgecolors="black",
                   linewidths=2.5 if "병목" in sg.get(s, {}).get("types", ()) else 0.5, zorder=3)
        ax.text(x, y + 0.35, s.split("_")[1], ha="center", fontsize=6)
    lines_present = sorted({station_key(s)[0] for s in stations})
    ax.set_yticks([line_row_y(l) for l in lines_present], [f"L{l}" for l in lines_present])
    ax.set_xticks([])
    _img(doc, fig)

    _bold(doc, "설비별 흐름 추이 (이상이 많은 설비)")
    t_end = ser["t_end"]
    rf = sm["time"]["split_time"]
    vis = (t_end >= rf) & (t_end <= now)
    top = [r["station"] for r in _repeat_upto(events, tick)][:3]
    for s in top:
        i = stations.index(s)
        fig, axs = plt.subplots(3, 1, figsize=(10, 5.5), sharex=True)
        x = t_end[vis]
        to_min = TICKS_PER_UNIT * mpt          # 도착 간격은 환산 분으로 표시 (값 자체는 저장된 계산 결과)
        for ax, key, base, name, k in ((axs[0], "throughput", ser["base_thr_mean"][i], "처리량", 1.0),
                                       (axs[1], "gap", ser["base_gap_p95"][i], "도착 간격(환산 분)", to_min),
                                       (axs[2], "queue", ser["base_q_med"][i], "대기 부품 수", 1.0)):
            ax.plot(x, ser[key][i][vis] * k, color="black", lw=0.8)
            ax.axhline(base * k, ls="--", color="gray", lw=0.8)
            ax.set_ylabel(name, fontsize=8)
        for e in shown:
            if e["station"] == s and e["type"] == "정지 의심":
                end = e["end_time"] if e["end_time"] is not None and e["end_tick"] <= tick else now
                for ax in axs:
                    ax.axvspan(e["start_time"], end, color="#f8d0d0", alpha=0.6, lw=0)
        axs[0].set_title(f"{s} (점선 = 정상 범위, 연한 빨강 = 설비 정지 의심 구간)", fontsize=9)
        if len(x):
            clock_xaxis(axs[2], float(x.min()), float(x.max()))
        axs[2].set_xlabel("시각", fontsize=8)
        _img(doc, fig, 15)

    # 4. 최적화 결과
    doc.add_page_break()
    _bold(doc, "4. BM 인원 최적 배치 (모니터링 구간 전체)")
    cmp = sm["comparison"]
    r = sm["rules"]
    doc.add_paragraph(
        f"설정값: 투입 가능 정비 인원 {r['total_staff']}명, 1건당 출동 인원 관찰 필요 {r['staff_obs']}명 / 조치 권고 {r['staff_act']}명. "
        f"조치 시간(데이터 기반): 관찰 필요 {dr(r['dur_obs'] / TICKS_PER_UNIT)} ({r['dur_obs'] / TICKS_PER_UNIT}), "
        f"조치 권고 {dr(r['dur_act'] / TICKS_PER_UNIT)} ({r['dur_act'] / TICKS_PER_UNIT}).")
    doc.add_paragraph(
        f"기존 방식(발생 순) 누적 지연 부품 수 {cmp['baseline_loss']:,}개 → 최적 배치 {cmp['total_loss']:,}개 "
        + (f"(감소율 {cmp['reduction_pct']}%, " if cmp.get("reduction_pct") is not None and cmp["baseline_loss"] > 0
           else "(비교할 대응 없음, ")
        + f"대상 이상 {cmp['jobs']}건). 누적 지연 부품 수 = Σ 지연 부품 수 × 대응 완료까지 시간."
        + (" 이상이 많아 최근 300건으로 비교했습니다." if cmp.get("note") else ""))
    a2 = cmp["optimized"]["assignments"][:30]
    if a2:
        fig, ax = plt.subplots(figsize=(10, max(2.5, len(a2) * 0.25)))
        for k, a in enumerate(a2):
            ax.barh(k, a["end"] - a["start"], left=a["start"], color=GRADE_COLOR.get(a["grade"], "gray"),
                    edgecolor="black", lw=0.4)
            ax.text(a["end"], k, f" {a['staff']}명", va="center", fontsize=6)
        ax.set_yticks(range(len(a2)), [f"{a['order']}. {a['station']}" for a in a2], fontsize=6)
        ax.invert_yaxis()
        clock_xaxis(ax, min(a["start"] for a in a2), max(a["end"] for a in a2))
        ax.set_xlabel("시각", fontsize=8)
        ax.set_title("정비 인원 투입 일정 (막대 = 대응 기간, 숫자 = 정비 인원)", fontsize=9)
        _img(doc, fig, 15)

    # 5. 정비 인원별 지연 비교 · 적정 인원 (화면에는 없고 보고서에만 둔다)
    _bold(doc, "5. 정비 인원별 지연 비교 · 적정 인원")
    cv = sm["curve"]
    cur = int(sm["rules"]["total_staff"])
    if cv["rows"]:
        _table(doc, ["투입 가능 정비 인원(명)", "최적 배치 누적 지연 부품 수(개)", "기존 방식 누적 지연 부품 수(개)"],
               [[f"{x['total_staff']}" + (" (현재 설정)" if int(x["total_staff"]) == cur else ""),
                 f"{round(x['optimized_loss']):,}", f"{round(x['fifo_loss']):,}"] for x in cv["rows"]])
        fig, ax = plt.subplots(figsize=(6, 3))
        xs = [x["total_staff"] for x in cv["rows"]]
        ax.plot(xs, [x["optimized_loss"] for x in cv["rows"]], "o-", color="black", label="최적 배치")
        ax.plot(xs, [x["fifo_loss"] for x in cv["rows"]], "s--", color="gray", label="기존 방식(발생 순)")
        if cv.get("knee") is not None and not cv.get("knee_reason", "").startswith("비교한 범위 끝까지"):
            ax.axvline(cv["knee"], color="#e5484d", ls=":")
            ax.text(cv["knee"], ax.get_ylim()[1], " 적정 정비 인원", color="#e5484d", fontsize=8, va="top")
        if cur in xs:
            ax.axvline(cur, color="#1E3A8A", ls=":", lw=0.8)
            ax.text(cur, ax.get_ylim()[1], " 현재 설정", color="#1E3A8A", fontsize=8, va="top")
        ax.set_xlabel("투입 가능 정비 인원 (명)")
        ax.set_ylabel("누적 지연 부품 수 (개)")
        ax.legend(fontsize=7)
        _img(doc, fig, 11)
    doc.add_paragraph(sm.get("staffing_text", "") if sm.get("staffing_source") == "llm" else template_staffing(cv))
    doc.add_paragraph(f"비교한 인원: {', '.join(str(x) for x in sm['settings']['compare_staff'])}명 "
                      f"(현재 투입 가능 정비 인원 {cur}명). 다른 인원을 확인하려면 화면 왼쪽 '상세 설정'의 "
                      "'비교할 정비 인원'(예: 10, 20, 30, 40)이나 '투입 가능 정비 인원'을 바꾸고 "
                      "'설정 반영'을 누른 뒤 보고서를 다시 만듭니다.")

    # 6. 반복 정지 설비 (BM 개선 대상)
    _bold(doc, "6. 반복 정지 설비 (BM 개선 대상)")
    reps = _repeat_upto(events, tick)
    _table(doc, ["설비", "설비 정지 의심(회)", "병목(회)", "조치 권고(회)", "BM 개선 대상"],
           [[x["station"], x["stop"], x["bottleneck"], x["action"], "예" if x["review"] else ""] for x in reps], 20)
    targets = [x["station"] for x in reps if x["review"]]
    if targets:
        doc.add_paragraph("개선 검토 대상 " + ", ".join(targets[:10]) +
                          ": 예방보전 전환 검토, 점검 주기 재검토, 보전 인력 배치 조정을 검토합니다.")
    if sm["data_checks"]["unreliable_baseline_stations"] or sm["data_checks"]["reversed_transitions"]:
        doc.add_paragraph("데이터 수집 개선: 기록이 적어 정상 범위를 정하기 어려운 설비 "
                          f"{len(sm['data_checks']['unreliable_baseline_stations'])}개, "
                          f"순서가 뒤바뀐 이동 기록 {sm['data_checks']['reversed_transitions']}건.")

    # 7. 참고
    _bold(doc, "7. 신뢰도 점검")
    v = sm["validation"]
    if v.get("match_rate") is None:
        doc.add_paragraph("병목 판정 교차 검증(APM 일치율): 비교 구간 없음.")
    else:
        doc.add_paragraph(f"병목 판정 교차 검증(APM 일치율, 목표 90%): 비교 구간 {v.get('compared')}개 기준 1위 일치율 "
                          f"{v.get('match_rate')}%, APM 상위 3개 포함률 {v.get('top3_rate')}%. 통과 기록에는 막힘이 남지 않아 "
                          "APM은 공급이 끊기지 않는 앞쪽 설비를 1위로 뽑는 경향이 있어 목표에 미달합니다.")
    tr, df = ser["TR"], ser["DF"]
    qrows = [[stations[i], int(tr[i]), int(df[i]), round(df[i] / tr[i] * 100, 2)]
             for i in np.argsort(-(df / np.maximum(tr, 1))) if tr[i] >= 30][:10]
    if qrows:
        _bold(doc, "불량 참고 (판정에 사용 안 함, train 부품 기준)")
        _table(doc, ["설비", "통과 부품(개)", "불량(개)", "불량률(%)"], qrows)
    doc.add_paragraph("설정값과 데이터 기반 값을 구분해 표기했습니다.")

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _repeat_upto(events, tick):
    from pipeline import repeat_table
    return repeat_table(events, upto_tick=tick)

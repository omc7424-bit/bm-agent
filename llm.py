"""⑥ LLM 해석

- 탐지·최적화 결과 JSON을 1_prompt.md 시스템 프롬프트와 함께 LLM에 보내 문장만 받는다.
- 받은 문장에 입력 JSON에 없는 숫자가 있으면 폐기하고 '검토 필요'로 표시한 뒤 문장 틀을 쓴다.
- API 키가 없거나 호출이 실패하면 정해진 문장 틀로 쓴다 → 화면에 "LLM 미연결(문장 틀 사용)".
- API 키는 코드에 쓰지 않고 .env의 ANTHROPIC_API_KEY에서 읽는다.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

BASE_DIR = Path(__file__).resolve().parent
if load_dotenv:
    load_dotenv(BASE_DIR / ".env")

from stream import DEFAULT_MIN_PER_TICK, clock_note, clock_text, duration_text

# 프롬프트 문서 위치: 코드 폴더(4_구동파일) 옆의 3_마크다운 폴더. 코드 폴더에 따로 두면 그것을 먼저 쓴다.
PROMPT_PATH = next((p for p in (BASE_DIR / "1_prompt.md", BASE_DIR.parent / "3_마크다운" / "1_prompt.md") if p.exists()),
                   BASE_DIR.parent / "3_마크다운" / "1_prompt.md")
DEFAULT_MODEL = "claude-opus-5-5"
STATION_CODE = re.compile(r"L\d+_S\d+")
NUM = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")

LABEL_TEMPLATE = "LLM 미연결(문장 틀 사용)"


# ---------------------------------------------------------------------------
# 숫자 검증
# ---------------------------------------------------------------------------
def _collect_numbers(obj, out: set) -> None:
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        out.add(float(obj))
    elif isinstance(obj, str):
        for m in NUM.findall(STATION_CODE.sub(" ", obj)):
            out.add(float(m))
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect_numbers(v, out)


def invalid_numbers(text: str, payload: dict) -> list[str]:
    """문장 속 숫자 중 입력 JSON에 없는 것 (반올림 표기는 허용, 스테이션 코드는 제외)."""
    allowed: set = set()
    _collect_numbers(payload, allowed)
    bad = []
    for tok in NUM.findall(STATION_CODE.sub(" ", text)):
        v = float(tok)
        dec = len(tok.split(".")[1]) if "." in tok else 0
        if not any(abs(a - v) < 1e-9 or round(a, dec) == v for a in allowed):
            bad.append(tok)
    return bad


# ---------------------------------------------------------------------------
# 문장 틀 (API 키가 없을 때 / LLM 문장을 폐기했을 때)
# ---------------------------------------------------------------------------
def _when(ev: dict, mpt: float) -> str:
    """'05-18(일) 01:24 (원본 1202.94)부터 지속 5일 3시간 0분 (12.3)' — 표시용 환산."""
    return (f"{clock_text(ev['start_time'], mpt)} (원본 {ev['start_time']})부터 "
            f"지속 {duration_text(ev['duration'], mpt)} ({ev['duration']})")


# 화면·문장에 쓰는 이름 (판정 값 자체는 detector 그대로)
TYPE_LABEL = {"정지 의심": "설비 정지 의심", "병목": "병목"}
GRADE_LABEL = {"판단 보류": "확인 필요"}


def type_label(t: str) -> str:
    return TYPE_LABEL.get(t, t)


def grade_label(g: str) -> str:
    return GRADE_LABEL.get(g, g)


def template_alert(ev: dict, mpt: float = DEFAULT_MIN_PER_TICK) -> str:
    """ev: 1_prompt.md의 events 항목 + assignment(staff, order). 시각은 환산 시각과 원래 데이터 시각을 함께 쓴다."""
    a = ev.get("assignment") or {}
    head = f"[{grade_label(ev['grade'])}] {ev['station']} {type_label(ev['type'])}"
    if ev["grade"] == "판단 보류" or not ev.get("evidence"):
        return "\n".join([
            head,
            f"- 발생: {_when(ev, mpt)}",
            "- 판단: 기록 누락 가능 구간 - 확인 필요",
            f"- 근거: {ev.get('evidence') or '근거 없음 - 확인 필요'}",
            "- BM 대응: 정비 인원 투입 전 기록 확인 (담당자 승인 대기)",
        ])
    aff = "·".join(ev.get("affected_stations") or [])
    lines = [
        head,
        f"- 발생: {_when(ev, mpt)}",
        f"- 영향: 지연 부품 {ev['delayed_parts']}개" + (f", 후공정 {aff}로 확산" if aff else ", 후공정 영향 확인 안 됨"),
        f"- 근거: {ev['evidence']}",
        f"- 반복: 최근 {ev['repeat_count']}회째",
    ]
    if a:
        lines.append(f"- BM 대응: 정비 인원 {a['staff']}명 투입, 대응 순서 {a['order']}번, "
                     f"{clock_text(a['start'], mpt)} – {clock_text(a['end'], mpt)} (담당자 승인 대기)")
    else:
        lines.append("- BM 대응: 배정 대기 (담당자 승인 대기)")
    lines.append("- 현장 확인: " + ("정지 원인 확인 필요" if ev["type"] == "정지 의심" else "병목 원인 확인 필요"))
    return "\n".join(lines)


def template_summary(p: dict) -> str:
    """p: {events, optimization, repeat_top}"""
    opt = p.get("optimization") or {}
    asg = opt.get("assignments") or []
    s = []
    if asg:
        first = asg[0]
        s.append(f"지금 가장 먼저 정비 인원을 보낼 설비는 {first['station']}({grade_label(first['grade'])})이며, 대응 순서는 "
                 + " → ".join(a["station"] for a in asg[:5]) + "이다.")
    else:
        s.append("지금 정비 인원 투입이 필요한 설비는 없다.")
    if opt.get("baseline_loss") is not None:
        if opt.get("reduction_pct") is None:
            s.append("기존 방식(발생 순) 대응과 비교할 대응이 없다.")
        else:
            s.append(f"최적 배치의 누적 지연 부품 수는 {round(opt['total_loss']):,}개로, "
                     f"기존 방식(발생 순) {round(opt['baseline_loss']):,}개 대비 "
                     f"{opt['reduction_pct']}% 줄었다.")
    rep = p.get("repeat_top") or []
    if rep:
        s.append("반복 정지 설비는 " + ", ".join(f"{r['station']}(설비 정지 의심 {r.get('stop', r['count'])}회)" for r in rep[:3])
                 + "이며 BM 개선 대상이다.")
    else:
        s.append("반복 정지 설비는 아직 없다.")
    return " ".join(s)


def template_staffing(curve: dict) -> str:
    rows = curve.get("rows") or []
    if not rows:
        return "정비 인원별 비교 결과가 없다."
    parts = [f"{r['total_staff']}명 {round(r['optimized_loss']):,}개" for r in rows]
    t = "정비 인원별 누적 지연 부품 수는 " + ", ".join(parts) + "이다."
    if curve.get("knee") is None:
        t += f" {curve.get('knee_reason') or '비교할 이벤트 없음 - 적정 인원 판단 불가'}."
    elif curve.get("knee_reason", "").startswith("비교한 범위 끝까지"):
        t += (f" 비교한 범위({rows[0]['total_staff']}~{rows[-1]['total_staff']}명)에서는 정비 인원을 늘릴수록 지연이 계속 줄어"
              " 적정 정비 인원을 정할 수 없다. 더 큰 인원 값을 비교해야 한다.")
    elif curve.get("knee") is not None:
        t += f" 추가 인원 효과가 줄어드는 지점은 {curve['knee']}명이며, 이를 적정 정비 인원으로 본다."
        if curve.get("knee_reason"):
            t += f" ({curve['knee_reason']})"
    return t


# ---------------------------------------------------------------------------
class Interpreter:
    def __init__(self, max_calls: int | None = None):
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        self.model = os.getenv("LLM_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
        self.max_calls = int(os.getenv("LLM_MAX_CALLS", "30")) if max_calls is None else max_calls
        self.calls = 0
        self.errors: list[str] = []
        self.client = None
        if self.api_key and not self.api_key.startswith("여기에"):
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
            except Exception as e:      # 패키지 없음 등
                self.errors.append(f"LLM 초기화 실패: {e}")
        self.system_prompt = PROMPT_PATH.read_text(encoding="utf-8") if PROMPT_PATH.exists() else ""

    @property
    def connected(self) -> bool:
        return self.client is not None

    @property
    def label(self) -> str:
        return f"LLM 연결됨 ({self.model})" if self.connected else LABEL_TEMPLATE

    def _ask(self, instruction: str, payload: dict) -> str | None:
        if not self.connected or self.calls >= self.max_calls:
            return None
        import anthropic
        self.calls += 1
        try:
            resp = self.client.beta.messages.create(
                model=self.model,
                max_tokens=2000,
                system=self.system_prompt,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                output_config={"effort": "low"},
                messages=[{"role": "user", "content": instruction + "\n\n입력 JSON:\n"
                           + json.dumps(payload, ensure_ascii=False)}],
            )
        except anthropic.AuthenticationError:
            self.errors.append("API 키 인증 실패 → 문장 틀 사용")
            self.client = None
            return None
        except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.APIStatusError) as e:
            self.errors.append(f"LLM 호출 실패: {type(e).__name__}")
            return None
        except Exception as e:
            self.errors.append(f"LLM 호출 실패: {e}")
            return None
        if resp.stop_reason == "refusal":
            self.errors.append("LLM 응답 거절 → 문장 틀 사용")
            return None
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        return text or None

    def _checked(self, text: str | None, payload: dict, fallback: str) -> tuple[str, str]:
        """(문장, 출처). 출처: 'llm' / 'template' / 'template(검토 필요: LLM 문장 폐기)'."""
        if text is None:
            return fallback, "template"
        bad = invalid_numbers(text, payload)
        if bad:
            return "[검토 필요] 입력에 없는 숫자가 있어 LLM 문장을 폐기함\n" + fallback, \
                   f"template(검토 필요: LLM 문장 폐기, 숫자 {', '.join(bad[:5])})"
        return text, "llm"

    def alert(self, ev: dict, mpt: float = DEFAULT_MIN_PER_TICK) -> tuple[str, str]:
        e = {k: ev[k] for k in ("station", "type", "grade", "start_time", "duration", "delayed_parts",
                                "affected_stations", "repeat_count", "evidence") if k in ev}
        # 표시용 환산 시각은 코드가 미리 만들어 넣는다 (LLM이 환산 계산을 하지 않게)
        e["start_clock"] = clock_text(ev["start_time"], mpt)
        e["duration_text"] = duration_text(ev["duration"], mpt)
        a = ev.get("assignment")
        if a:
            a = {**a, "start_clock": clock_text(a["start"], mpt), "end_clock": clock_text(a["end"], mpt)}
        payload = {"timestamp": ev.get("timestamp"),
                   "timestamp_clock": clock_text(ev["timestamp"], mpt) if ev.get("timestamp") is not None else None,
                   "clock_note": clock_note(mpt), "events": [e], "assignment": a}
        text = self._ask("4-1 즉시 알림 형식으로 이 이벤트 1건의 알림만 써라. 다른 설명은 쓰지 마라.", payload)
        return self._checked(text, payload, template_alert(ev, mpt))

    def summary(self, p: dict) -> tuple[str, str]:
        text = self._ask("4-2 보고서 요약 문단(세 문장)만 써라. 다른 설명은 쓰지 마라.", p)
        return self._checked(text, p, template_summary(p))

    def staffing(self, curve: dict) -> tuple[str, str]:
        text = self._ask("4-3 인원 수별 비교 해설만 써라. 다른 설명은 쓰지 마라.", curve)
        return self._checked(text, curve, template_staffing(curve))

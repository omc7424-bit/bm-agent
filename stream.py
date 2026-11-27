"""① 데이터 수신·재생

- 원본 zip(train_date / test_date / train_numeric)을 압축을 풀지 않고 pandas로 chunk 단위로 읽는다.
- 처음 한 번만 long 형식(part_id, line, station, time_in, time_out, set)으로 바꿔 parquet로 저장한다.
  이후 실행에서는 parquet만 읽는다.
- 뒤 기간을 시간 순서대로 내보내는 재생기(Replayer)를 제공한다.
  실제 공장 적용 시에는 이 모듈만 MES 연동으로 교체하면 된다.

명령줄 실행:
    python stream.py                 # 변환 (이미 있으면 건너뜀)
    python stream.py --force         # 다시 변환
    python stream.py --chunksize 1000
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv
except ImportError:  # dotenv가 없어도 변환은 가능
    load_dotenv = None

BASE_DIR = Path(__file__).resolve().parent
if load_dotenv:
    load_dotenv(BASE_DIR / ".env")

DATA_DIR = Path(os.getenv("BM_DATA_DIR", str(BASE_DIR / "data")))
CACHE_DIR = Path(os.getenv("BM_CACHE_DIR", str(DATA_DIR)))

ZIP_TRAIN_DATE = DATA_DIR / "train_date.csv.zip"
ZIP_TEST_DATE = DATA_DIR / "test_date.csv.zip"
ZIP_TRAIN_NUMERIC = DATA_DIR / "train_numeric.csv.zip"

LONG_PARQUET = CACHE_DIR / "flow_long.parquet"
TRANS_PARQUET = CACHE_DIR / "flow_transitions.parquet"
RESP_PARQUET = CACHE_DIR / "response.parquet"
STATIONS_JSON = CACHE_DIR / "stations.json"
META_JSON = CACHE_DIR / "flow_meta.json"          # 변환 완료 표시 (마지막에 기록)
PROGRESS_JSON = CACHE_DIR / "convert_progress.json"
TMP_DIR = CACHE_DIR / "_tmp_convert"

SAMPLE_CSV = BASE_DIR / "bosch_sample_long.csv"

TICKS_PER_UNIT = 100          # 데이터 시간 해상도 0.01 → 1 tick
DEFAULT_CHUNKSIZE = 2000      # 원본 date 파일 1행 = 1,157열. 2,000행이면 float32로 약 9MB
BUCKET_UNITS = 50             # 시간 정렬용 임시 구간 폭 (데이터 시간 단위)
FLUSH_ROWS = 100_000          # 임시 구간 버퍼를 디스크로 내보내는 행 수

# Kaggle Bosch 데이터의 행 수 (진행률 표시용 추정값)
EST_ROWS = {"train": 1_183_747, "test": 1_183_748, "numeric": 1_183_747}

DEFAULT_MIN_PER_TICK = 6.0   # 환산 비율 기본값: 0.01 데이터 시간 단위 = 6분 (16.8 단위 = 1주)

# 날짜 맞추기: 데이터의 주간 휴무(공장 전체 기록 공백)는 16.8 단위 주기 중 약 7.6 지점에서 시작한다
# (모니터링 전 기간 116건 측정, 2026-11-20). 이 지점을 토요일 00:00으로, 기준 연도를 2024년 1월로 둔다.
WEEKEND_PHASE_UNITS = 7.6
ANCHOR_SATURDAY = datetime(2024, 1, 6, 0, 0)     # 토요일
CLOCK_BASE: datetime | None = None               # 화면에서 기준 일시를 바꾸면 여기에 넣는다 (None = 기본값)
WEEKDAY = "월화수목금토일"

COL_RE = re.compile(r"^L(\d+)_S(\d+)_D(\d+)$")


# ---------------------------------------------------------------------------
# 표시용 시각 환산 (계산에는 쓰지 않는다. 모든 계산은 원본 시간값으로 한다)
# ---------------------------------------------------------------------------
def default_clock_base(mpt: float = DEFAULT_MIN_PER_TICK) -> datetime:
    """원본 시간값 0에 해당하는 일시 (주간 휴무가 토요일 00:00에 시작하도록 맞춤)."""
    return ANCHOR_SATURDAY - timedelta(minutes=WEEKEND_PHASE_UNITS * TICKS_PER_UNIT * mpt)


def clock_base(mpt: float = DEFAULT_MIN_PER_TICK) -> datetime:
    return CLOCK_BASE or default_clock_base(mpt)


def to_datetime(t: float, mpt: float = DEFAULT_MIN_PER_TICK) -> datetime:
    return clock_base(mpt) + timedelta(minutes=int(round(float(t) * TICKS_PER_UNIT * mpt)))


def clock_note(mpt: float = DEFAULT_MIN_PER_TICK) -> str:
    return (f"시각은 익명화된 원본 시간값을 0.01 단위 = {mpt:g}분으로, 주간 휴무가 토·일요일이 되도록 날짜를 맞춰 환산한 값입니다 "
            f"(원본 0 = {clock_full(0, mpt)})")


def clock_text(t: float, mpt: float = DEFAULT_MIN_PER_TICK) -> str:
    """원본 시간값 → '05-18(일) 05:24'."""
    dt = to_datetime(t, mpt)
    return f"{dt:%m-%d}({WEEKDAY[dt.weekday()]}) {dt:%H:%M}"


def clock_full(t: float, mpt: float = DEFAULT_MIN_PER_TICK) -> str:
    """원본 시간값 → '2025-05-18(일) 05:24'."""
    dt = to_datetime(t, mpt)
    return f"{dt:%Y-%m-%d}({WEEKDAY[dt.weekday()]}) {dt:%H:%M}"


def duration_text(du: float, mpt: float = DEFAULT_MIN_PER_TICK) -> str:
    """데이터 시간 길이 → 'N일 H시간 M분'."""
    minutes = int(round(float(du) * TICKS_PER_UNIT * mpt))
    d, r = divmod(minutes, 1440)
    h, m = divmod(r, 60)
    parts = ([f"{d}일"] if d else []) + ([f"{h}시간"] if h or d else []) + [f"{m}분"]
    return " ".join(parts)


def clock_ticks(lo: float, hi: float, mpt: float = DEFAULT_MIN_PER_TICK, n: int = 7) -> tuple[list, list]:
    """그래프 가로축 눈금: 위치는 원래 데이터 시각, 글자는 환산 시각."""
    span_min = max((hi - lo) * TICKS_PER_UNIT * mpt, 1)
    steps = [1, 5, 10, 15, 30, 60, 120, 180, 360, 720, 1440, 2880, 4320, 10080, 20160, 43200]
    step = next((s for s in steps if span_min / s <= n), steps[-1])
    # 눈금이 실제 시각의 정각·자정에 오도록 기준 일시의 하루 중 분(分)만큼 맞춘다
    base = clock_base(mpt)
    off = base.hour * 60 + base.minute
    start_min = np.ceil((lo * TICKS_PER_UNIT * mpt + off) / step) * step - off
    vals_min = np.arange(start_min, hi * TICKS_PER_UNIT * mpt + 1e-9, step)
    vals = [float(v / (TICKS_PER_UNIT * mpt)) for v in vals_min]
    return vals, [clock_text(v, mpt).replace(" ", "<br>") for v in vals]
STATION_RE = re.compile(r"^L(\d+)_S(\d+)$")


# ---------------------------------------------------------------------------
# 공통 유틸
# ---------------------------------------------------------------------------
def station_key(code: str) -> tuple[int, int]:
    m = STATION_RE.match(code)
    return (int(m.group(1)), int(m.group(2))) if m else (99, 999)


def station_line(code: str) -> int:
    return station_key(code)[0]


def to_tick(t: np.ndarray) -> np.ndarray:
    """데이터 시각(0.01 단위) → 정수 tick."""
    return np.rint(np.asarray(t, dtype=np.float64) * TICKS_PER_UNIT).astype(np.int64)


def full_data_ready() -> bool:
    return META_JSON.exists() and LONG_PARQUET.exists() and TRANS_PARQUET.exists()


def zips_present() -> dict[str, bool]:
    return {p.name: p.exists() for p in (ZIP_TRAIN_DATE, ZIP_TEST_DATE, ZIP_TRAIN_NUMERIC)}


def read_progress() -> dict | None:
    try:
        return json.loads(PROGRESS_JSON.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_progress(**kw) -> None:
    kw["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        PROGRESS_JSON.write_text(json.dumps(kw, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# wide(원본) → long 변환
# ---------------------------------------------------------------------------
def _station_groups(columns) -> tuple[list[str], list[str], list[np.ndarray]]:
    """열 이름에서 스테이션별 열 위치를 묶는다."""
    date_cols = [c for c in columns if COL_RE.match(c)]
    by_station: dict[str, list[int]] = {}
    for i, c in enumerate(date_cols):
        m = COL_RE.match(c)
        by_station.setdefault(f"L{m.group(1)}_S{m.group(2)}", []).append(i)
    stations = sorted(by_station, key=station_key)
    groups = [np.array(by_station[s], dtype=np.int64) for s in stations]
    return date_cols, stations, groups


def wide_chunk_to_long(ids: np.ndarray, arr: np.ndarray, groups: list[np.ndarray]):
    """원본 chunk(부품 × 측정열) → (부품, 스테이션, 진입, 이탈) 배열.

    스테이션별 측정 시각의 최솟값 = 진입 시각, 최댓값 = 이탈 시각.
    """
    P, S, TI, TO = [], [], [], []
    for si, idx in enumerate(groups):
        sub = arr[:, idx]
        tin = np.fmin.reduce(sub, axis=1)    # NaN 무시 최솟값 (모두 NaN이면 NaN)
        mask = ~np.isnan(tin)
        if not mask.any():
            continue
        tout = np.fmax.reduce(sub, axis=1)
        P.append(ids[mask])
        S.append(np.full(mask.sum(), si, dtype=np.int16))
        TI.append(tin[mask])
        TO.append(tout[mask])
    if not P:
        e = np.array([], dtype=np.float32)
        return (np.array([], np.int32), np.array([], np.int16), e, e)
    return (np.concatenate(P).astype(np.int32), np.concatenate(S),
            np.concatenate(TI).astype(np.float32), np.concatenate(TO).astype(np.float32))


def make_transitions(P, S, TI, TO):
    """부품별 통과 순서대로 (앞 스테이션 → 다음 스테이션) 이동 기록을 만든다."""
    if len(P) < 2:
        e = np.array([], dtype=np.float32)
        return (np.array([], np.int32), np.array([], np.int16), np.array([], np.int16), e, e)
    order = np.lexsort((S, TI, P))
    P, S, TI, TO = P[order], S[order], TI[order], TO[order]
    same = P[1:] == P[:-1]
    return (P[1:][same], S[:-1][same], S[1:][same], TO[:-1][same], TI[1:][same])


class _BucketWriter:
    """시간 구간별 임시 parquet. 마지막에 구간 순서대로 정렬해 하나로 합친다."""

    def __init__(self, folder: Path, width_units: float):
        import pyarrow as pa
        import pyarrow.parquet as pq
        self.pa, self.pq = pa, pq
        self.folder = folder
        self.width = width_units
        self.buffers: dict[int, list] = {}
        self.counts: dict[int, int] = {}
        self.writers: dict[int, object] = {}
        self.schema = pa.schema([("part_id", pa.int32()), ("station_idx", pa.int16()),
                                 ("time_in", pa.float32()), ("time_out", pa.float32()),
                                 ("is_test", pa.int8())])

    def add(self, P, S, TI, TO, is_test: int):
        b = np.floor(TI / self.width).astype(np.int64)
        for key in np.unique(b):
            m = b == key
            self.buffers.setdefault(int(key), []).append(
                (P[m], S[m], TI[m], TO[m], np.full(m.sum(), is_test, np.int8)))
            self.counts[int(key)] = self.counts.get(int(key), 0) + int(m.sum())
            if self.counts[int(key)] >= FLUSH_ROWS:
                self._flush(int(key))

    def _flush(self, key: int):
        parts = self.buffers.pop(key, [])
        self.counts[key] = 0
        if not parts:
            return
        cols = [np.concatenate([p[i] for p in parts]) for i in range(5)]
        table = self.pa.Table.from_arrays([self.pa.array(c) for c in cols], schema=self.schema)
        if key not in self.writers:
            self.writers[key] = self.pq.ParquetWriter(self.folder / f"bucket_{key:05d}.parquet", self.schema)
        self.writers[key].write_table(table)

    def close(self) -> list[Path]:
        for key in list(self.buffers):
            self._flush(key)
        for w in self.writers.values():
            w.close()
        return [self.folder / f"bucket_{k:05d}.parquet" for k in sorted(self.writers)]


def convert(chunksize: int = DEFAULT_CHUNKSIZE, force: bool = False, log=print) -> dict:
    """원본 zip → parquet 변환. 이미 끝났으면 아무것도 하지 않는다."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if full_data_ready() and not force:
        log("변환된 parquet가 이미 있습니다. (다시 하려면 --force)")
        return json.loads(META_JSON.read_text(encoding="utf-8"))

    missing = [k for k, v in zips_present().items() if not v]
    if missing:
        raise FileNotFoundError(f"data 폴더에 zip 파일이 없습니다: {', '.join(missing)}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    TMP_DIR.mkdir(parents=True)
    for p in (META_JSON,):
        if p.exists():
            p.unlink()

    started = time.time()
    header = pd.read_csv(ZIP_TRAIN_DATE, nrows=0).columns
    date_cols, stations, groups = _station_groups(header)
    log(f"스테이션 {len(stations)}개, 측정 시각 열 {len(date_cols)}개")

    bucket = _BucketWriter(TMP_DIR, BUCKET_UNITS)
    trans_schema = pa.schema([("part_id", pa.int32()), ("from_idx", pa.int16()), ("to_idx", pa.int16()),
                              ("t_from_out", pa.float32()), ("t_to_in", pa.float32())])
    trans_tmp = TMP_DIR / "transitions.parquet"
    trans_writer = pq.ParquetWriter(trans_tmp, trans_schema)
    trans_buf: list = []
    trans_rows = 0

    def flush_trans():
        nonlocal trans_buf
        if trans_buf:
            cols = [np.concatenate([b[i] for b in trans_buf]) for i in range(5)]
            trans_writer.write_table(pa.Table.from_arrays([pa.array(c) for c in cols], schema=trans_schema))
            trans_buf = []

    n_parts = 0
    n_rows = 0
    t_min, t_max = np.inf, -np.inf
    dtypes = {c: np.float32 for c in date_cols}
    dtypes["Id"] = np.int32
    usecols = ["Id"] + date_cols

    for set_name, path, is_test in (("train", ZIP_TRAIN_DATE, 0), ("test", ZIP_TEST_DATE, 1)):
        cols = pd.read_csv(path, nrows=0).columns
        if list(c for c in cols if COL_RE.match(c)) != date_cols:
            raise ValueError(f"{path.name}의 열 구성이 train_date와 다릅니다.")
        done = 0
        reader = pd.read_csv(path, usecols=usecols, dtype=dtypes, chunksize=chunksize)
        for chunk in reader:
            ids = chunk["Id"].to_numpy(np.int32)
            arr = chunk[date_cols].to_numpy(np.float32)
            del chunk
            P, S, TI, TO = wide_chunk_to_long(ids, arr, groups)
            del arr
            if len(P):
                bucket.add(P, S, TI, TO, is_test)
                tr = make_transitions(P, S, TI, TO)
                trans_buf.append(tr)
                trans_rows += len(tr[0])
                if sum(len(b[0]) for b in trans_buf) >= FLUSH_ROWS * 5:
                    flush_trans()
                t_min = min(t_min, float(TI.min()))
                t_max = max(t_max, float(TO.max()))
                n_parts += len(np.unique(P))
                n_rows += len(P)
            done += len(ids)
            pct = min(done / EST_ROWS[set_name], 1.0)
            _write_progress(stage=f"{set_name}_date 읽기", percent=round(pct * 100, 1),
                            rows_done=done, elapsed_sec=round(time.time() - started),
                            done=False, error=None)
        log(f"{set_name}_date 완료: 누적 기록 {n_rows:,}행")
    flush_trans()
    trans_writer.close()

    # 시간 구간별로 정렬해 하나의 parquet로 합친다 (메모리에는 한 구간씩만 올림)
    _write_progress(stage="시간순 정렬·저장", percent=0, done=False, error=None,
                    elapsed_sec=round(time.time() - started))
    files = bucket.close()
    station_dict = pa.array(stations, type=pa.string())
    set_dict = pa.array(["train", "test"], type=pa.string())
    long_schema = pa.schema([
        ("part_id", pa.int32()), ("line", pa.int8()),
        ("station", pa.dictionary(pa.int16(), pa.string())), ("station_idx", pa.int16()),
        ("time_in", pa.float32()), ("time_out", pa.float32()),
        ("set", pa.dictionary(pa.int8(), pa.string())),
    ])
    line_of = np.array([station_line(s) for s in stations], dtype=np.int8)
    long_tmp = TMP_DIR / "flow_long.parquet"
    writer = pq.ParquetWriter(long_tmp, long_schema)
    for i, f in enumerate(files):
        t = pq.read_table(f)
        P = t["part_id"].to_numpy()
        S = t["station_idx"].to_numpy()
        TI = t["time_in"].to_numpy()
        TO = t["time_out"].to_numpy()
        T = t["is_test"].to_numpy()
        del t
        o = np.lexsort((S, P, TI))
        P, S, TI, TO, T = P[o], S[o], TI[o], TO[o], T[o]
        table = pa.Table.from_arrays([
            pa.array(P), pa.array(line_of[S]),
            pa.DictionaryArray.from_arrays(pa.array(S), station_dict), pa.array(S),
            pa.array(TI), pa.array(TO),
            pa.DictionaryArray.from_arrays(pa.array(T), set_dict),
        ], schema=long_schema)
        writer.write_table(table, row_group_size=500_000)
        f.unlink()
        _write_progress(stage="시간순 정렬·저장", percent=round((i + 1) / len(files) * 100, 1),
                        done=False, error=None, elapsed_sec=round(time.time() - started))
    writer.close()

    # 품질 판정: train_numeric에서 Id, Response 두 열만 읽는다
    resp_parts = []
    done = 0
    reader = pd.read_csv(ZIP_TRAIN_NUMERIC, usecols=["Id", "Response"],
                         dtype={"Id": np.int32, "Response": np.int8}, chunksize=chunksize * 10)
    for chunk in reader:
        resp_parts.append(chunk)
        done += len(chunk)
        _write_progress(stage="train_numeric(Response) 읽기",
                        percent=round(min(done / EST_ROWS["numeric"], 1) * 100, 1),
                        done=False, error=None, elapsed_sec=round(time.time() - started))
    resp = pd.concat(resp_parts, ignore_index=True)
    resp_tmp = TMP_DIR / "response.parquet"
    resp.to_parquet(resp_tmp, index=False)

    # 완성 파일로 옮기고, 마지막에 meta를 써서 '변환 완료'를 표시한다
    shutil.move(str(long_tmp), str(LONG_PARQUET))
    shutil.move(str(trans_tmp), str(TRANS_PARQUET))
    shutil.move(str(resp_tmp), str(RESP_PARQUET))
    STATIONS_JSON.write_text(json.dumps(stations), encoding="utf-8")
    meta = {
        "stations": len(stations), "records": n_rows, "parts": n_parts,
        "transitions": trans_rows, "t_min": round(t_min, 2), "t_max": round(t_max, 2),
        "defect_rate": round(float(resp["Response"].mean()), 5),
        "chunksize": chunksize, "elapsed_sec": round(time.time() - started),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    META_JSON.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    _write_progress(stage="완료", percent=100, done=True, error=None, elapsed_sec=meta["elapsed_sec"])
    log(f"변환 완료: {meta}")
    return meta


# ---------------------------------------------------------------------------
# 분석용 데이터 읽기
# ---------------------------------------------------------------------------
class FlowSource:
    """분석 모듈이 읽는 통과 기록 묶음. 샘플 CSV와 전체 parquet를 같은 방식으로 다룬다.

    long_batches()  → (part_id, station_idx, time_in, is_test) 배열 묶음
    trans_batches() → (from_idx, to_idx, t_from_out, t_to_in) 배열 묶음
    response_lookup → part_id 위치에 0/1 (-1은 판정 없음)
    """

    def __init__(self, kind: str, stations: list[str], t_min: float, t_max: float,
                 long_batches, trans_batches, response_lookup: np.ndarray, info: dict):
        self.kind = kind
        self.stations = stations
        self.t_min = t_min
        self.t_max = t_max
        self.long_batches = long_batches
        self.trans_batches = trans_batches
        self.response_lookup = response_lookup
        self.info = info

    @property
    def cache_key(self) -> str:
        return f"{self.kind}:{self.info.get('created', '')}:{self.info.get('records', '')}"


def load_sample(path: Path = SAMPLE_CSV) -> FlowSource:
    """샘플 CSV(part_id, station, time_in, set, response). time_out이 없어 time_in과 같게 둔다."""
    df = pd.read_csv(path)
    stations = sorted(df["station"].unique(), key=station_key)
    sidx = {s: i for i, s in enumerate(stations)}
    P = df["part_id"].to_numpy(np.int32)
    S = df["station"].map(sidx).to_numpy(np.int16)
    TI = df["time_in"].to_numpy(np.float32)
    TO = (df["time_out"] if "time_out" in df else df["time_in"]).to_numpy(np.float32)
    is_test = (df["set"] == "test").to_numpy(np.int8)
    tr = make_transitions(P, S, TI, TO)

    lookup = np.full(int(P.max()) + 1, -1, dtype=np.int8)
    r = df.dropna(subset=["response"]).drop_duplicates("part_id")
    lookup[r["part_id"].to_numpy(np.int64)] = r["response"].to_numpy(np.int8)

    info = {"records": len(df), "parts": int(df["part_id"].nunique()), "transitions": len(tr[0]),
            "t_min": float(TI.min()), "t_max": float(TO.max()), "created": "sample",
            "defect_rate": float(r["response"].mean()) if len(r) else None}
    return FlowSource("sample", stations, float(TI.min()), float(TO.max()),
                      lambda: iter([(P, S, TI, is_test)]),
                      lambda: iter([(tr[1], tr[2], tr[3], tr[4])]),
                      lookup, info)


def load_full(batch_rows: int = 1_000_000) -> FlowSource:
    """변환된 parquet를 batch 단위로 읽는다 (전체를 한 번에 올리지 않음)."""
    import pyarrow.parquet as pq

    if not full_data_ready():
        raise FileNotFoundError("변환된 parquet가 없습니다. 먼저 `python stream.py`를 실행하세요.")
    meta = json.loads(META_JSON.read_text(encoding="utf-8"))
    stations = json.loads(STATIONS_JSON.read_text(encoding="utf-8"))

    def long_batches():
        f = pq.ParquetFile(LONG_PARQUET)
        for b in f.iter_batches(batch_size=batch_rows, columns=["part_id", "station_idx", "time_in", "set"]):
            st = b.column("set")
            is_test = (st.indices.to_numpy() == 1).astype(np.int8) if hasattr(st, "indices") \
                else (np.asarray(st.to_pylist()) == "test").astype(np.int8)
            yield (b.column("part_id").to_numpy(), b.column("station_idx").to_numpy(),
                   b.column("time_in").to_numpy(), is_test)

    def trans_batches():
        f = pq.ParquetFile(TRANS_PARQUET)
        for b in f.iter_batches(batch_size=batch_rows):
            yield (b.column("from_idx").to_numpy(), b.column("to_idx").to_numpy(),
                   b.column("t_from_out").to_numpy(), b.column("t_to_in").to_numpy())

    lookup = np.full(1, -1, dtype=np.int8)
    if RESP_PARQUET.exists():
        r = pd.read_parquet(RESP_PARQUET)
        lookup = np.full(int(r["Id"].max()) + 1, -1, dtype=np.int8)
        lookup[r["Id"].to_numpy(np.int64)] = r["Response"].to_numpy(np.int8)
    return FlowSource("full", stations, meta["t_min"], meta["t_max"],
                      long_batches, trans_batches, lookup, meta)


# ---------------------------------------------------------------------------
# 실시간 재생
# ---------------------------------------------------------------------------
class Replayer:
    """뒤 기간을 tick(0.01 데이터 시간 단위) 순서대로 내보낸다.

    재생 속도 1배 = 화면 갱신 1회당 BASE_TICKS_PER_STEP tick.
    MES 연동 시에는 advance()가 '새로 들어온 기록까지의 시각'을 돌려주도록 바꾸면 된다.
    """

    BASE_TICKS_PER_STEP = 5   # 1배 = 갱신 1회당 0.05 데이터 시간 단위

    def __init__(self, start_bin: int, end_bin: int):
        self.start_bin = start_bin
        self.end_bin = end_bin
        self.current = start_bin - 1   # 아직 아무 tick도 처리하지 않음

    @property
    def finished(self) -> bool:
        return self.current >= self.end_bin - 1

    @property
    def progress(self) -> float:
        total = max(self.end_bin - self.start_bin, 1)
        return min(max((self.current - self.start_bin + 1) / total, 0.0), 1.0)

    def advance(self, speed: float) -> int:
        """다음에 처리할 마지막 bin 번호를 돌려준다."""
        step = max(int(round(self.BASE_TICKS_PER_STEP * speed)), 1)
        self.current = min(self.current + step, self.end_bin - 1)
        return self.current

    def to_end(self) -> int:
        self.current = self.end_bin - 1
        return self.current


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Bosch zip → parquet 변환")
    ap.add_argument("--chunksize", type=int, default=DEFAULT_CHUNKSIZE,
                    help=f"원본을 한 번에 읽는 행 수 (기본 {DEFAULT_CHUNKSIZE}, 메모리가 부족하면 더 작게)")
    ap.add_argument("--force", action="store_true", help="이미 변환돼 있어도 다시 변환")
    args = ap.parse_args(argv)
    try:
        convert(chunksize=args.chunksize, force=args.force)
    except Exception as e:
        _write_progress(stage="오류", percent=0, done=False, error=f"{type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

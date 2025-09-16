
import os, time, json, re
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Body
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

# ------------------------------
# Load environment
# ------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DB_URL = os.getenv("DB_URL", "mysql+pymysql://user:pass@localhost:3306/venus_ai?charset=utf8mb4")
ALLOWED_SCHEMAS = set([s.strip() for s in os.getenv("ALLOWED_SCHEMAS", "venus_ai").split(",") if s.strip()])
ALLOWED_TABLES = set([t.strip() for t in os.getenv("ALLOWED_TABLES", "analytics_shipments,analytics_shipments_with_gc,analytics_mv_monthly_therapy_supplier").split(",") if t.strip()])
MAX_ROWS = int(os.getenv("MAX_ROWS", "5000"))
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "900"))

# ------------------------------
# OpenAI + DB engine
# ------------------------------
client = OpenAI(api_key=OPENAI_API_KEY)
engine: Engine = create_engine(DB_URL, pool_pre_ping=True)

app = FastAPI(title="Athena Orchestrator (MySQL + Python)")

# ------------------------------
# Simple in-memory session store
# ------------------------------
RUNNING_SUMMARIES: Dict[str, str] = {}  # session_id -> compact summary

# ------------------------------
# Schema Card & Rulebook (edit these to match your DB)
# ------------------------------
SCHEMA_CARD = """
You are using **MySQL 8**. Use functions YEAR(date), MONTH(date), DATE_FORMAT(date, '%Y-%m-01').
Tables/views you MAY query by default:
- analytics_shipments(date, year, month, therapy, product_name, skus, dosage_form, uom, quantity, value_inr, price_inr_per_unit, indian_company, supplier, country, city, continent)
- analytics_shipments_with_gc(same fields + cost_inr_per_unit, cost_inr_total, gc_inr, gc_pct)
- analytics_mv_monthly_therapy_supplier(month_date, therapy, supplier, country, dosage_form, qty, value_inr, median_price_inr)

Never query raw tables. Stick to the views above. Always add LIMIT {MAX_ROWS} when previewing rows.
"""

RULEBOOK = {
  "time": {
    "default_period": "L12M",
    "presets": ["YTD", "L24M", "LAST_CAL_YEAR", "FULL_8Y"],
    "anchor": "max_data_date",
    "use_mv_if_months_gt": 24
  },
  "metrics": {
    "leader_metric_default": "SUM(value_inr)",
    "leader_metric_alternates": ["SUM(quantity)", "MEDIAN(price_inr_per_unit)"],
    "top_n_default": 5
  },
  "price": {
    "unit_price_field": "price_inr_per_unit",
    "bands": ["P25","P50","P75"]
  },
  "gc": {
    "compute_gc": False,  # set True when you create analytics_shipments_with_gc
    "gc_view": "analytics_shipments_with_gc"
  },
  "safety": {
    "sql_must_be_select_only": True,
    "enforce_limit": True,
    "max_rows": MAX_ROWS
  }
}

# ------------------------------
# Tool specs (model can call run_sql)
# ------------------------------
RUN_SQL_TOOL = {
  "type": "function",
  "function": {
    "name": "run_sql",
    "description": "Execute a safe, read-only MySQL SELECT on analytics_* views. Always include LIMIT when previewing.",
    "parameters": {
      "type": "object",
      "properties": {
        "query": { "type": "string", "description": "MySQL 8 SELECT query only. Use allowed views; include LIMIT when needed." }
      },
      "required": ["query"]
    }
  }
}

# ------------------------------
# SQL safety checks
# ------------------------------
DANGEROUS = re.compile(r"\\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|GRANT|REVOKE|CREATE|MERGE)\\b", re.I)

TABLE_REGEXES = [
    re.compile(r"\\bFROM\\s+([`\"\\w\\.]+)", re.I),
    re.compile(r"\\bJOIN\\s+([`\"\\w\\.]+)", re.I)
]

def is_safe_select(sql: str) -> bool:
    s = sql.strip().rstrip(";")
    if not s.lower().startswith("select"):
        return False
    if DANGEROUS.search(s):
        return False
    return True

def referenced_tables(sql: str) -> List[str]:
    s = sql
    found = []
    for rgx in TABLE_REGEXES:
        for m in rgx.finditer(s):
            tbl = m.group(1).strip("`\" ")
            # If qualified like db.table, keep only table part for allow check
            tbl = tbl.split(".")[-1]
            found.append(tbl)
    return list({t for t in found})

def enforce_allowlist(sql: str) -> None:
    # In MySQL we typically use database.table; we keep a per-table allowlist here.
    tables = referenced_tables(sql)
    for t in tables:
        if t not in ALLOWED_TABLES and not t.startswith("analytics_"):
            raise ValueError(f"Table '{t}' not allowed. Allowed: {sorted(ALLOWED_TABLES)} or names starting with analytics_.")

def ensure_limit(sql: str) -> str:
    s = sql.strip().rstrip(";")
    if re.search(r"\\bLIMIT\\b", s, re.I):
        return s
    # Only append LIMIT if the query is not an aggregate-only with GROUP BY? Keep it simple: append always.
    return f"{s} LIMIT {MAX_ROWS}"

# ------------------------------
# Minimal cache (query signature -> rows)
# ------------------------------
from time import time as now
CACHE: Dict[str, Dict[str, Any]] = {}

def cache_get(key: str) -> Optional[Any]:
    v = CACHE.get(key)
    if not v:
        return None
    if now() - v["t"] > CACHE_TTL_SECONDS:
        CACHE.pop(key, None)
        return None
    return v["v"]

def cache_set(key: str, value: Any) -> None:
    CACHE[key] = {"t": now(), "v": value}

# ------------------------------
# Pydantic models
# ------------------------------
class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Unique id per chat thread")
    message: str = Field(..., description="User's question in plain English")

class ChatResponse(BaseModel):
    answer: str
    data_preview: Dict[str, Any]
    sql: Optional[str]
    latency_ms: int

# ------------------------------
# Helper: run SQL safely and return small result
# ------------------------------
def run_sql_query(sql: str) -> Dict[str, Any]:
    if not is_safe_select(sql):
        raise ValueError("Only SELECT statements are allowed.")
    enforce_allowlist(sql)
    if RULEBOOK["safety"]["enforce_limit"]:
        sql = ensure_limit(sql)

    sig = f"v1::{sql}"
    cached = cache_get(sig)
    if cached is not None:
        return cached

    start = time.time()
    rows: List[Dict[str, Any]] = []
    cols: List[str] = []
    try:
        with engine.connect() as conn:
            res = conn.execute(text(sql))
            cols = list(res.keys())
            for i, r in enumerate(res):
                if i >= MAX_ROWS:
                    break
                rows.append(dict(r._mapping))
    except SQLAlchemyError as e:
        raise ValueError(f"DB error: {e}") from e

    elapsed = int((time.time() - start) * 1000)
    payload = {"columns": cols, "rows": rows, "elapsed_ms": elapsed}
    cache_set(sig, payload)
    return payload

# ------------------------------
# Core chat flow
# ------------------------------
SYSTEM_PROMPT = f"""
You are Athena, a cautious, data-grounded analyst for Indian import/export (pharma). 
- Use **MySQL 8** syntax. 
- Prefer querying **analytics_shipments**; for GC questions, use **{RULEBOOK['gc']['gc_view']}** if RULEBOOK.gc.compute_gc is true.
- ALWAYS explain the data window and filters used. 
- NEVER guess GC if RULEBOOK.gc.compute_gc is false; use price bands instead and label as 'GC proxy' if asked.
- When previewing, include LIMIT.
{SCHEMA_CARD}
RULES: {json.dumps(RULEBOOK)}
"""

def model_decide_and_answer(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """Call OpenAI with tool-calling; if it calls run_sql, execute and send back result for final answer."""
    # 1) Ask model how to proceed
    first = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        tools=[RUN_SQL_TOOL],
        tool_choice="auto",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            *messages
        ]
    )

    msg = first.choices[0].message
    tool_calls = getattr(msg, "tool_calls", None)

    data_preview = {"columns": [], "rows": []}
    sql_used = None

    if tool_calls:
        # Execute only the first run_sql call for simplicity
        for tc in tool_calls:
            if tc.type == "function" and tc.function.name == "run_sql":
                args = json.loads(tc.function.arguments or "{}")
                raw_sql = args.get("query", "")
                data_preview = run_sql_query(raw_sql)
                sql_used = ensure_limit(raw_sql)
                break

        # 2) Send tool result back for final narrative
        second = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            tools=[RUN_SQL_TOOL],
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *messages,
                {
                    "role": "tool",
                    "tool_call_id": tool_calls[0].id,
                    "name": "run_sql",
                    "content": json.dumps(data_preview)
                }
            ]
        )
        final_text = second.choices[0].message.content or ""
    else:
        # No tool call; answer directly
        final_text = msg.content or ""

    return {"answer": final_text, "data_preview": data_preview, "sql": sql_used}

# ------------------------------
# API
# ------------------------------
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest = Body(...)):
    # Build conversation context: running summary + last user message
    summary = RUNNING_SUMMARIES.get(req.session_id, "")
    context: List[Dict[str, str]] = []
    if summary:
        context.append({"role": "system", "content": f"Running summary so far: {summary}"})
    context.append({"role": "user", "content": req.message})

    t0 = time.time()
    result = model_decide_and_answer(context)
    latency_ms = int((time.time() - t0) * 1000)

    # Naive summary update (keep it tiny)
    clipped = (req.message[:120] + "...") if len(req.message) > 120 else req.message
    RUNNING_SUMMARIES[req.session_id] = f"{summary} | {clipped}"[:1000]

    return ChatResponse(
        answer=result["answer"],
        data_preview=result["data_preview"],
        sql=result["sql"],
        latency_ms=latency_ms
    )

# ------------------------------
# Health
# ------------------------------
@app.get("/healthz")
def health():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False
    return {"ok": True, "db": db_ok}
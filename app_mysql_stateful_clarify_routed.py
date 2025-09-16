import os, time, json, re
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Body
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from decimal import Decimal

# ------------------------------
# Load environment
# ------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DB_URL = os.getenv("DB_URL", "mysql+pymysql://athena_ro:pass@localhost:3306/venus_ai?charset=utf8mb4")
ALLOWED_TABLES = set([t.strip() for t in os.getenv("ALLOWED_TABLES", "analytics_shipments,analytics_shipments_with_gc,analytics_mv_monthly_therapy_supplier").split(",") if t.strip()])
MAX_ROWS = int(os.getenv("MAX_ROWS", "5000"))
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "900"))
AUTO_CREATE_STATE_SCHEMA = os.getenv("AUTO_CREATE_STATE_SCHEMA", "0") == "1"

# Model routing (env overrides)
MODEL_PRIMARY  = os.getenv("OPENAI_MODEL", "gpt-5")          # complex/default
MODEL_SIMPLE   = os.getenv("OPENAI_MODEL_SIMPLE", "gpt-4o-mini")
MODEL_SUMMARY  = os.getenv("OPENAI_MODEL_SUMMARY", MODEL_SIMPLE)

# ------------------------------
# OpenAI + DB engine
# ------------------------------
client = OpenAI(api_key=OPENAI_API_KEY)
engine: Engine = create_engine(DB_URL, pool_pre_ping=True)

app = FastAPI(title="Athena Orchestrator (MySQL + Python, Stateful + Clarify + Routing)")

# ------------------------------
# RULEBOOK & SCHEMA CARD (same as previous clarify build)
# ------------------------------
RULEBOOK = {
  "time": {
    "default_period": "L12M",
    "presets": ["YTD", "L24M", "LAST_CAL_YEAR", "FULL_8Y"],
    "anchor": "max_data_date",
    "use_mv_if_months_gt": 24,
    "mv_table": "analytics_mv_monthly_therapy_supplier",
    "mv_fields": { "month": "month_date", "value": "value_inr", "qty": "qty" }
  },
  "entities": {
    "primary_competitor_dim": "supplier",
    "alternate_dims": ["indian_company", "country", "dosage_form", "therapy", "product_name", "city"],
    "therapy_field": "therapy",
    "therapy_handling": "case_insensitive_trim",
    "unknown_therapy_label": "Unknown"
  },
  "metrics": {
    "leader_metric_default": "SUM(value_inr)",
    "leader_metric_alternates": ["SUM(quantity)", "AVG(price_inr_per_unit)"],
    "share_formula": "group_metric / total_metric",
    "top_n_default": 5,
    "tie_breakers": ["SUM(value_inr) DESC", "SUM(quantity) DESC", "name ASC"]
  },
  "price": {
    "unit_price_field": "price_inr_per_unit",
    "bands": ["P25","P50","P75"],
    "winsorize_percentiles": [0.01, 0.99],
    "rounding": { "price_decimals": 2, "large_value_format": "INR_lakh_crore" }
  },
  "growth": {
    "yoy_percent": "(v2 - v1) / NULLIF(v1,0)",
    "cagr_percent": "POWER(v_end / NULLIF(v_start,0), 1.0/years) - 1.0",
    "seasonality_flag": "month_value > 1.5 * ma6_value"
  },
  "gc": {
    "compute_gc": False,
    "gc_view": "analytics_shipments_with_gc",
    "unit_cost_field": "cost_inr_per_unit",
    "formula": {
      "total_cost": "cost_inr_per_unit * quantity",
      "gc_inr": "value_inr - (cost_inr_per_unit * quantity)",
      "gc_pct": "CASE WHEN value_inr>0 THEN gc_inr/value_inr ELSE NULL END"
    },
    "proxy_when_no_cost": "price_premium_vs_market_median",
    "label_proxy_as": "GC proxy (price premium)"
  },
  "data_quality": {
    "row_validity_filters": ["quantity > 0", "value_inr > 0"],
    "min_samples_for_pricing": 5,
    "null_handling": "exclude_from_stats_but_keep_in_totals_if_value_present"
  },
  "explainability": {
    "always_include": ["data_window", "filters", "metric_basis", "how_computed"],
    "expose_sql": True,
    "show_counts": ["row_count", "latency_ms"]
  },
  "currency": {
    "base": "INR",
    "fx_conversion": { "enabled": False, "table": "analytics_fx_rates", "target": "USD", "fallback": "skip" }
  },
  "safety": {
    "dialect": "mysql8",
    "sql_must_be_select_only": True,
    "enforce_limit": True,
    "max_rows": MAX_ROWS
  },
  "clarify": {
    "ask_if_blocking": True,
    "max_questions_per_turn": 1,
    "preferred_slots": ["time_window", "metric", "scope", "uom", "dosage_form"],
    "defaults": { "time_window": "L12M", "metric": "SUM(value_inr)", "scope": "all_data" }
  }
}

SCHEMA_CARD = f"""
You are using **MySQL 8**. Use functions YEAR(date), MONTH(date), and DATE_FORMAT(date, '%Y-%m-01').
IMPORTANT: MySQL does NOT have MEDIAN() function. For percentiles, use:
- PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY column) OVER() for median
- For price bands, use NTILE(4) OVER (ORDER BY price) to create quartiles
- Or calculate manually with subqueries
CRITICAL: When using GROUP BY, ALL non-aggregated columns in SELECT must be in GROUP BY clause.

GROWTH RATE CALCULATIONS:
- For simple growth rates, use: (current_value - previous_value) / previous_value * 100
- For YoY growth, compare same month in different years
- For month-over-month growth, use CTEs (WITH statements) to calculate properly:
  WITH monthly_data AS (
    SELECT supplier, YEAR(date) AS year, MONTH(date) AS month, SUM(value_inr) AS total_value
    FROM analytics_shipments 
    WHERE [conditions]
    GROUP BY supplier, YEAR(date), MONTH(date)
  )
  SELECT supplier, year, month, total_value,
         LAG(total_value) OVER (PARTITION BY supplier ORDER BY year, month) AS prev_value,
         (total_value - LAG(total_value) OVER (PARTITION BY supplier ORDER BY year, month)) / 
         NULLIF(LAG(total_value) OVER (PARTITION BY supplier ORDER BY year, month), 0) * 100 AS growth_rate
  FROM monthly_data
- NEVER use LAG(SUM(...)) directly in SELECT - use CTEs first to aggregate, then apply window functions

PRICE BANDS AND PERCENTILES:
- For price quartiles, use: NTILE(4) OVER (ORDER BY price) AS price_quartile
- For custom percentiles, use subqueries:
  WITH price_stats AS (
    SELECT AVG(price) as p25, AVG(price) as p50, AVG(price) as p75
    FROM (SELECT price, NTILE(4) OVER (ORDER BY price) as quartile FROM data) t
    WHERE quartile IN (1,2,3)
  )
- Example price bands: CASE WHEN price < p25 THEN 'P25' WHEN price < p50 THEN 'P50' ELSE 'P75+' END

Default views to query (SELECT-only):
- analytics_shipments(date, year, month, therapy, product_name, skus, dosage_form, uom, quantity, value_inr, price_inr_per_unit, indian_company, supplier, country, city, continent)
- analytics_shipments_with_gc(same fields + cost_inr_per_unit, cost_inr_total, gc_inr, gc_pct)
- analytics_mv_monthly_therapy_supplier(month_date, therapy, supplier, country, dosage_form, qty, value_inr, median_price_inr)

SAFETY:
- Query only the views above (or tables explicitly allowlisted in env).
- SELECT-only. No DDL/DML. Always include LIMIT {MAX_ROWS} when previewing rows.
- Use AVG() for averages, not MEDIAN() which doesn't exist in MySQL.
- GROUP BY RULE: If you SELECT MONTH(date), YEAR(date), or any non-aggregated column, it MUST be in GROUP BY.
- WINDOW FUNCTIONS: Use proper syntax with PARTITION BY and ORDER BY clauses.
"""

# ------------------------------
# State schema helpers (same as clarify build)
# ------------------------------
STATE_DDL = """
CREATE TABLE IF NOT EXISTS athena_sessions (
  session_id VARCHAR(64) PRIMARY KEY,
  title VARCHAR(255) NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS athena_messages (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  session_id VARCHAR(64) NOT NULL,
  role ENUM('user','assistant','system','tool') NOT NULL,
  content MEDIUMTEXT NOT NULL,
  tool_name VARCHAR(64) NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  INDEX ix_sess_time (session_id, created_at),
  CONSTRAINT fk_msg_session
    FOREIGN KEY (session_id) REFERENCES athena_sessions(session_id)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS athena_summaries (
  session_id VARCHAR(64) PRIMARY KEY,
  summary TEXT NOT NULL,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  CONSTRAINT fk_sum_session
    FOREIGN KEY (session_id) REFERENCES athena_sessions(session_id)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

def ensure_state_schema(connection: Engine):
    if AUTO_CREATE_STATE_SCHEMA:
        with connection.begin() as conn:
            for stmt in STATE_DDL.strip().split(";\n\n"):
                s = stmt.strip()
                if s:
                    conn.execute(text(s))

def ensure_session(connection: Engine, session_id: str, title: Optional[str] = None):
    with connection.begin() as conn:
        conn.execute(text("""
            INSERT INTO athena_sessions (session_id, title)
            VALUES (:sid, :title)
            ON DUPLICATE KEY UPDATE title = COALESCE(:title, athena_sessions.title)
        """), {"sid": session_id, "title": title})

def append_message(connection: Engine, session_id: str, role: str, content: str, tool_name: Optional[str] = None):
    with connection.begin() as conn:
        conn.execute(text("""
            INSERT INTO athena_messages (session_id, role, content, tool_name)
            VALUES (:sid, :role, :content, :tool)
        """), {"sid": session_id, "role": role, "content": content, "tool": tool_name})

def recent_dialogue(connection: Engine, session_id: str, max_turns: int = 8) -> List[Dict[str, str]]:
    with connection.begin() as conn:
        res = conn.execute(text("""
            SELECT role, content
            FROM athena_messages
            WHERE session_id = :sid
            ORDER BY created_at DESC
            LIMIT :lim
        """), {"sid": session_id, "lim": max_turns})
        rows = res.fetchall()
    rows = list(reversed([{"role": r[0], "content": r[1]} for r in rows]))
    return rows

def get_summary(connection: Engine, session_id: str) -> str:
    with connection.begin() as conn:
        res = conn.execute(text("""
            SELECT summary FROM athena_summaries WHERE session_id = :sid
        """), {"sid": session_id}).fetchone()
        return res[0] if res else ""

def save_summary(connection: Engine, session_id: str, summary: str):
    clipped = summary[:1000]
    with connection.begin() as conn:
        conn.execute(text("""
            INSERT INTO athena_summaries (session_id, summary)
            VALUES (:sid, :s)
            ON DUPLICATE KEY UPDATE summary = :s
        """), {"sid": session_id, "s": clipped})
        conn.execute(text("""
            UPDATE athena_sessions SET updated_at = CURRENT_TIMESTAMP WHERE session_id = :sid
        """), {"sid": session_id})

# ------------------------------
# Tools
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

REQUEST_CLARIFICATION_TOOL = {
  "type": "function",
  "function": {
    "name": "request_clarification",
    "description": "Ask the user ONE minimal blocking question (e.g., time window) with up to 4 option chips.",
    "parameters": {
      "type": "object",
      "properties": {
        "question": { "type": "string" },
        "options":  { "type": "array", "items": { "type": "string" }, "maxItems": 4 }
      },
      "required": ["question"]
    }
  }
}

TOOLS = [RUN_SQL_TOOL, REQUEST_CLARIFICATION_TOOL]

# ------------------------------
# SQL safety
# ------------------------------
DANGEROUS = re.compile(r"\\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|GRANT|REVOKE|CREATE|MERGE)\\b", re.I)
TABLE_REGEXES = [
    re.compile(r"\\bFROM\\s+([`\"\\w\\.]+)", re.I),
    re.compile(r"\\bJOIN\\s+([`\"\\w\\.]+)", re.I)
]

def is_safe_select(sql: str) -> bool:
    s = sql.strip().rstrip(";")
    # Allow both SELECT and WITH (Common Table Expression) statements
    if not (s.lower().startswith("select") or s.lower().startswith("with")):
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
            tbl = tbl.split(".")[-1]
            found.append(tbl)
    return list({t for t in found})

def enforce_allowlist(sql: str) -> None:
    tables = referenced_tables(sql)
    for t in tables:
        if t not in ALLOWED_TABLES and not t.startswith("analytics_"):
            raise ValueError(f"Table '{t}' not allowed. Allowed: {sorted(ALLOWED_TABLES)} or names starting with analytics_.")

def ensure_limit(sql: str) -> str:
    s = sql.strip().rstrip(";")
    if re.search(r"\bLIMIT\b", s, re.I):
        return s
    return f"{s} LIMIT {MAX_ROWS}"

# ------------------------------
# Minimal cache
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
    title: Optional[str] = Field(None, description="Optional session title")

class ChatResponse(BaseModel):
    answer: str
    data_preview: Dict[str, Any]
    sql: Optional[str]
    latency_ms: int
    clarify: Optional[Dict[str, Any]] = None

# ------------------------------
# Routing helpers
# ------------------------------
COMPLEX_KEYWORDS = [
    "why","explain","trend","growth","cagr","yoy","forecast","seasonality","anomaly",
    "gc","gross contribution","price band","bands","median","percentile","compare","vs","year over year"
]

def pick_model(user_text: str) -> str:
    """Return simple model for short, unambiguous asks; else primary model."""
    lt = (user_text or "").lower()
    if len(lt) < 180 and not any(k in lt for k in COMPLEX_KEYWORDS):
        return MODEL_SIMPLE
    return MODEL_PRIMARY

# ------------------------------
# System prompt
# ------------------------------
SYSTEM_PROMPT = f"""
You are Athena, a cautious, data-grounded analyst for Indian import/export (pharma). 
- Use **MySQL 8** syntax. Prefer **analytics_shipments**; for GC questions, use **{RULEBOOK['gc']['gc_view']}** if RULEBOOK.gc.compute_gc is true.
- If the user request is ambiguous and would change the result materially, CALL the tool `request_clarification` with ONE crisp question and up to 4 options.
- Otherwise, proceed with RULEBOOK.clarify.defaults and state assumptions in one line.
- ALWAYS explain the data window and filters. When previewing rows, include LIMIT.
{SCHEMA_CARD}
RULES: {json.dumps(RULEBOOK)}
"""

# ------------------------------
# Helper function to convert Decimal to float for JSON serialization
# ------------------------------
def convert_decimals(obj):
    """Convert Decimal objects to float for JSON serialization"""
    if isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_decimals(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_decimals(item) for item in obj]
    else:
        return obj

# ------------------------------
# SQL execution
# ------------------------------
def run_sql_query(sql: str) -> Dict[str, Any]:
    print(f"🔍 DEBUG: SQL Query being checked:")
    print(f"SQL: {sql}")
    print(f"Starts with SELECT: {sql.strip().lower().startswith('select')}")
    print(f"Contains dangerous keywords: {bool(DANGEROUS.search(sql))}")
    
    if not is_safe_select(sql):
        print(f"❌ SQL REJECTED by safety check")
        raise ValueError("Only SELECT statements are allowed.")
    else:
        print(f"✅ SQL PASSED safety check")
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
    # Convert Decimal objects to float for JSON serialization
    payload = convert_decimals(payload)
    cache_set(sig, payload)
    return payload

# ------------------------------
# Model call with tools
# ------------------------------
def trim_messages_for_context(messages: List[Dict[str, str]], max_messages: int = 20) -> List[Dict[str, str]]:
    """Trim messages to prevent context length exceeded errors"""
    if len(messages) <= max_messages:
        return messages
    
    # Keep the first few messages (system context) and the last few messages (recent conversation)
    system_messages = [msg for msg in messages[:5] if msg.get("role") in ["system", "assistant"] and "SCHEMA_CARD" in msg.get("content", "")]
    recent_messages = messages[-(max_messages - len(system_messages)):]
    
    # Combine system context with recent conversation
    trimmed = system_messages + recent_messages
    
    print(f"🔧 Trimmed messages from {len(messages)} to {len(trimmed)} to prevent context overflow")
    return trimmed

def model_decide_and_answer(user_text: str, messages: List[Dict[str, str]]) -> Dict[str, Any]:
    # Trim messages to prevent context length exceeded errors
    messages = trim_messages_for_context(messages)
    
    first_model = pick_model(user_text)
    first = client.chat.completions.create(
        model=first_model,
        temperature=0.2,
        tools=TOOLS,
        tool_choice="auto",
        messages=messages
    )
    msg = first.choices[0].message
    tool_calls = getattr(msg, "tool_calls", None)

    data_preview = {"columns": [], "rows": []}
    sql_used = None
    clarify_payload = None

    if tool_calls:
        for tc in tool_calls:
            if tc.type == "function" and tc.function.name == "request_clarification":
                args = json.loads(tc.function.arguments or "{}")
                question = args.get("question") or "Please clarify your request."
                options = args.get("options") or []
                clarify_payload = {"question": question, "options": options[:4]}
                return {"answer": question, "data_preview": data_preview, "sql": None, "clarify": clarify_payload}

            if tc.type == "function" and tc.function.name == "run_sql":
                args = json.loads(tc.function.arguments or "{}")
                raw_sql = args.get("query", "")
                data_preview = run_sql_query(raw_sql)
                sql_used = ensure_limit(raw_sql)
                break

        if sql_used:
            # Use primary model for final synthesis
            # Find the specific run_sql tool call that was executed
            run_sql_tool_call = None
            for tc in tool_calls:
                if tc.type == "function" and tc.function.name == "run_sql":
                    run_sql_tool_call = tc
                    break
            
            if run_sql_tool_call:
                # Build proper message sequence: original messages + assistant message with tool call + tool result
                second_messages = messages + [
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [run_sql_tool_call]
                    },
                    {
                        "role": "tool",
                        "tool_call_id": run_sql_tool_call.id,
                        "name": "run_sql",
                        "content": json.dumps(data_preview)
                    }
                ]
                second = client.chat.completions.create(
                    model=MODEL_PRIMARY,
                    temperature=0.2,
                    tools=TOOLS,
                    messages=second_messages
                )
                final_text = second.choices[0].message.content or ""
            else:
                final_text = msg.content or ""
        else:
            final_text = msg.content or ""
    else:
        final_text = msg.content or ""

    return {"answer": final_text, "data_preview": data_preview, "sql": sql_used, "clarify": clarify_payload}

# ------------------------------
# API
# ------------------------------
class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Unique id per chat thread")
    message: str = Field(..., description="User's question in plain English")
    title: Optional[str] = Field(None, description="Optional session title")

class ChatResponse(BaseModel):
    answer: str
    data_preview: Dict[str, Any]
    sql: Optional[str]
    latency_ms: int
    clarify: Optional[Dict[str, Any]] = None

@app.on_event("startup")
def boot():
    ensure_state_schema(engine)

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest = Body(...)):
    # Ensure session
    ensure_session(engine, req.session_id, title=req.title)

    # Build context: summary + last turns
    summary = get_summary(engine, req.session_id)
    recent = recent_dialogue(engine, req.session_id, max_turns=8)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if summary:
        messages.append({"role": "system", "content": f"Conversation summary: {summary}"})
    messages.extend(recent)
    messages.append({"role": "user", "content": req.message})

    # Save user message
    append_message(engine, req.session_id, "user", req.message)

    # Call model
    t0 = time.time()
    result = model_decide_and_answer(req.message, messages)
    latency_ms = int((time.time() - t0) * 1000)

    # Save assistant message
    append_message(engine, req.session_id, "assistant", result["answer"])

    # Update running summary (skip if clarifying)
    if not result.get("clarify"):
        sum_prompt = [
            {"role": "system", "content": "Summarize this chat for future turns in <1000 chars focusing on filters, assumptions, and goals."},
            {"role": "user", "content": f"Previous summary:\n{summary}\n\nLatest exchange:\nUser: {req.message}\nAssistant: {result['answer']}"}
        ]
        try:
            s2 = client.chat.completions.create(model=MODEL_SUMMARY, temperature=0.2, messages=sum_prompt)
            new_summary = s2.choices[0].message.content or ""
            save_summary(engine, req.session_id, new_summary)
        except Exception:
            pass

    return ChatResponse(
        answer=result["answer"],
        data_preview=result["data_preview"],
        sql=result["sql"],
        latency_ms=latency_ms,
        clarify=result["clarify"]
    )

@app.get("/healthz")
def health():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False
    return {"ok": True, "db": db_ok}

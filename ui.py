import streamlit as st
import requests
import json
import threading

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="miniVllm Dashboard", layout="wide")
st.title("miniVllm Dashboard")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_USERS = 8

DEFAULT_PROMPTS = [
    "Explain how neural networks learn using backpropagation, step by step.",
    "Explain how neural networks learn using backpropagation, step by step. Include the math behind gradient descent.",
    "What is the capital of France?",
    "Write a Python function that checks if a number is prime.",
]
DEFAULT_MAX_TOKENS = [256, 512, 32, 128]
GENERIC_PROMPT     = "Enter your prompt here."
GENERIC_MAX_TOKENS = 256


def _default_prompt(slot_id: int) -> str:
    return DEFAULT_PROMPTS[slot_id] if slot_id < len(DEFAULT_PROMPTS) else GENERIC_PROMPT


def _default_max_tokens(slot_id: int) -> int:
    return DEFAULT_MAX_TOKENS[slot_id] if slot_id < len(DEFAULT_MAX_TOKENS) else GENERIC_MAX_TOKENS


# ---------------------------------------------------------------------------
# Per-slot session state helpers
# ---------------------------------------------------------------------------
def _init_slot(slot_id: int):
    """Idempotently initialise session state for a slot."""
    st.session_state.setdefault(f"response_{slot_id}",   "")
    st.session_state.setdefault(f"metrics_{slot_id}",    None)
    st.session_state.setdefault(f"error_{slot_id}",      "")
    st.session_state.setdefault(f"generating_{slot_id}", False)
    if f"buf_{slot_id}" not in st.session_state:
        st.session_state[f"buf_{slot_id}"] = {
            "tokens": [],
            "metrics": None,
            "done": True,
            "error": "",
            "lock": threading.Lock(),
        }


# Ordered list of active slot IDs and a monotonic counter for new ones
if "active_slots" not in st.session_state:
    st.session_state.active_slots = list(range(4))   # [0, 1, 2, 3]
    st.session_state.next_slot_id = 4

for _sid in st.session_state.active_slots:
    _init_slot(_sid)


# ---------------------------------------------------------------------------
# Streaming helper
# ---------------------------------------------------------------------------
def start_stream(buf, prompt, max_tokens):
    """Reset buffer and launch background thread that streams from the API."""
    with buf["lock"]:
        buf["tokens"]  = []
        buf["metrics"] = None
        buf["done"]    = False
        buf["error"]   = ""

    def _run():
        try:
            with requests.post(
                f"{API_BASE}/v1/completions",
                json={"prompt": prompt, "max_tokens": max_tokens, "stream": True},
                stream=True,
                timeout=120,
            ) as r:
                if not r.ok:
                    with buf["lock"]:
                        buf["error"] = f"API error {r.status_code}: {r.text[:200]}"
                    return
                for line in r.iter_lines():
                    if not line:
                        continue
                    if line.startswith(b"data: "):
                        data = line[6:]
                        if data == b"[DONE]":
                            continue
                        chunk = json.loads(data)
                        if chunk.get("type") == "metrics":
                            with buf["lock"]:
                                buf["metrics"] = chunk["metrics"]
                        else:
                            token = chunk["choices"][0]["delta"]["content"]
                            with buf["lock"]:
                                buf["tokens"].append(token)
        except requests.exceptions.ConnectionError:
            with buf["lock"]:
                buf["error"] = "Cannot reach backend — is the server running?"
        except Exception as e:
            with buf["lock"]:
                buf["error"] = str(e)
        finally:
            with buf["lock"]:
                buf["done"] = True

    threading.Thread(target=_run, daemon=True).start()


# ---------------------------------------------------------------------------
# Global metrics — own fragment so Refresh never touches user panels
# ---------------------------------------------------------------------------
@st.fragment(run_every=10)
def render_global_metrics():
    try:
        r = requests.get(f"{API_BASE}/metrics", timeout=2)
        gm = r.json() if r.ok else None
    except Exception:
        gm = None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active Requests", gm["active_requests"] if gm else "—")
    c2.metric("Queue Size",      gm["queue_size"]       if gm else "—")
    c3.metric("Total Requests",  gm["total_requests"]   if gm else "—")
    with c4:
        if st.button("Refresh Metrics", use_container_width=True):
            st.rerun(scope="fragment")
    if not gm:
        st.error("Backend offline")


render_global_metrics()
st.divider()


# ---------------------------------------------------------------------------
# Per-user inference metrics — 3 columns × 2 rows so values don't get clipped
# ---------------------------------------------------------------------------
def render_inference_metrics(m):
    r1c1, r1c2, r1c3 = st.columns(3)
    r1c1.metric("Token/sec",    m["tokens_per_sec"])
    r1c2.metric("TTFT",         f"{m['ttft_ms']} ms")
    r1c3.metric("Prefix Cache", m["prefix_cache"])

    r2c1, r2c2, r2c3 = st.columns(3)
    r2c1.metric("KV Blocks",    m["kv_blocks"])
    r2c2.metric("Batch ID",     m["batch_id"])
    r2c3.metric("Latency",      f"{m['latency_ms']} ms")

    if m["prefix_cache"] == "HIT":
        st.success("Prefix cache hit — tokens reused from a previous request")


# ---------------------------------------------------------------------------
# User panel — independent fragment that polls its own buffer.
# run_every=0.5 drives the streaming display without blocking other panels.
# ---------------------------------------------------------------------------
@st.fragment(run_every=0.5)
def render_user(slot_id: int):
    buf           = st.session_state[f"buf_{slot_id}"]
    is_generating = st.session_state[f"generating_{slot_id}"]

    # Header: label + Remove button
    hdr_left, hdr_right = st.columns([4, 1])
    user_num = st.session_state.active_slots.index(slot_id) + 1
    hdr_left.markdown(f"### User {user_num}")
    if hdr_right.button("Remove", key=f"remove_{slot_id}", use_container_width=True):
        st.session_state.active_slots.remove(slot_id)
        st.rerun()   # full-page rerun to rebuild grid

    prompt = st.text_area(
        "Prompt",
        value=_default_prompt(slot_id),
        key=f"prompt_{slot_id}",
        height=90,
        label_visibility="collapsed",
    )
    col_btn, col_slider = st.columns([1, 2])
    with col_btn:
        clicked = st.button(
            "Generating..." if is_generating else "Generate",
            key=f"submit_{slot_id}",
            use_container_width=True,
            disabled=is_generating or not (prompt or "").strip(),
            type="primary",
        )
    with col_slider:
        max_tokens = st.slider(
            "Max tokens", 32, 512,
            _default_max_tokens(slot_id),
            key=f"max_tokens_{slot_id}",
        )

    # Fixed-height scrollable response area keeps boxes aligned across columns
    with st.container(height=250):
        response_placeholder = st.empty()

    # --- Button clicked: reset state, launch thread, rerun to show disabled button ---
    if clicked:
        st.session_state[f"generating_{slot_id}"] = True
        st.session_state[f"response_{slot_id}"]   = ""
        st.session_state[f"metrics_{slot_id}"]    = None
        st.session_state[f"error_{slot_id}"]      = ""
        start_stream(buf, prompt.strip(), max_tokens)
        st.rerun(scope="fragment")

    # --- Poll buffer and update display ---
    if is_generating:
        with buf["lock"]:
            full_text = "".join(buf["tokens"])
            is_done   = buf["done"]
            metrics   = buf["metrics"]
            error     = buf["error"]

        if error:
            st.session_state[f"error_{slot_id}"]      = error
            st.session_state[f"generating_{slot_id}"] = False
        elif is_done:
            response_placeholder.markdown(full_text)
            st.session_state[f"response_{slot_id}"]   = full_text
            st.session_state[f"metrics_{slot_id}"]    = metrics
            st.session_state[f"generating_{slot_id}"] = False
        else:
            response_placeholder.markdown(full_text + " ▌")
    elif st.session_state[f"response_{slot_id}"]:
        response_placeholder.markdown(st.session_state[f"response_{slot_id}"])

    if st.session_state[f"error_{slot_id}"]:
        st.error(st.session_state[f"error_{slot_id}"])

    m = st.session_state[f"metrics_{slot_id}"]
    if m:
        render_inference_metrics(m)


# ---------------------------------------------------------------------------
# Add User button + dynamic 2-column grid
# ---------------------------------------------------------------------------
slots   = st.session_state.active_slots
can_add = len(slots) < MAX_USERS

if st.button(
    "+ Add User",
    disabled=not can_add,
    help=f"Maximum {MAX_USERS} users" if not can_add else "",
):
    new_id = st.session_state.next_slot_id
    st.session_state.next_slot_id += 1
    _init_slot(new_id)
    st.session_state.active_slots.append(new_id)
    st.rerun()

for row_start in range(0, len(slots), 2):
    row_slots = slots[row_start : row_start + 2]
    cols = st.columns(len(row_slots), gap="large")
    for col, sid in zip(cols, row_slots):
        with col:
            with st.container(border=True):
                render_user(sid)

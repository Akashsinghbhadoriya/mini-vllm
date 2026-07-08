import streamlit as st
import requests
import json
import threading

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="miniVllm Dashboard", layout="wide")
st.title("miniVllm Dashboard")

# ---------------------------------------------------------------------------
# Per-user stream buffers stored in session_state.
# Background threads mutate the dict contents (not the key), which avoids
# the ScriptRunContext error that happens when threads do st.session_state[k]=v
# ---------------------------------------------------------------------------
NUM_USERS = 4

DEFAULT_PROMPTS = [
    "Explain how neural networks learn using backpropagation, step by step.",
    "Explain how neural networks learn using backpropagation, step by step. Include the math behind gradient descent.",
    "What is the capital of France?",
    "Write a Python function that checks if a number is prime.",
]

DEFAULT_MAX_TOKENS = [256, 512, 32, 128]

for i in range(NUM_USERS):
    st.session_state.setdefault(f"response_{i}",   "")
    st.session_state.setdefault(f"metrics_{i}",    None)
    st.session_state.setdefault(f"error_{i}",      "")
    st.session_state.setdefault(f"generating_{i}", False)
    if f"buf_{i}" not in st.session_state:
        st.session_state[f"buf_{i}"] = {
            "tokens": [],
            "metrics": None,
            "done": True,
            "error": "",
            "lock": threading.Lock(),
        }


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
    c2.metric("Queue Size",       gm["queue_size"]       if gm else "—")
    c3.metric("Total Requests",   gm["total_requests"]   if gm else "—")
    with c4:
        if st.button("Refresh Metrics", use_container_width=True):
            st.rerun(scope="fragment")
    if not gm:
        st.error("Backend offline")


render_global_metrics()
st.divider()


# ---------------------------------------------------------------------------
# Per-user inference metrics display
# ---------------------------------------------------------------------------
def render_inference_metrics(m):
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Token/sec",    m["tokens_per_sec"])
    m2.metric("TTFT",         f"{m['ttft_ms']} ms")
    m3.metric("Prefix Cache", m["prefix_cache"])
    m4.metric("KV Blocks",    m["kv_blocks"])
    m5.metric("Batch ID",     m["batch_id"])
    m6.metric("Latency",      f"{m['latency_ms']} ms")
    if m["prefix_cache"] == "HIT":
        st.success("Prefix cache hit — tokens reused from a previous request")


# ---------------------------------------------------------------------------
# User panel — each is an independent fragment that polls its own buffer.
# run_every=0.5 drives the streaming display without blocking other panels.
# ---------------------------------------------------------------------------
@st.fragment(run_every=0.5)
def render_user(i):
    buf          = st.session_state[f"buf_{i}"]
    is_generating = st.session_state[f"generating_{i}"]

    st.markdown(f"### User {i + 1}")

    prompt = st.text_area(
        "Prompt",
        value=DEFAULT_PROMPTS[i],
        key=f"prompt_{i}",
        height=90,
        label_visibility="collapsed",
    )
    col_btn, col_slider = st.columns([1, 2])
    with col_btn:
        clicked = st.button(
            "Generating..." if is_generating else "Generate",
            key=f"submit_{i}",
            use_container_width=True,
            disabled=is_generating or not (prompt or "").strip(),
            type="primary",
        )
    with col_slider:
        max_tokens = st.slider("Max tokens", 32, 512, DEFAULT_MAX_TOKENS[i], key=f"max_tokens_{i}")

    # Response area — always rendered here, below the controls
    response_placeholder = st.empty()

    # --- Button clicked: reset state, launch thread, rerun to show disabled button ---
    if clicked:
        st.session_state[f"generating_{i}"] = True
        st.session_state[f"response_{i}"]   = ""
        st.session_state[f"metrics_{i}"]    = None
        st.session_state[f"error_{i}"]      = ""
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
            st.session_state[f"error_{i}"]      = error
            st.session_state[f"generating_{i}"] = False
        elif is_done:
            response_placeholder.markdown(full_text)
            st.session_state[f"response_{i}"]   = full_text
            st.session_state[f"metrics_{i}"]    = metrics
            st.session_state[f"generating_{i}"] = False
        else:
            response_placeholder.markdown(full_text + " ▌")
    elif st.session_state[f"response_{i}"]:
        response_placeholder.markdown(st.session_state[f"response_{i}"])

    if st.session_state[f"error_{i}"]:
        st.error(st.session_state[f"error_{i}"])

    m = st.session_state[f"metrics_{i}"]
    if m:
        render_inference_metrics(m)


# ---------------------------------------------------------------------------
# 2×2 grid
# ---------------------------------------------------------------------------
row1 = st.columns(2, gap="large")
row2 = st.columns(2, gap="large")

for i, col in enumerate([row1[0], row1[1], row2[0], row2[1]]):
    with col:
        with st.container(border=True):
            render_user(i)

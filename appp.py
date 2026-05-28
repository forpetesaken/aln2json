import re
import streamlit as st
import json
import zipfile
import io
from collections import Counter

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ALN → AlphaFold JSON",
    page_icon="🧬",
    layout="wide",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}
h1, h2, h3 {
    font-family: 'IBM Plex Mono', monospace !important;
}

/* Dark science-lab aesthetic */
.stApp {
    background-color: #0d1117;
    color: #c9d1d9;
}
.block-container {
    padding-top: 2rem;
}

/* Cards */
.card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}

/* Sequence display */
.seq-block {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    line-height: 1.7;
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 4px;
    padding: 0.75rem 1rem;
    overflow-x: auto;
    white-space: pre;
    color: #8b949e;
}
.seq-block .highlight {
    background: #1f4e35;
    color: #56d364;
    border-radius: 2px;
}

/* Metric boxes */
.metric-row {
    display: flex;
    gap: 1rem;
    margin: 1rem 0;
}
.metric-box {
    flex: 1;
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 0.75rem 1rem;
    text-align: center;
}
.metric-box .val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    color: #58a6ff;
}
.metric-box .lbl {
    font-size: 0.72rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Tag chips */
.chip {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    background: #1f4e35;
    color: #56d364;
    border-radius: 20px;
    padding: 2px 10px;
    margin: 2px 3px;
}
.chip-warn {
    background: #3d2b00;
    color: #e3b341;
}

/* Override Streamlit button */
div.stButton > button {
    font-family: 'IBM Plex Mono', monospace;
    background: #238636;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 0.5rem 1.5rem;
    font-size: 0.9rem;
    font-weight: 600;
    width: 100%;
    transition: background 0.15s;
}
div.stButton > button:hover {
    background: #2ea043;
    color: white;
    border: none;
}

div.stDownloadButton > button {
    font-family: 'IBM Plex Mono', monospace;
    background: #1f6feb;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 0.5rem 1.5rem;
    font-size: 0.9rem;
    font-weight: 600;
    width: 100%;
}
div.stDownloadButton > button:hover {
    background: #388bfd;
    color: white;
    border: none;
}

/* Input labels */
label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.82rem !important;
    color: #8b949e !important;
}

/* Divider */
hr {
    border-color: #21262d;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #30363d;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_aln(text: str) -> dict[str, str]:
    """Parse Clustal .aln or multi-FASTA into {name: gapped_sequence}."""
    sequences: dict[str, str] = {}
    lines = text.splitlines()

    # Detect format
    is_fasta = any(l.startswith(">") for l in lines)

    if is_fasta:
        current = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                current = line[1:].split()[0]
                sequences[current] = ""
            elif current:
                sequences[current] += line.upper()
    else:
        # Clustal: lines are "NAME  SEQUENCE", skip header / annotation lines
        for line in lines:
            line = line.strip()
            if not line or line.startswith("CLUSTAL") or set(line) <= set("*:. "):
                continue
            parts = line.split()
            if len(parts) >= 2:
                name, seq = parts[0], parts[-1]
                if all(c in "ACDEFGHIKLMNPQRSTVWY-" for c in seq.upper()):
                    sequences.setdefault(name, "")
                    sequences[name] += seq.upper()
    return sequences


def ref_to_aln_index(ref_seq: str) -> list[int]:
    """Map reference (ungapped) residue numbers (1-based) → alignment column indices."""
    mapping = []  # mapping[ref_pos] = aln_col  (ref_pos is 0-based)
    for col, aa in enumerate(ref_seq):
        if aa != "-":
            mapping.append(col)
    return mapping


def extract_region(sequences: dict[str, str],
                   ref_name: str,
                   res_start: int,
                   res_end: int) -> dict[str, str]:
    """
    Extract alignment columns corresponding to reference residues res_start..res_end (1-based, inclusive).
    Returns {name: ungapped_sequence} for each sequence.
    """
    ref_seq = sequences[ref_name]
    mapping = ref_to_aln_index(ref_seq)

    # Validate
    if res_start < 1 or res_end > len(mapping) or res_start > res_end:
        raise ValueError(
            f"Residue range {res_start}–{res_end} is out of bounds "
            f"(reference has {len(mapping)} residues)."
        )

    col_start = mapping[res_start - 1]
    col_end   = mapping[res_end - 1]

    extracted = {}
    for name, seq in sequences.items():
        region = seq[col_start : col_end + 1].replace("-", "")
        if region:
            extracted[name] = region
    return extracted


def sanitize_name(name: str) -> str:
    """Replace any character that isn't alphanumeric, hyphen, or underscore with '_'."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", name)


def build_af_json(name: str, sequence: str) -> dict:
    return [{
        "name": name,
        "modelSeeds": [],
        "sequences": [{"proteinChain": {"sequence": sequence, "count": 1}}],
        "dialect": "alphafoldserver",
        "version": 1,
    }]


def make_zip(extracted: dict[str, str], ref_name: str, res_start: int, res_end: int) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # Individual JSONs
        for name, seq in extracted.items():
            safe = sanitize_name(name)
            job_name = f"{safe}_r{res_start}-{res_end}"
            data = json.dumps(build_af_json(job_name, seq), indent=2)
            zf.writestr(f"{safe}.json", data)

        # FASTA
        fasta_lines = [f">{n}\n{s}" for n, s in extracted.items()]
        zf.writestr("sequences.fasta", "\n".join(fasta_lines))

    return buf.getvalue()


def render_sequence_preview(seq: str, col_start: int, col_end: int, max_cols: int = 80) -> str:
    """Return HTML showing alignment with highlighted region (first 3 sequences)."""
    before = seq[:col_start]
    region = seq[col_start:col_end + 1]
    after  = seq[col_end + 1:]
    # Truncate for display
    if len(before) > 20:
        before = "…" + before[-20:]
    if len(after) > 20:
        after = after[:20] + "…"
    return (
        f'<span style="color:#8b949e">{before}</span>'
        f'<span style="background:#1f4e35;color:#56d364;border-radius:2px">{region}</span>'
        f'<span style="color:#8b949e">{after}</span>'
    )


# ── UI ────────────────────────────────────────────────────────────────────────

st.markdown("## 🧬 ALN → AlphaFold JSON")
st.markdown(
    "<p style='color:#8b949e;font-size:0.9rem;margin-top:-0.5rem'>"
    "Upload a multiple sequence alignment, specify a residue range on your reference sequence, "
    "and download per-sequence AlphaFold Server JSON files."
    "</p>",
    unsafe_allow_html=True,
)

st.markdown("---")

# ── Sidebar: settings ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    min_seq_len = st.slider("Min output sequence length (aa)", 5, 100, 20,
                            help="Sequences shorter than this after gap removal are excluded.")
    st.markdown("---")
    st.markdown(
        "<p style='color:#8b949e;font-size:0.78rem'>"
        "Output JSONs are formatted for <b>AlphaFold Server</b> "
        "(<code>alphafoldserver</code> dialect, v1)."
        "</p>",
        unsafe_allow_html=True,
    )

# ── Step 1: Upload ─────────────────────────────────────────────────────────────
st.markdown("#### 1 · Upload alignment")
uploaded = st.file_uploader("", type=["aln", "fasta", "fa", "fas", "txt"],
                             label_visibility="collapsed")

sequences = {}
if uploaded:
    raw = uploaded.read().decode("utf-8", errors="replace")
    try:
        sequences = parse_aln(raw)
        seq_names = list(sequences.keys())
        aln_len   = len(next(iter(sequences.values())))

        st.markdown(
            f"<div class='metric-row'>"
            f"<div class='metric-box'><div class='val'>{len(sequences)}</div><div class='lbl'>Sequences</div></div>"
            f"<div class='metric-box'><div class='val'>{aln_len}</div><div class='lbl'>Alignment columns</div></div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown("#### 1b · Select species / sequences to include")
        selected_names = st.multiselect(
            "Sequences to include in output",
            options=seq_names,
            default=seq_names,
            help="Only selected sequences will appear in the extracted JSONs. "
                 "The reference sequence is always kept.",
        )
        # Ensure at least the reference key exists; filter after ref is chosen below
        if not selected_names:
            st.warning("No sequences selected — all will be used.")
            selected_names = seq_names

        sequences = {k: v for k, v in sequences.items() if k in selected_names}

    except Exception as e:
        st.error(f"Could not parse alignment: {e}")
        sequences = {}

# ── Step 2: Reference + range ─────────────────────────────────────────────────
if sequences:
    # Always keep reference sequence in the dict even if somehow deselected
    st.markdown("#### 2 · Reference sequence & residue range")

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        ref_name = st.selectbox("Reference sequence", list(sequences.keys()),
                                help="Residue numbers are counted on this sequence (gaps ignored).")
    
    ref_ungapped = sequences[ref_name].replace("-", "")
    ref_len = len(ref_ungapped)

    with col2:
        res_start = st.number_input("Start residue", min_value=1, max_value=ref_len,
                                    value=1, step=1)
    with col3:
        res_end = st.number_input("End residue", min_value=1, max_value=ref_len,
                                  value=min(50, ref_len), step=1)

    # Quick reference preview
    preview_seq = ref_ungapped
    start_idx = res_start - 1
    end_idx   = res_end
    before = preview_seq[:start_idx]
    region = preview_seq[start_idx:end_idx]
    after  = preview_seq[end_idx:]
    if len(before) > 25: before = "…" + before[-25:]
    if len(after) > 25:  after  = after[:25] + "…"

    st.markdown(
        f"<div class='card'>"
        f"<div style='font-family:IBM Plex Mono,monospace;font-size:0.72rem;color:#8b949e;margin-bottom:4px'>"
        f"{ref_name} · residues {res_start}–{res_end} of {ref_len}"
        f"</div>"
        f"<div class='seq-block'>"
        f"<span style='color:#8b949e'>{before}</span>"
        f"<span style='background:#1f4e35;color:#56d364;padding:0 1px'>{region}</span>"
        f"<span style='color:#8b949e'>{after}</span>"
        f"</div>"
        f"<div style='margin-top:8px;font-size:0.78rem;color:#8b949e'>"
        f"Region length: <span style='color:#56d364;font-family:IBM Plex Mono,monospace'>"
        f"{res_end - res_start + 1} residues</span>"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Step 3: Extract ────────────────────────────────────────────────────────
    st.markdown("#### 3 · Extract & export")

    if res_start > res_end:
        st.warning("Start residue must be ≤ end residue.")
    else:
        if st.button("✂️  Extract region"):
            try:
                extracted_raw = extract_region(sequences, ref_name, res_start, res_end)

                # Apply min length filter
                extracted = {
                    name: seq for name, seq in extracted_raw.items()
                    if len(seq) >= min_seq_len
                }
                skipped = len(extracted_raw) - len(extracted)

                st.session_state["extracted"] = extracted
                st.session_state["res_start"] = res_start
                st.session_state["res_end"]   = res_end
                st.session_state["skipped"]   = skipped

            except ValueError as e:
                st.error(str(e))

    # ── Results ────────────────────────────────────────────────────────────────
    if "extracted" in st.session_state:
        extracted = st.session_state["extracted"]
        rs = st.session_state["res_start"]
        re_ = st.session_state["res_end"]
        skipped = st.session_state["skipped"]

        st.markdown("---")
        st.markdown("#### Results")

        # Metrics
        lengths = [len(s) for s in extracted.values()]
        avg_len = sum(lengths) / len(lengths) if lengths else 0
        st.markdown(
            f"<div class='metric-row'>"
            f"<div class='metric-box'><div class='val'>{len(extracted)}</div><div class='lbl'>Sequences extracted</div></div>"
            f"<div class='metric-box'><div class='val'>{int(avg_len)}</div><div class='lbl'>Avg length (aa)</div></div>"
            f"<div class='metric-box'><div class='val'>{min(lengths) if lengths else 0}–{max(lengths) if lengths else 0}</div><div class='lbl'>Length range</div></div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        if skipped:
            st.markdown(
                f"<p style='color:#e3b341;font-size:0.82rem'>⚠️ {skipped} sequence(s) skipped "
                f"(shorter than {min_seq_len} aa after gap removal).</p>",
                unsafe_allow_html=True,
            )

        # Sequence table preview
        with st.expander("Preview extracted sequences", expanded=True):
            for name, seq in list(extracted.items())[:10]:
                short = seq if len(seq) <= 60 else seq[:60] + "…"
                warn = " <span class='chip chip-warn'>short</span>" if len(seq) < 30 else ""
                st.markdown(
                    f"<div style='margin-bottom:6px'>"
                    f"<span style='font-family:IBM Plex Mono,monospace;font-size:0.78rem;color:#58a6ff'>{name}</span>"
                    f"<span style='color:#8b949e;font-size:0.72rem'> · {len(seq)} aa</span>{warn}<br>"
                    f"<span style='font-family:IBM Plex Mono,monospace;font-size:0.76rem;color:#8b949e'>{short}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            if len(extracted) > 10:
                st.markdown(
                    f"<p style='color:#8b949e;font-size:0.78rem'>… and {len(extracted)-10} more</p>",
                    unsafe_allow_html=True,
                )

        # Download
        st.markdown("#### Download")
        zip_bytes = make_zip(extracted, ref_name, rs, re_)

        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button(
                label=f"⬇️  Download all ({len(extracted)} JSONs + FASTA)",
                data=zip_bytes,
                file_name=f"af_region_{rs}-{re_}.zip",
                mime="application/zip",
            )
        with dl_col2:
            # Single JSON for reference sequence only
            ref_seq = extracted.get(ref_name, "")
            if ref_seq:
                ref_json = json.dumps(build_af_json(f"{ref_name}_r{rs}-{re_}", ref_seq), indent=2)
                st.download_button(
                    label=f"⬇️  Reference only ({ref_name})",
                    data=ref_json,
                    file_name=f"{sanitize_name(ref_name)}_r{rs}-{re_}.json",
                    mime="application/json",
                )

        st.markdown(
            "<p style='color:#8b949e;font-size:0.78rem;margin-top:0.5rem'>"
            "ZIP contains one <code>.json</code> per sequence "
            "and a <code>sequences.fasta</code>."
            "</p>",
            unsafe_allow_html=True,
        )

else:
    st.markdown(
        "<div class='card' style='text-align:center;padding:2.5rem;color:#8b949e'>"
        "<div style='font-size:2rem;margin-bottom:0.5rem'>📂</div>"
        "<div style='font-family:IBM Plex Mono,monospace;font-size:0.85rem'>"
        "Upload a <code>.aln</code>, <code>.fasta</code>, or <code>.fas</code> file to get started"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

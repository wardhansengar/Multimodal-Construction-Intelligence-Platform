"""
Multimodal Construction Intelligence Platform
Skill Wallet Capstone Project 1  |  Python 3.9+  |  Streamlit
Gemini is called over REST (no heavy SDK) -> 1-click Streamlit Cloud deploy, no database.
"""
import base64, io, json, random, urllib.parse
from datetime import datetime

import requests
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Construction Intelligence Platform", page_icon="🏗️", layout="wide")

# ───────────────────────── Styling ─────────────────────────
st.markdown("""
<style>
.hero{background:linear-gradient(135deg,#0f2027,#203a43,#f5a623 160%);padding:1.4rem 1.8rem;border-radius:16px;color:#fff;margin-bottom:1rem}
.hero h1{margin:0;font-size:1.9rem}.hero p{margin:.3rem 0 0;opacity:.85}
.card{background:var(--secondary-background-color);border:1px solid rgba(128,128,128,.25);border-radius:14px;padding:1rem 1.2rem;text-align:center}
.card .v{font-size:1.9rem;font-weight:700}.card .l{font-size:.8rem;opacity:.7;text-transform:uppercase;letter-spacing:.05em}
.pill{display:inline-block;padding:.2rem .8rem;border-radius:999px;font-weight:600;font-size:.85rem;color:#fff}
.alert{border-left:6px solid;border-radius:8px;padding:.7rem 1rem;margin:.5rem 0;background:var(--secondary-background-color)}
</style>""", unsafe_allow_html=True)

COLORS = {"Low": "#2e9e5b", "Medium": "#e6a100", "High": "#e8590c", "Critical": "#c92a2a",
          "Compliant": "#2e9e5b", "Needs Improvement": "#e6a100", "Non-Compliant": "#c92a2a",
          "Safe": "#2e9e5b", "Monitor": "#e6a100", "Repair Soon": "#e8590c", "Immediate": "#c92a2a"}

def pill(text):
    return f"<span class='pill' style='background:{COLORS.get(text, '#555')}'>{text}</span>"

def card(label, value):
    return f"<div class='card'><div class='v'>{value}</div><div class='l'>{label}</div></div>"

# ───────────────────────── Gemini client ─────────────────────────
API = "https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent"

def _post(key, model, body):
    r = requests.post(API.format(m=model), params={"key": key}, json=body, timeout=90)
    if r.status_code != 200:
        raise RuntimeError(f"Gemini API error {r.status_code}: {r.text[:300]}")
    data = r.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        raise RuntimeError("Gemini returned no content (possibly blocked by safety filters).")

@st.cache_data(ttl=3600, show_spinner=False)
def list_models(key):
    """Return vision/text-capable Gemini models available to this key (newest first)."""
    try:
        r = requests.get("https://generativelanguage.googleapis.com/v1beta/models", params={"key": key, "pageSize": 200}, timeout=20)
        r.raise_for_status()
        names = [m["name"].split("/")[-1] for m in r.json().get("models", [])
                 if "generateContent" in m.get("supportedGenerationMethods", []) and m["name"].split("/")[-1].startswith("gemini")
                 and not any(x in m["name"] for x in ("tts", "image", "embedding", "live", "robotics", "audio"))]
        return sorted(set(names), reverse=True)
    except Exception:
        return []

def gemini_vision_json(key, model, prompt, img_bytes, mime):
    body = {"contents": [{"parts": [{"text": prompt},
                                    {"inline_data": {"mime_type": mime, "data": base64.b64encode(img_bytes).decode()}}]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0.2}}
    txt = _post(key, model, body).strip()
    if txt.startswith("```"):
        txt = txt.strip("`").removeprefix("json").strip()
    return json.loads(txt)

def gemini_chat(key, model, history, system):
    contents = [{"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]} for m in history]
    body = {"systemInstruction": {"parts": [{"text": system}]}, "contents": contents,
            "generationConfig": {"temperature": 0.5}}
    return _post(key, model, body)

def prep_image(upload):
    img = Image.open(upload).convert("RGB")
    img.thumbnail((1600, 1600))
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=88)
    return img, buf.getvalue()

# ───────────────────────── Prompts (JSON contracts) ─────────────────────────
PPE_PROMPT = """You are a certified construction safety officer (OSHA 29 CFR 1926 and IS 456:2000 / IS 3764 site practice).
Analyze the site photo for worker PPE compliance (helmet, hi-vis vest, steel-toe boots, safety harness) and site hazards.
Return ONLY JSON matching exactly:
{"safety_score":int 0-100,"overall_status":"Compliant"|"Needs Improvement"|"Non-Compliant",
"workers_detected":int,
"ppe_detection":{"helmet":{"detected":bool,"note":str},"hi_vis_vest":{"detected":bool,"note":str},
"steel_toe_boots":{"detected":bool,"note":str},"safety_harness":{"detected":bool,"note":str}},
"hazards":[{"hazard":str,"severity":"Low"|"Medium"|"High"|"Critical","standard":str,"remedy":str}],
"summary":str}
If no workers are visible, set workers_detected 0 and judge the site hazards only. Do not invent details."""

CRACK_PROMPT = """You are a structural engineer (IS 456:2000, IS 15988, IS 13935). Analyze this concrete/wall surface image.
Return ONLY JSON matching exactly:
{"defect_type":"Hairline"|"Settlement"|"Structural Shear"|"Thermal Expansion"|"Spalling"|"No Defect",
"confidence":int 0-100,"estimated_width_mm":number,"pattern":str,"root_causes":[str],
"recommended_repairs":[{"method":str,"details":str}],
"urgency":"Safe"|"Monitor"|"Repair Soon"|"Immediate","risk_summary":str,"follow_up_tests":[str]}
Crack width is an estimate from visual cues only; state that in risk_summary."""

CHAT_SYSTEM = """You are an expert Indian construction engineering site assistant. Answer concisely and practically, citing IS codes
(IS 456, IS 800, IS 1786, IS 269, IS 383, IS 4326) and OSHA where relevant. Prioritise safety. If uncertain, say a licensed
structural engineer must verify. Current site weather: {w}."""

# ───────────────────────── Demo-mode data ─────────────────────────
DEMO_PPE = {"safety_score": 62, "overall_status": "Needs Improvement", "workers_detected": 3,
    "ppe_detection": {"helmet": {"detected": True, "note": "2 of 3 workers wearing helmets"},
        "hi_vis_vest": {"detected": True, "note": "All workers visible in vests"},
        "steel_toe_boots": {"detected": False, "note": "Footwear not verifiable / sneakers observed"},
        "safety_harness": {"detected": False, "note": "No harness at elevated edge work"}},
    "hazards": [
        {"hazard": "Worker at open slab edge without fall protection", "severity": "Critical", "standard": "OSHA 1926.501", "remedy": "Stop work; install guardrails and mandate full-body harness with anchor points."},
        {"hazard": "Unprotected rebar ends", "severity": "High", "standard": "OSHA 1926.701(b)", "remedy": "Fit mushroom rebar caps on all protruding bars."},
        {"hazard": "Loose material stacked near edge", "severity": "Medium", "standard": "OSHA 1926.250", "remedy": "Relocate and secure stacks; install toe boards."}],
    "summary": "[DEMO] Sample audit. Add a Gemini API key for real image analysis."}
DEMO_CRACK = {"defect_type": "Structural Shear", "confidence": 78, "estimated_width_mm": 2.5,
    "pattern": "Diagonal 45° stair-step crack near beam-column joint",
    "root_causes": ["Excess shear demand / inadequate stirrup spacing", "Differential loading", "Possible foundation movement"],
    "recommended_repairs": [{"method": "Epoxy injection", "details": "Low-viscosity epoxy pressure injection per IS 15988 after cleaning the crack."},
                            {"method": "Structural re-grouting", "details": "Non-shrink cementitious grout; add FRP wrap or jacketing if rebar exposed."}],
    "urgency": "Immediate", "risk_summary": "[DEMO] Width estimated visually; a site measurement with a crack gauge is required.",
    "follow_up_tests": ["Rebound hammer", "UPV test", "Crack-width monitoring (tell-tales) for 28 days"]}

DEMO_CHAT = {"curing": "Cure concrete for min. 7 days (OPC) / 10 days (PPC/blended) per IS 456 cl. 13.5. Use wet hessian or ponding; protect from wind and sun for the first 24 hours.",
    "slump": "Typical slump: 25–75 mm for beams/slabs lightly reinforced, 75–100 mm for columns, 100–150 mm for pumped concrete (IS 456 cl. 7.1 guidance).",
    "m20": "M20 nominal mix is roughly 1 : 1.5 : 3 (cement : sand : aggregate) with w/c about 0.5; design mixes are preferred for M25+.",
    "ppe": "Minimum PPE: IS 2925 helmet, hi-vis vest, IS 15298 safety shoes, gloves, eye protection; full-body harness (IS 3521) above 1.8 m."}

# ───────────────────────── Valuation model ─────────────────────────
COST_PSF = {"Residential (RCC Frame)": 2200, "Commercial Building": 2800, "Industrial Shed (PEB)": 1800,
            "Villa / Premium": 3500, "Warehouse": 1600}
TIER_MULT = {"Tier 1 (Mumbai, Delhi, Bengaluru)": 1.35, "Tier 2 (Pune, Indore, Jaipur)": 1.0, "Tier 3 (Bhopal, Raipur, Nashik)": 0.82}
MARKET_PSF = {"Tier 1 (Mumbai, Delhi, Bengaluru)": 9500, "Tier 2 (Pune, Indore, Jaipur)": 5200, "Tier 3 (Bhopal, Raipur, Nashik)": 3400}
STEEL_KG_PSF = {"Residential (RCC Frame)": 4.0, "Commercial Building": 5.0, "Industrial Shed (PEB)": 8.0, "Villa / Premium": 4.5, "Warehouse": 6.0}
MILESTONES = [(0, "Site prep"), (10, "Foundation"), (30, "Plinth & columns"), (50, "Slab & structure"), (70, "Masonry & MEP"), (90, "Finishing"), (100, "Handover")]

def inr(v):
    return f"₹{v/1e7:,.2f} Cr" if v >= 1e7 else f"₹{v/1e5:,.2f} Lakhs"

def valuation(stype, sqft, tier, pct):
    budget = COST_PSF[stype] * TIER_MULT[tier] * sqft
    capex = budget * pct / 100
    market_full = MARKET_PSF[tier] * sqft * (1.25 if "Villa" in stype else 1.0)
    # asset value: land+structure recognised proportionally, floor 30% of finished value for land share
    asset = market_full * (0.30 + 0.70 * pct / 100)
    steel_t = STEEL_KG_PSF[stype] * sqft / 1000
    cement_bags = 0.4 * sqft * (1.1 if "Industrial" not in stype else 0.7)
    return dict(budget=budget, capex=capex, market_full=market_full, asset=asset, steel_t=steel_t, cement_bags=cement_bags)

WEATHER = {
 "Normal": ("#2e9e5b", ["Standard daily toolbox talk and PPE checks.", "Maintain routine housekeeping and material stacking limits."]),
 "High Wind": ("#c92a2a", ["🚫 Suspend crane and hoisting operations when sustained wind exceeds ~36 km/h (OSHA 1926.1417 / manufacturer limit).",
     "Secure scaffolding sheeting, formwork, and loose sheets; tie down stacked materials.", "Halt work at height and on exposed edges; evacuate tower-crane radius."]),
 "Heavy Rain": ("#e8590c", ["⚡ Electrical hazard: de-energise temporary supply, use ELCB/RCCB-protected GFCI circuits, keep joints off the ground.",
     "Cover fresh concrete; avoid pouring during downpour; check slump and w/c ratio after rain on aggregates.",
     "Inspect excavation and trench shoring for water ingress; pump out sumps.", "Anti-slip checks on ramps, ladders, and scaffolds."]),
 "Extreme Heat": ("#e6a100", ["🌡️ Concrete curing: pour early morning/night, use chilled water/ice, retarders; start curing immediately (IS 456 cl. 13.5), keep surface continuously moist.",
     "Heat stress: mandatory water-rest-shade cycles, ORS availability, rotate workers every 45–60 min.", "Avoid steel handling at midday; watch for formwork thermal cracking."]),
}

# ───────────────────────── Sidebar ─────────────────────────
with st.sidebar:
    st.title("🏗️ ConstructIQ")
    key_in = st.text_input("Google Gemini API key", type="password", help="Never stored; lives only in this session.")
    api_key = key_in.strip()
    if not api_key:
        try:
            api_key = st.secrets.get("GEMINI_API_KEY", "")
        except Exception:
            api_key = ""
    DEMO = not api_key
    available = list_models(api_key) if api_key else []
    if available:
        flash = [m for m in available if "flash" in m and "lite" not in m]
        model = st.selectbox("Model", available, index=available.index(flash[0]) if flash else 0,
                             help="Loaded live from your API key, so it never goes out of date.")
    else:
        model = st.text_input("Model", value="gemini-3.8-flash", help="Could not list models; type a model name manually.")
    if DEMO:
        st.warning("🟡 Demonstration mode (no API key)")
    else:
        st.success("🟢 Live Gemini mode")
    st.markdown("---")
    st.subheader("Navigation")
    st.markdown("1. 🦺 Safety & PPE Audit\n2. 🧱 Crack Diagnostics\n3. 💰 Planning & Valuation\n4. 🌦️ Site Assistant")
    with st.expander("Architecture overview"):
        st.code("Streamlit UI\n ├─ Pillow (image prep)\n ├─ Gemini REST (multimodal)\n │    └─ JSON data contracts\n ├─ Valuation engine (pure Python)\n └─ Pollinations (3D concept)\n\nNo database · session_state only", language="text")
    st.caption("Skill Wallet Capstone Project 1")

st.markdown("<div class='hero'><h1>🏗️ Multimodal Construction Intelligence Platform</h1>"
            "<p>AI-powered safety auditing, structural diagnostics, project valuation and site advisory.</p></div>", unsafe_allow_html=True)
if DEMO:
    st.info("Demonstration mode: analysis tabs return illustrative sample results. Add a Gemini API key in the sidebar for real analysis.")

def run_analysis(prompt, demo_data, upload):
    img, raw = prep_image(upload)
    st.image(img, use_container_width=True)
    if DEMO:
        return demo_data
    return gemini_vision_json(api_key, model, prompt, raw, "image/jpeg")

tab1, tab2, tab3, tab4 = st.tabs(["🦺 Safety & PPE Audit", "🧱 Crack Diagnostics", "💰 Planning & Valuation", "🌦️ Site Assistant"])

# ───────────────────────── Tab 1 ─────────────────────────
with tab1:
    st.subheader("Site Safety & PPE Audit")
    up = st.file_uploader("Upload site photo", type=["jpg", "jpeg", "png"], key="ppe_up")
    if up and st.button("Run safety audit", type="primary", key="ppe_btn"):
        try:
            with st.spinner("Analyzing site..."):
                st.session_state.ppe = run_analysis(PPE_PROMPT, DEMO_PPE, up)
        except json.JSONDecodeError:
            st.error("The model returned malformed JSON. Please retry.")
        except Exception as e:
            st.error(f"Analysis failed: {e}")
    r = st.session_state.get("ppe")
    if r:
        try:
            hz = r.get("hazards", [])
            c = st.columns(4)
            c[0].markdown(card("Safety Score", f"{r.get('safety_score', 0)}/100"), unsafe_allow_html=True)
            c[1].markdown(card("Status", pill(r.get("overall_status", "N/A"))), unsafe_allow_html=True)
            c[2].markdown(card("Workers", r.get("workers_detected", "–")), unsafe_allow_html=True)
            c[3].markdown(card("Hazards", len(hz)), unsafe_allow_html=True)
            st.progress(max(0, min(100, int(r.get("safety_score", 0)))) / 100)
            st.markdown(f"**Summary:** {r.get('summary', '')}")
            st.markdown("##### PPE checklist")
            names = {"helmet": "⛑️ Helmet", "hi_vis_vest": "🦺 Hi-vis vest", "steel_toe_boots": "🥾 Steel-toe boots", "safety_harness": "🪢 Safety harness"}
            for k, label in names.items():
                item = r.get("ppe_detection", {}).get(k, {})
                st.markdown(f"{'✅' if item.get('detected') else '❌'} **{label}** — {item.get('note', 'n/a')}")
            st.markdown("##### Identified hazards")
            for h in sorted(hz, key=lambda x: ["Critical", "High", "Medium", "Low"].index(x.get("severity", "Low"))
                            if x.get("severity") in COLORS else 9):
                col = COLORS.get(h.get("severity"), "#555")
                st.markdown(f"<div class='alert' style='border-color:{col}'>{pill(h.get('severity', '?'))} <b>{h.get('hazard')}</b>"
                            f"<br><small>📘 {h.get('standard', '')}</small><br>🛠️ {h.get('remedy', '')}</div>", unsafe_allow_html=True)
            st.download_button("⬇️ Download JSON report", json.dumps({"generated": datetime.now().isoformat(), "report": r}, indent=2),
                               "ppe_safety_report.json", "application/json")
        except Exception as e:
            st.error(f"Could not render report: {e}")

# ───────────────────────── Tab 2 ─────────────────────────
with tab2:
    st.subheader("Crack Diagnostics & Structural Integrity")
    up2 = st.file_uploader("Upload concrete / wall surface photo", type=["jpg", "jpeg", "png"], key="crack_up")
    if up2 and st.button("Diagnose defect", type="primary", key="crack_btn"):
        try:
            with st.spinner("Diagnosing..."):
                st.session_state.crack = run_analysis(CRACK_PROMPT, DEMO_CRACK, up2)
        except json.JSONDecodeError:
            st.error("The model returned malformed JSON. Please retry.")
        except Exception as e:
            st.error(f"Diagnosis failed: {e}")
    d = st.session_state.get("crack")
    if d:
        try:
            c = st.columns(4)
            c[0].markdown(card("Defect", d.get("defect_type", "–")), unsafe_allow_html=True)
            c[1].markdown(card("Width (est.)", f"{d.get('estimated_width_mm', 0)} mm"), unsafe_allow_html=True)
            c[2].markdown(card("Confidence", f"{d.get('confidence', 0)}%"), unsafe_allow_html=True)
            c[3].markdown(card("Urgency", pill(d.get("urgency", "N/A"))), unsafe_allow_html=True)
            st.markdown(f"**Pattern:** {d.get('pattern', '')}")
            l, rr = st.columns(2)
            with l:
                st.markdown("##### Root causes")
                for x in d.get("root_causes", []): st.markdown(f"- {x}")
                st.markdown("##### Follow-up tests")
                for x in d.get("follow_up_tests", []): st.markdown(f"- {x}")
            with rr:
                st.markdown("##### Recommended repairs")
                for x in d.get("recommended_repairs", []): st.markdown(f"**{x.get('method')}** — {x.get('details')}")
            st.warning(d.get("risk_summary", ""))
            st.download_button("⬇️ Download JSON report", json.dumps(d, indent=2), "crack_diagnostic_report.json", "application/json")
        except Exception as e:
            st.error(f"Could not render diagnosis: {e}")

# ───────────────────────── Tab 3 ─────────────────────────
with tab3:
    st.subheader("Project Planning & Real Estate Valuation")
    a, b = st.columns(2)
    stype = a.selectbox("Structure type", list(COST_PSF))
    tier = b.selectbox("Location tier", list(TIER_MULT))
    sqft = a.number_input("Built-up area (sqft)", 200, 500000, 1500, step=100)
    pct = b.slider("Completion %", 0, 100, 40)
    v = valuation(stype, sqft, tier, pct)
    c = st.columns(4)
    c[0].markdown(card("Total budget", inr(v["budget"])), unsafe_allow_html=True)
    c[1].markdown(card("CapEx incurred", inr(v["capex"])), unsafe_allow_html=True)
    c[2].markdown(card("Current asset value", inr(v["asset"])), unsafe_allow_html=True)
    c[3].markdown(card("Market value (complete)", inr(v["market_full"])), unsafe_allow_html=True)
    c = st.columns(2)
    c[0].markdown(card("Steel requirement", f"{v['steel_t']:,.1f} tonnes"), unsafe_allow_html=True)
    c[1].markdown(card("Cement requirement", f"{v['cement_bags']:,.0f} bags (50 kg)"), unsafe_allow_html=True)
    st.markdown("##### Milestone progress")
    st.progress(pct / 100)
    current = max((m for m in MILESTONES if m[0] <= pct), key=lambda m: m[0])
    st.caption(" → ".join(f"**{n}**" if (p, n) == current else n for p, n in MILESTONES))
    st.caption("Indicative estimates using simplified per-sqft rates; not a certified valuation or BOQ.")

    st.markdown("##### 🎨 Conceptual 3D architectural rendering (Pollinations AI)")
    style = st.text_input("Style notes", "modern, glass facade, landscaped garden, golden hour")
    seed = st.session_state.setdefault("seed", 42)
    ptxt = f"photorealistic 3D architectural rendering of a {stype.lower()}, {sqft} sqft, Indian {tier.split(' (')[0]} city, {style}, ultra detailed"
    if st.button("Generate / regenerate concept"):
        st.session_state.seed = random.randint(1, 99999); st.session_state.show_img = True
    st.text_area("Generated prompt", ptxt, height=70)
    if st.session_state.get("show_img"):
        url = f"https://image.pollinations.ai/prompt/{urllib.parse.quote(ptxt)}?width=1024&height=640&nologo=true&seed={st.session_state.seed}"
        st.image(url, caption="AI concept — illustrative only", use_container_width=True)

# ───────────────────────── Tab 4 ─────────────────────────
with tab4:
    st.subheader("Daily Site Assistant & Weather-Aware Advisory")
    w = st.radio("Today's weather", list(WEATHER), horizontal=True)
    col, items = WEATHER[w]
    for it in items:
        st.markdown(f"<div class='alert' style='border-color:{col}'>{it}</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("##### 💬 Construction engineering assistant")
    hist = st.session_state.setdefault("chat", [{"role": "assistant", "content": "Hi! Ask me about concrete, rebar, scaffolding, IS codes, scheduling or site safety."}])
    for m in hist:
        with st.chat_message(m["role"]): st.markdown(m["content"])
    if q := st.chat_input("Ask a site engineering question..."):
        hist.append({"role": "user", "content": q})
        with st.chat_message("user"): st.markdown(q)
        with st.chat_message("assistant"):
            try:
                if DEMO:
                    ans = next((v_ for k, v_ in DEMO_CHAT.items() if k in q.lower()),
                               "[Demo] I can answer sample questions on **curing**, **slump**, **M20 mix**, and **PPE**. Add a Gemini key for full answers.")
                else:
                    with st.spinner("Thinking..."):
                        ans = gemini_chat(api_key, model, hist, CHAT_SYSTEM.format(w=w))
            except Exception as e:
                ans = f"⚠️ Assistant error: {e}"
            st.markdown(ans)
        hist.append({"role": "assistant", "content": ans})
    if st.button("Clear chat"):
        st.session_state.pop("chat", None); st.rerun()

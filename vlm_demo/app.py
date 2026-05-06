"""
ARIA — AI Risk & Incident Analyzer
VLM Reasoning Layer for Industrial Hazard Recognition
CIVS × Purdue University Northwest · SDI Butler Steel Plant

Main Streamlit demo application.
Provides a chat-style interface for querying VLMs about industrial hazard footage.
Supports three VLM backends: Gemini 2.0 Flash, GPT-4o, and local Gemma 3 27B via vLLM.

Demonstrates the domain gap effect: same model, same zero-shot setup,
dramatically different hazard reasoning with vs without plant-specific context.

Part of the research:
  "Bridging the Domain Gap in Industrial Safety with Vision Language Models"
  CIVS × Purdue University Northwest for SDI Butler Steel Plant
"""

import base64, os
import streamlit as st
from pathlib import Path

# Optional: set EGO_VIDEO_PATH to enable the egocentric video demo clip.
# Example: export EGO_VIDEO_PATH=/path/to/your/video.mp4
_EGO_VIDEO_PATH = os.environ.get("EGO_VIDEO_PATH", "")

DEMO_DIR   = Path(__file__).parent
BASE_DIR   = DEMO_DIR.parent
CLIPS_DIR  = DEMO_DIR / "clips"
FRAMES_DIR = DEMO_DIR / "hero_frames"

# ── SDI Plant clips (the real story) + general hazard clips ───────────────────
CLIPS = [
    # ── SDI Steel Plant footage ──────────────────────────────────────────────
    {
        "id":       "sdi_hero",
        "label":    "⭐  SDI Plant — Both Vehicles + Glowing Pot (Hero Moment)",
        "video":    "sdi_hero_both_vehicles.mp4",
        "frame":    FRAMES_DIR / "sdi_hero_t209.jpg",
        "section":  "SDI Steel Plant — Live Footage",
        "context":  (
            "You are analyzing Camera 2 (south-facing wide view) of the SDI Butler steel plant bay. "
            "This is the most important camera — it captures both the north and south bay simultaneously. "
            "At this moment (approx. t=209s), TWO vehicles are simultaneously active: "
            "(1) a pot hauler entering from the north, currently carrying a GLOWING MOLTEN METAL POT — "
            "visible as an orange/red heat signature, and "
            "(2) a separate vehicle operating on the south side of the bay. "
            "A pot hauler is a large, heavy vehicle that transports crucibles of liquid steel at extreme temperatures (1500°C+). "
            "The bay uses a zone system: RED ZONES are active operation areas where personnel entry is restricted. "
            "Physical swing-gate blockers must be confirmed in place for a zone to be SAFE. "
            "The existing YOLO system has already detected the vehicles and issued a zone alert. "
            "Your role as Stage 2 is to explain WHAT is happening, WHY it is dangerous, and WHAT action is required."
        ),
        "questions": [
            "What is the most dangerous thing happening in this scene right now?",
            "Two vehicles are active simultaneously — what are the collision and thermal risks?",
            "The north vehicle is carrying a molten metal pot. What hazards does that create?",
            "The YOLO system flagged this. What would you tell the safety officer in plain English?",
            "What would happen if a worker entered the bay right now?",
            "Rate the combined risk level and explain your reasoning.",
        ],
    },
    {
        "id":       "sdi_blindspot_entry",
        "label":    "SDI Plant — Vehicle Entering Camera Blind Spot",
        "video":    "sdi_blindspot_entry.mp4",
        "frame":    FRAMES_DIR / "sdi_blindspot_entry_t100.jpg",
        "section":  "SDI Steel Plant — Live Footage",
        "context":  (
            "You are analyzing Camera 3 (north bay, alternative angle) of the SDI Butler steel plant. "
            "Camera 3 exists specifically to cover a BLIND SPOT that Camera 1 cannot see. "
            "At this point in the footage (t=100s), a pot hauler is entering or moving through "
            "the area that falls outside Camera 1's field of view. "
            "The vehicle is currently not carrying a molten pot. "
            "This blind spot scenario is a key safety concern: the existing YOLO system on Camera 1 "
            "loses track of this vehicle during this window. Camera 3 is the only coverage. "
            "Note what the vehicle is doing and assess the risk of the blind spot itself."
        ),
        "questions": [
            "What is the vehicle doing and where is it heading?",
            "Why is a camera blind spot dangerous in this kind of plant environment?",
            "The YOLO system on Camera 1 has lost this vehicle. What should the safety system do?",
            "What would you recommend to eliminate this blind spot risk?",
            "Is the scene currently SAFE or UNSAFE? Give a one-line verdict first.",
        ],
    },
    {
        "id":       "sdi_blindspot_pot",
        "label":    "SDI Plant — Vehicle Re-emerges from Blind Spot with Glowing Pot",
        "video":    "sdi_blindspot_pot.mp4",
        "frame":    FRAMES_DIR / "sdi_blindspot_pot_t204.jpg",
        "section":  "SDI Steel Plant — Live Footage",
        "context":  (
            "You are analyzing Camera 3 of the SDI Butler steel plant. "
            "This is the critical moment: the same vehicle that entered the camera blind spot "
            "at t=100s has now re-emerged at t=204s — and it is NOW CARRYING A GLOWING MOLTEN METAL POT. "
            "The pot contains liquid steel at approximately 1500°C. The orange/red glow is the heat signature. "
            "The vehicle was INVISIBLE to the main monitoring system during the approximately 100-second gap. "
            "This gap — a vehicle disappearing, loading molten metal, and reappearing — "
            "is precisely the safety scenario that traditional YOLO-only systems cannot fully capture. "
            "Assess the current hazard and the significance of the 100-second visibility gap."
        ),
        "questions": [
            "What changed between when this vehicle entered the blind spot and now?",
            "What does the glowing pot tell you about the hazard level?",
            "The monitoring system had a 100-second gap in tracking this vehicle. What is the risk of that?",
            "How does this scene demonstrate the limitation of camera-only YOLO systems?",
            "What immediate actions should be triggered right now?",
            "Could a worker have entered the bay during that blind spot window without the system knowing?",
        ],
    },
    {
        "id":       "sdi_pot_north",
        "label":    "SDI Plant — Pot Hauler Returning North with Molten Load",
        "video":    "sdi_pot_hauler_north.mp4",
        "frame":    FRAMES_DIR / "sdi_pot_hauler_t240.jpg",
        "section":  "SDI Steel Plant — Live Footage",
        "context":  (
            "You are analyzing Camera 1 (north end of the SDI Butler plant bay). "
            "At t=240s, a pot hauler is returning north through the bay — this time carrying "
            "a glowing molten metal pot (liquid steel at ~1500°C). "
            "The pot hauler is a large vehicle that moves along a fixed path through the bay. "
            "When carrying a molten load, the risk profile changes dramatically: "
            "any collision, sudden stop, or equipment failure risks a molten metal spill — "
            "an immediately fatal event for anyone in proximity. "
            "The RED ZONE around the vehicle's path must be clear of all personnel."
        ),
        "questions": [
            "What is this vehicle carrying and why does that make this scene critical?",
            "What is the exclusion zone that should be enforced around a loaded pot hauler?",
            "How does the risk differ between an empty pot hauler vs a loaded one?",
            "What sensors or systems should be active when a molten pot is in transit?",
            "If a person were detected in this frame, what would the severity be?",
        ],
    },
    # ── General hazard footage (Phase 1 baseline) ────────────────────────────
    {
        "id":       "factory_machinery",
        "label":    "Factory Floor — Unguarded Machinery (CCTV)",
        "video":    "factory_floor_machinery.mp4",
        "frame":    BASE_DIR / "phase1_opensource/machinery_danger/frames/machinery_unguarded_equipment_t0010.00s_f000300.jpg",
        "section":  "General Hazard Footage — Baseline",
        "context":  (
            "Real CCTV footage from a manufacturing facility recorded on 08-09-2018. "
            "Multiple workers are present on an active factory floor with CNC machines and industrial equipment. "
            "This is Phase 1 baseline footage — clear, well-lit, general industry setting. "
            "Assess any visible hazards, machinery guarding issues, and worker proximity risks."
        ),
        "questions": [
            "What hazards are visible on this factory floor?",
            "Are there any machinery guarding violations?",
            "What PPE should workers be wearing here?",
            "How does this scene compare to a steel plant in terms of hazard complexity?",
        ],
    },
    {
        "id":       "construction_fall",
        "label":    "Construction Site — Fall Hazard (CCTV)",
        "video":    "construction_fall_cctv.mp4",
        "frame":    BASE_DIR / "phase1_opensource/fall_from_height/frames/fall_construction_cctv_t0015.00s_f000450.jpg",
        "section":  "General Hazard Footage — Baseline",
        "context":  (
            "Real CCTV footage from an active construction site. "
            "Excavation work is underway with an open pit. A worker is visible near the excavation edge. "
            "Safety cones are present. This is Phase 1 baseline footage for fall hazard detection. "
            "Assess fall risks, OSHA compliance, and required controls."
        ),
        "questions": [
            "What fall hazards are present in this scene?",
            "What OSHA standard applies here and what does it require?",
            "The safety cones are present — does that make this scene SAFE?",
            "What additional controls would you require before work continues?",
        ],
    },
    # ── Egocentric / General video ──────────────────────────────────────────
    # Demonstrates VLM general video understanding beyond industrial safety.
    # Set EGO_VIDEO_PATH env var to enable: export EGO_VIDEO_PATH=/path/to/video.mp4
    *([{
        "id":          "ego_general",
        "label":       "🎥 General Video — First-Person / Egocentric",
        "video":       None,
        "video_path":  Path(_EGO_VIDEO_PATH),
        "frame":       None,
        "section":     "General Video Understanding",
        "context":     "First-person egocentric video of a daily activity. Describe what the person is doing, what objects are involved, and what goal they appear to be working toward.",
        "questions": [
            "What is the person doing in this video?",
            "Describe the sequence of actions from start to finish.",
            "What objects or tools are being used?",
            "What environment or setting is this taking place in?",
            "What can you infer about the person's goal or task?",
            "How does this general scene understanding compare to the industrial hazard reasoning above?",
        ],
    }] if _EGO_VIDEO_PATH else []),
]

# ── Generic (no-context) system prompt — used when domain toggle is OFF ────────
BARE_SYSTEM = """\
You are a workplace safety AI. Analyze the image and identify any hazards.
Respond with: HAZARD_DETECTED (YES/NO), SEVERITY (CRITICAL/HIGH/MEDIUM/LOW), CONFIDENCE, \
WHAT_I_SEE (describe exactly what you observe), and HAZARD_ANALYSIS (explain the risk).
Do not assume any domain-specific context. Only describe what you literally see.
"""

# ── Pre-loaded demo responses (real inference data from experiment_queue.csv) ──
# These load instantly when the clip is selected — no API call needed for the
# initial contrast. The user can then ask live follow-up questions.
DEMO_RESPONSES = {
    "sdi_hero": {
        "generic": {
            "q": "Describe the hazard in this scene.",
            "a": (
                "HAZARD_DETECTED: YES\n"
                "SEVERITY: HIGH\n"
                "CONFIDENCE: HIGH\n\n"
                "WHAT_I_SEE:\n"
                "The image shows a cluttered workspace with numerous extension cords and power strips "
                "running across the floor, creating a tripping hazard and potential fire risk.\n\n"
                "HAZARD_ANALYSIS:\n"
                "Multiple extension cords are running across walkways, creating a significant tripping "
                "hazard for employees. The excessive use of extension cords and power strips suggests a "
                "lack of sufficient permanent electrical outlets. This is both a trip hazard and a fire "
                "risk due to potential overloading of circuits.\n\n"
                "*(Source: actual Gemma 3 27B zero-shot response on this frame — no plant context provided)*"
            ),
        },
        "domain": {
            "q": "Describe the hazard in this scene.",
            "a": (
                "**CRITICAL — Two vehicles simultaneously active in the bay.**\n\n"
                "This frame captures the highest-risk scenario in the SDI Butler plant: "
                "two pot haulers operating simultaneously in the same bay. "
                "The north vehicle is carrying a **glowing molten metal pot** (~1500°C heat signature visible "
                "as the orange/red glow). The south vehicle is active concurrently on the opposite bay path.\n\n"
                "**Why this is CRITICAL:**\n"
                "- Multi-vehicle simultaneous operation is the plant's highest-risk configuration\n"
                "- A loaded pot hauler requires full RED ZONE clearance — no crossing traffic\n"
                "- Any collision or sudden stop risks a molten metal spill — immediately fatal to anyone within proximity\n"
                "- Swing-gate blockers on all zone crossing points must be confirmed before either vehicle moves\n\n"
                "**Required action:** STOP WORK ORDER for both vehicles. Confirm zone isolation. "
                "Evacuate bay. Resume only after sequential (not simultaneous) operation is re-established."
            ),
        },
    },
    "sdi_blindspot_entry": {
        "generic": {
            "q": "Describe the hazard in this scene.",
            "a": (
                "HAZARD_DETECTED: YES\n"
                "SEVERITY: MEDIUM\n"
                "CONFIDENCE: HIGH\n\n"
                "WHAT_I_SEE:\n"
                "The image shows a large storage or warehouse area with equipment and materials stacked "
                "along the walls. The area appears dimly lit with some overhead lighting.\n\n"
                "HAZARD_ANALYSIS:\n"
                "The primary hazard is the potential for stacked materials to fall or shift, creating "
                "a risk of injury to workers in the area. The dim lighting also increases the risk of "
                "trips and falls. No immediate life-threatening hazard is visible.\n\n"
                "*(Source: actual Gemma 3 27B zero-shot response on this frame — no plant context provided)*"
            ),
        },
        "domain": {
            "q": "Describe the hazard in this scene.",
            "a": (
                "**UNSAFE — Vehicle entering Camera 1 blind spot. Tracking gap initiated.**\n\n"
                "This vehicle is moving into the zone that falls outside Camera 1's field of view. "
                "As of this moment, the primary YOLO monitoring system has **lost track of this vehicle**. "
                "Camera 3 (this feed) is now the only active coverage.\n\n"
                "**The blind spot risk:**\n"
                "- The vehicle's next action — whether it loads a molten pot, stops, or reverses — "
                "is invisible to the main safety system for the duration of the gap\n"
                "- A worker could enter the bay during this window with no system awareness of the vehicle\n"
                "- The gap window is approximately 100 seconds (observed from this footage)\n\n"
                "**Required action:** Flag blind spot entry in safety log. Lock out personnel bay access "
                "until vehicle re-emerges with confirmed load status. This is the exact scenario "
                "Camera 3 was installed to monitor."
            ),
        },
    },
    "sdi_blindspot_pot": {
        "generic": {
            "q": "Describe the hazard in this scene.",
            "a": (
                "HAZARD_DETECTED: YES\n"
                "SEVERITY: MEDIUM\n"
                "CONFIDENCE: HIGH\n\n"
                "WHAT_I_SEE:\n"
                "The image shows a large, open area with numerous identical workstations, each with a "
                "computer and chair, appearing to be an office or call center environment.\n\n"
                "HAZARD_ANALYSIS:\n"
                "The primary hazard is potential ergonomic issues due to the repetitive nature of work "
                "and the seemingly uniform workstation setup, which may not accommodate individual needs. "
                "There is also visible clutter and cables creating potential trip hazards between workstations. "
                "This could lead to musculoskeletal disorders over time.\n\n"
                "*(Source: actual Gemma 3 27B zero-shot response on this exact frame — no plant context provided)*"
            ),
        },
        "domain": {
            "q": "Describe the hazard in this scene.",
            "a": (
                "**CRITICAL — Molten load in transit directly over personnel. Red light active.**\n\n"
                "This frame shows a pot hauler that has **re-emerged from the Camera 1 blind spot** "
                "now carrying a glowing molten metal pot (~1500°C). During the ~100-second blind spot "
                "window, the vehicle loaded liquid steel — invisible to the primary monitoring system.\n\n"
                "Critically: **two personnel are standing directly beneath the load**. "
                "The red ground light is illuminated, confirming this is a live hazard zone.\n\n"
                "**Hazards present:**\n"
                "- Personnel in the active load path — risk of fatal thermal injury from any spill\n"
                "- Red light indicator active — zone is marked UNSAFE, personnel should not be here\n"
                "- 100-second visibility gap: system had no record of this vehicle loading\n\n"
                "**Required action:**\n"
                "1. IMMEDIATE emergency stop on the pot hauler\n"
                "2. Evacuate all personnel from beneath the load path\n"
                "3. Investigate why personnel entered an active red-light zone\n"
                "4. Review swing-gate blocker deployment for this bay\n\n"
                "*(Source: actual Gemma 3 27B domain-aware response on this frame)*"
            ),
        },
    },
    "sdi_pot_north": {
        "generic": {
            "q": "Describe the hazard in this scene.",
            "a": (
                "HAZARD_DETECTED: YES\n"
                "SEVERITY: MEDIUM\n"
                "CONFIDENCE: HIGH\n\n"
                "WHAT_I_SEE:\n"
                "The image shows a large, open area with numerous identical workstations, each with a "
                "computer and chair, and a floor covered in loose cables and wires.\n\n"
                "HAZARD_ANALYSIS:\n"
                "The most prominent hazard is a trip and fall risk due to the extensive network of "
                "cables strewn across the floor. These cables create an uneven walking surface and "
                "could easily cause someone to stumble. There is no immediate life-threatening hazard visible.\n\n"
                "*(Source: actual Gemma 3 27B zero-shot response on this frame — no plant context provided)*"
            ),
        },
        "domain": {
            "q": "Describe the hazard in this scene.",
            "a": (
                "**CRITICAL — Active pot hauler in transit northbound carrying molten load.**\n\n"
                "The pot hauler is returning north through the bay with a glowing molten pot — "
                "liquid steel at approximately 1500°C. The heat signature (orange/red glow) confirms "
                "an active load is in transit.\n\n"
                "**Risk profile for a loaded transit:**\n"
                "- Any person within the vehicle's path is at risk of fatal thermal injury in the "
                "event of a spill, sudden stop, or equipment failure\n"
                "- Required exclusion zone: minimum 10 meters from vehicle path, all crossing points blocked\n"
                "- Swing-gate blockers must be confirmed deployed on all zone crossings before movement\n\n"
                "**Current status:** If a person were detected in this frame, severity would escalate "
                "to CRITICAL IMMEDIATE — stop work, emergency halt, evacuation. "
                "Frame appears clear of personnel — but zone lockout must remain active until vehicle "
                "reaches the furnace and load is transferred."
            ),
        },
    },
}

# ── SDI-specific system prompt ─────────────────────────────────────────────────
SDI_SYSTEM = """\
You are ARIA (AI Risk & Incident Analyzer), the Stage 2 reasoning layer of an industrial \
safety system deployed at SDI (Steel Dynamics Inc.) Butler steel plant, built by CIVS at \
Purdue University Northwest.

YOUR ROLE IN THE ARCHITECTURE:
- Stage 1 (already running): A YOLO + DeepSORT system detects vehicles, personnel, and zone violations in real time.
- Stage 2 (you): You receive the frame that triggered the alert and provide human-readable explanation, \
  severity assessment, and recommended action.

PLANT KNOWLEDGE YOU HAVE:
- Pot haulers: Large heavy vehicles that transport crucibles (pots) of liquid steel at ~1500°C. \
  A glowing orange/red object in the frame is a hot pot. This is the highest-risk load in the plant.
- Bay zones: The bay is divided into RED ZONES (active operation, restricted) and GREEN ZONES (safe corridors). \
  Physical swing-gate blockers must be confirmed in place for a zone to be considered safe.
- Red light on the ground: Means UNSAFE regardless of everything else.
- Blind spots: Camera 3 covers a blind spot that Camera 1 misses. A vehicle disappearing \
  from Camera 1 and reappearing on Camera 3 with a molten load is a critical safety gap.
- Multi-vehicle simultaneous operation is the highest-risk scenario in this bay.

HOW YOU RESPOND:
- Lead with a one-line verdict: SAFE / UNSAFE / CRITICAL
- Then explain specifically what you see in the frame
- Name the hazard precisely — "pot hauler carrying molten load in transit through active zone" \
  not "there is a vehicle and it looks dangerous"
- State what action should be triggered
- Be direct — your audience is plant managers and safety officers, not academics
- If you cannot clearly see something in the frame, say so
"""

GENERAL_SYSTEM = """\
You are a general-purpose video understanding assistant.

Your task is to describe and explain what is happening in the video as clearly and naturally as possible.

- Describe the scene, setting, and key objects or people you observe.
- Explain the actions, activities, or events that are taking place across the video frames.
- Note any interesting details, context clues, or temporal progression you can infer.
- If there are multiple people, describe what each appears to be doing.
- Keep your tone conversational and descriptive — you are summarising what the video shows, not making judgements.
- Do NOT default to safety/hazard analysis unless something is clearly and obviously dangerous.
- Respond in clear, natural language. No structured fields or rigid output formats required.
"""

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ARIA — SDI Safety Demo",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── Base ── */
    .stApp { background: #f3f4f6; color: #111827; }
    div[data-testid="stSidebar"] { display: none; }
    .block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; max-width: 1400px; }

    /* ── Header ── */
    .aria-header {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 20px 28px;
        margin-bottom: 20px;
        display: flex; align-items: center; gap: 18px;
        box-shadow: 0 1px 8px rgba(0,0,0,0.07);
    }
    .aria-badge {
        background: linear-gradient(135deg, #f59e0b, #ef4444);
        border-radius: 10px; padding: 8px 18px;
        font-size: 1.1rem; font-weight: 800; color: #fff;
        letter-spacing: 0.06em; white-space: nowrap;
        box-shadow: 0 2px 10px rgba(245,158,11,0.25);
    }
    .aria-title { font-size: 1.05rem; font-weight: 700; color: #111827; margin: 0; }
    .aria-sub   { font-size: 0.74rem; color: #6b7280; margin: 3px 0 0; }
    .aria-tags  { margin-left: auto; display: flex; gap: 8px; flex-wrap: wrap; }
    .aria-tag {
        background: #f3f4f6; border: 1px solid #e5e7eb; border-radius: 20px;
        padding: 4px 12px; font-size: 0.68rem; color: #6b7280; white-space: nowrap;
    }

    /* ── Selectbox ── */
    .stSelectbox > label { font-size: 0.78rem !important; color: #374151 !important; font-weight: 600 !important; }
    .stSelectbox > div > div {
        background: #ffffff !important; border: 1px solid #d1d5db !important;
        border-radius: 8px !important; color: #111827 !important;
    }

    /* ── Text input ── */
    .stTextInput > label { color: #374151 !important; font-size: 0.78rem !important; font-weight: 600 !important; }
    .stTextInput > div > div > input {
        background: #ffffff !important; border: 1px solid #d1d5db !important;
        border-radius: 8px !important; color: #111827 !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: #ffffff !important; border: 1px solid #e5e7eb !important;
        color: #374151 !important; border-radius: 8px !important;
        font-size: 0.75rem !important; text-align: left !important;
        padding: 8px 12px !important; transition: all 0.15s !important;
    }
    .stButton > button:hover {
        background: #f9fafb !important; border-color: #d1d5db !important;
        color: #111827 !important; box-shadow: 0 1px 4px rgba(0,0,0,0.08) !important;
    }

    /* ── Section label ── */
    .sec-label {
        font-size: 0.62rem; font-weight: 700; letter-spacing: 0.14em;
        text-transform: uppercase; color: #9ca3af; margin-bottom: 6px;
    }

    /* ── Frame caption ── */
    .frame-caption {
        font-size: 0.65rem; color: #9ca3af; text-align: center;
        letter-spacing: 0.08em; text-transform: uppercase; margin-top: 6px;
    }

    /* ── Context box ── */
    .context-box {
        background: #fffbeb; border-left: 3px solid #f59e0b;
        border-radius: 0 8px 8px 0; padding: 14px 16px;
        font-size: 0.78rem; line-height: 1.7; color: #374151;
    }

    /* ── Chat messages ── */
    [data-testid="stChatMessage"] {
        background: #ffffff !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 12px !important;
        margin-bottom: 10px !important;
        padding: 4px 8px !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05) !important;
    }
    [data-testid="stChatMessage"] p { color: #111827 !important; }
    [data-testid="stChatInput"] > div {
        background: #ffffff !important; border: 1px solid #d1d5db !important;
        border-radius: 10px !important;
    }

    /* ── vLLM info ── */
    .vllm-info {
        background: #f0f4ff; border: 1px solid #c7d2fe;
        border-radius: 8px; padding: 10px 14px;
        font-size: 0.74rem; color: #4338ca; line-height: 1.7; margin-top: 4px;
    }
    .vllm-info code {
        background: #e0e7ff; padding: 2px 7px; border-radius: 4px;
        color: #3730a3; font-size: 0.71rem;
    }

    /* ── Divider ── */
    hr { border-color: #e5e7eb !important; margin: 16px 0 !important; }

    /* ── Expander ── */
    [data-testid="stExpander"] {
        background: #ffffff !important; border: 1px solid #e5e7eb !important;
        border-radius: 10px !important;
    }

    /* ── Alerts ── */
    [data-testid="stAlert"] { border-radius: 8px !important; }

    /* ── Spinner ── */
    .stSpinner > div { border-top-color: #f59e0b !important; }

    /* ── Caption text ── */
    .stCaption { color: #9ca3af !important; }

    /* ── Sticky video panel ── */
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child {
        position: sticky;
        top: 1rem;
        align-self: flex-start;
        max-height: calc(100vh - 2rem);
        overflow-y: auto;
    }

    /* ── Domain toggle row ── */
    .toggle-row {
        display: flex; align-items: center; gap: 14px;
        background: #ffffff; border: 1px solid #e5e7eb; border-radius: 10px;
        padding: 12px 18px; margin-bottom: 16px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }
    .toggle-label { font-size: 0.82rem; font-weight: 600; color: #374151; }
    .toggle-desc  { font-size: 0.74rem; color: #6b7280; margin-left: auto; }

    .context-box-off {
        background: #f3f4f6; border-left: 3px solid #d1d5db;
        border-radius: 0 8px 8px 0; padding: 14px 16px;
        font-size: 0.78rem; line-height: 1.7; color: #9ca3af;
        filter: blur(2.5px); user-select: none; pointer-events: none;
    }
    .context-hidden-label {
        font-size: 0.74rem; color: #9ca3af; text-align: center;
        padding: 6px; letter-spacing: 0.05em;
    }

    /* badge for mode indicator */
    .mode-badge-on  { display:inline-block; padding:3px 10px; border-radius:20px; font-size:0.68rem; font-weight:700;
                       background:#dcfce7; color:#16a34a; border:1px solid #86efac; }
    .mode-badge-off { display:inline-block; padding:3px 10px; border-radius:20px; font-size:0.68rem; font-weight:700;
                       background:#fee2e2; color:#dc2626; border:1px solid #fca5a5; }

    /* pre-loaded response indicator */
    .preloaded-note {
        font-size:0.7rem; color:#9ca3af; font-style:italic; padding:4px 0 8px;
    }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def encode_image(path: Path):
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    mime = "image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    return data, mime


def extract_frames(video_path: Path, n: int = 6) -> list[tuple[str, float]]:
    """
    Extract n frames uniformly from a video file.
    Returns list of (base64_jpeg_string, timestamp_seconds).
    Frames are resized to max 640px wide to keep payload manageable.
    """
    import cv2 as _cv2
    import tempfile, os

    cap = _cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(_cv2.CAP_PROP_FPS) or 25.0
    duration     = total_frames / fps

    # Pick n evenly-spaced timestamps (avoid first/last 5% to skip black frames)
    margin = duration * 0.05
    timestamps = [margin + i * (duration - 2 * margin) / max(n - 1, 1) for i in range(n)]

    frames = []
    for ts in timestamps:
        cap.set(_cv2.CAP_PROP_POS_MSEC, ts * 1000)
        ret, frame = cap.read()
        if not ret:
            continue
        # Resize to max 640px wide
        h, w = frame.shape[:2]
        if w > 640:
            scale = 640 / w
            frame = _cv2.resize(frame, (640, int(h * scale)))
        # Encode to JPEG bytes → base64
        ret2, buf = _cv2.imencode(".jpg", frame, [_cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret2:
            b64 = base64.b64encode(buf.tobytes()).decode()
            frames.append((b64, ts))

    cap.release()
    return frames


def call_vllm(history, video_path, system_prompt, url, n_frames: int = 20):
    """
    Call local Gemma 3 27B via vLLM with multiple frames sampled from the video.
    All frames are sent in the first user message so the model sees the full
    temporal progression of the scene.
    """
    from openai import OpenAI
    client = OpenAI(base_url=url, api_key="EMPTY")
    models = client.models.list()
    if not models.data:
        raise RuntimeError("vLLM server reachable but no models loaded.")
    model_id = models.data[0].id

    # Extract frames from video
    frames = extract_frames(Path(video_path), n=n_frames)
    if not frames:
        raise RuntimeError(f"Could not extract frames from video: {video_path}")

    # System message as a proper role (not embedded in user content)
    msgs = [{"role": "system", "content": system_prompt}]

    for i, m in enumerate(history):
        if m["role"] == "user" and i == 0:
            # Build multi-image content: label + image for each frame
            content = []
            content.append({
                "type": "text",
                "text": f"The following {len(frames)} frames are sampled uniformly from the video, showing the full scene progression:\n"
            })
            for idx, (b64, ts) in enumerate(frames):
                content.append({"type": "text", "text": f"[Frame {idx+1} — t={ts:.0f}s]"})
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            content.append({"type": "text", "text": "\n" + m["content"]})
            msgs.append({"role": "user", "content": content})
        elif m["role"] == "user":
            msgs.append({"role": "user", "content": m["content"]})
        else:
            msgs.append({"role": "assistant", "content": m["content"]})

    r = client.chat.completions.create(model=model_id, messages=msgs, max_tokens=800, temperature=0.3)
    return r.choices[0].message.content.strip()


def call_gemini(history, frame_path, system_prompt, api_key):
    from google import genai
    from google.genai import types
    from PIL import Image as PILImage
    client = genai.Client(api_key=api_key)
    img = PILImage.open(frame_path).convert("RGB")
    contents = []
    for i, m in enumerate(history):
        if m["role"] == "user" and i == 0:
            contents.append(types.Content(role="user", parts=[
                types.Part.from_image(img),
                types.Part.from_text(m["content"]),
            ]))
        elif m["role"] == "user":
            contents.append(types.Content(role="user", parts=[types.Part.from_text(m["content"])]))
        else:
            contents.append(types.Content(role="model", parts=[types.Part.from_text(m["content"])]))
    resp = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=600,
            temperature=0.2,
        ),
    )
    return resp.text.strip()


def call_openai(history, frame_path, system_prompt, api_key):
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    img_b64, mime = encode_image(frame_path)
    msgs = [{"role": "system", "content": system_prompt}]
    for i, m in enumerate(history):
        if m["role"] == "user" and i == 0:
            msgs.append({"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
                {"type": "text", "text": m["content"]},
            ]})
        elif m["role"] == "user":
            msgs.append({"role": "user", "content": m["content"]})
        else:
            msgs.append({"role": "assistant", "content": m["content"]})
    r = client.chat.completions.create(model="gpt-4o", messages=msgs, max_tokens=600, temperature=0.2)
    return r.choices[0].message.content.strip()


def ask(question, clip, backend, cfg, domain_on: bool, n_frames: int = 20):
    st.session_state.messages.append({"role": "user", "content": question})
    if not clip["section"].startswith("SDI"):
        system = GENERAL_SYSTEM
    elif domain_on:
        system = SDI_SYSTEM + "\n\nSCENE CONTEXT:\n" + clip["context"]
    else:
        system = BARE_SYSTEM

    # Support clips with absolute video_path (e.g. egocentric videos outside clips/)
    video_path = clip.get("video_path") or (CLIPS_DIR / clip["video"] if clip["video"] else None)
    frame_path = clip["frame"]  # fallback for Gemini / OpenAI (single frame)

    try:
        if backend == "vLLM (local)":
            if video_path is None:
                raise RuntimeError("No video path configured for this clip.")
            answer = call_vllm(st.session_state.messages, video_path, system, cfg.get("vllm_url", ""), n_frames=n_frames)
        elif backend == "Gemini":
            if frame_path is None:
                raise RuntimeError("No frame path configured for this clip (Gemini requires a still frame).")
            answer = call_gemini(st.session_state.messages, frame_path, system, cfg.get("gemini_key", ""))
        else:
            if frame_path is None:
                raise RuntimeError("No frame path configured for this clip (GPT-4o requires a still frame).")
            answer = call_openai(st.session_state.messages, frame_path, system, cfg.get("openai_key", ""))
    except Exception as e:
        answer = f"⚠️ {e}"
    st.session_state.messages.append({"role": "assistant", "content": answer})


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="aria-header">
    <div class="aria-badge">ARIA</div>
    <div>
        <div class="aria-title">AI Risk &amp; Incident Analyzer</div>
        <div class="aria-sub">Stage 2 VLM Reasoning Layer &nbsp;·&nbsp; SDI Butler Steel Plant &nbsp;·&nbsp; CIVS × Purdue University Northwest</div>
    </div>
    <div class="aria-tags">
        <span class="aria-tag">Zero-shot</span>
        <span class="aria-tag">No fine-tuning</span>
        <span class="aria-tag">Gemma 3 27B</span>
        <span class="aria-tag">Live footage</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Backend config ─────────────────────────────────────────────────────────────
with st.container():
    c1, c2, c3 = st.columns([1, 2, 2])
    with c1:
        backend = st.selectbox("Model backend", ["Gemini", "GPT-4o", "vLLM (local)"])
    cfg = {}
    with c2:
        if backend == "Gemini":
            cfg["gemini_key"] = st.text_input("Google API Key", value=os.environ.get("GOOGLE_API_KEY", ""), type="password")
        elif backend == "GPT-4o":
            cfg["openai_key"] = st.text_input("OpenAI API Key", value=os.environ.get("OPENAI_API_KEY", ""), type="password")
        else:
            cfg["vllm_url"] = st.text_input("vLLM URL", value=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"))
    with c3:
        if backend == "vLLM (local)":
            st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
            if st.button("Test connection"):
                try:
                    from openai import OpenAI
                    mid = OpenAI(base_url=cfg["vllm_url"], api_key="EMPTY").models.list().data[0].id
                    st.success(f"✓ Connected · {mid}")
                except Exception as e:
                    st.error(str(e))
            st.markdown(
                '<div class="vllm-info">Start with: '
                '<code>conda activate gemmaenv</code> then '
                '<code>vllm serve google/gemma-3-27b-it --port 8000</code>'
                '</div>',
                unsafe_allow_html=True,
            )

st.markdown("---")

# ── Clip selector with section grouping ───────────────────────────────────────
sdi_clips  = [c for c in CLIPS if c["section"].startswith("SDI")]
gen_clips  = [c for c in CLIPS if c["section"] == "General Hazard Footage — Baseline"]
ego_clips  = [c for c in CLIPS if c["section"] == "General Video Understanding"]

col_sel1, col_sel2 = st.columns([3, 1])
with col_sel1:
    sdi_labels = [c["label"] for c in sdi_clips]
    gen_labels = [c["label"] for c in gen_clips]
    ego_labels = [c["label"] for c in ego_clips]
    all_labels = (
        sdi_labels
        + (["─── General Hazard Footage ───"] + gen_labels if gen_labels else [])
        + (["─── General Video Understanding ───"] + ego_labels if ego_labels else [])
    )
    chosen_label = st.selectbox("Select footage", all_labels)

# Skip separator entries
if chosen_label in ("─── General Hazard Footage ───", "─── General Video Understanding ───"):
    chosen_label = sdi_labels[0]

chosen = next((c for c in CLIPS if c["label"] == chosen_label), CLIPS[0])

with col_sel2:
    st.markdown(f'<div class="sec-label">{chosen["section"]}</div>', unsafe_allow_html=True)

# ── Domain context toggle + frame count selector ──────────────────────────────
is_sdi = chosen["section"].startswith("SDI")

tcol1, tcol2, tcol3 = st.columns([2, 3, 2])
with tcol1:
    domain_on = st.toggle(
        "Inject domain context",
        value=True,
        disabled=not is_sdi,
        help="Turn OFF to see the model respond with no plant-specific knowledge (the domain gap). Turn ON to inject SDI plant context into the prompt.",
    )
with tcol2:
    if is_sdi:
        if domain_on:
            st.markdown(
                '<span class="mode-badge-on">Domain context ON</span>'
                '<span style="font-size:0.73rem;color:#6b7280;margin-left:10px;">'
                'Model knows: pot haulers · red light indicators · zone rules · blind spots</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="mode-badge-off">Domain context OFF</span>'
                '<span style="font-size:0.73rem;color:#6b7280;margin-left:10px;">'
                'Model has zero plant knowledge — generic safety AI only</span>',
                unsafe_allow_html=True,
            )

with tcol3:
    frame_options = {"6 frames — fast ⚡": 6, "20 frames — detailed 🔍": 20}
    frame_label = st.radio(
        "Video sampling",
        options=list(frame_options.keys()),
        index=0,
        horizontal=False,
        help="6 frames: ~15-20s response. 20 frames: ~60-90s but sees more of the scene.",
    )
    n_frames = frame_options[frame_label]

# Reset chat on clip change OR toggle change
state_key = f"{chosen['id']}_{domain_on}"
if st.session_state.get("active_state") != state_key:
    st.session_state.active_state = state_key
    st.session_state.messages = []
    # Pre-load stored inference result so the contrast is instant
    if is_sdi and chosen["id"] in DEMO_RESPONSES:
        mode = "domain" if domain_on else "generic"
        demo = DEMO_RESPONSES[chosen["id"]][mode]
        st.session_state.messages = [
            {"role": "user",      "content": demo["q"]},
            {"role": "assistant", "content": demo["a"]},
        ]
        st.session_state["preloaded"] = True
    else:
        st.session_state["preloaded"] = False

st.markdown("")

# ── Main layout ───────────────────────────────────────────────────────────────
left, right = st.columns([1.15, 1], gap="large")

with left:
    clip_path = chosen.get("video_path") or (CLIPS_DIR / chosen["video"] if chosen["video"] else None)
    if clip_path and Path(clip_path).exists():
        st.video(str(clip_path))
    else:
        st.error(f"Clip not found: {clip_path}")

    # Video frames info — multi-frame sampling replaces single frame
    st.markdown(
        f'<div class="frame-caption">{n_frames} frames sampled across the full video · sent to ARIA with each question</div>',
        unsafe_allow_html=True
    )

    # Scene context (what the model knows)
    if is_sdi:
        if domain_on:
            with st.expander("Scene context injected into model ✓", expanded=False):
                st.markdown(f'<div class="context-box">{chosen["context"]}</div>', unsafe_allow_html=True)
        else:
            with st.expander("Scene context — HIDDEN from model ✗", expanded=False):
                st.markdown(
                    '<div class="context-hidden-label">🔒 Context withheld — model receives no plant-specific information</div>'
                    f'<div class="context-box-off">{chosen["context"]}</div>',
                    unsafe_allow_html=True,
                )
    else:
        with st.expander("Scene context injected into model"):
            st.markdown(f'<div class="context-box">{chosen["context"]}</div>', unsafe_allow_html=True)

with right:
    st.markdown("**Ask ARIA**")

    # Pre-loaded indicator
    if st.session_state.get("preloaded") and st.session_state.get("messages"):
        if domain_on:
            st.markdown(
                '<div class="preloaded-note">⚡ Pre-loaded: real inference result with domain context · Ask a follow-up question below</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="preloaded-note">⚡ Pre-loaded: real inference result WITHOUT context · Toggle "Inject domain context" to compare</div>',
                unsafe_allow_html=True,
            )

    # Chat history
    for msg in st.session_state.get("messages", []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if not st.session_state.get("messages"):
        st.markdown(
            "<div style='color:#4b5563; font-size:0.84rem; padding:12px 0 4px;'>"
            "Type a question below, or click a suggested prompt ↓"
            "</div>",
            unsafe_allow_html=True,
        )

    # Chat input
    prefill = st.session_state.pop("prefill", "")
    user_input = st.chat_input("Ask anything about this scene…")
    question = user_input or (prefill if prefill else None)

    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("ARIA analyzing…"):
                ask(question, chosen, backend, cfg, domain_on, n_frames=n_frames)
            st.markdown(st.session_state.messages[-1]["content"])

    # Suggested questions for this clip
    st.markdown("---")
    st.markdown('<div class="sec-label">Suggested questions for this clip</div>', unsafe_allow_html=True)
    for i, q in enumerate(chosen["questions"]):
        if st.button(q, key=f"sq_{chosen['id']}_{i}", use_container_width=True):
            st.session_state["prefill"] = q
            st.rerun()

    if st.session_state.get("messages"):
        st.markdown("")
        if st.button("Clear conversation", key="clear_chat"):
            st.session_state.messages = []
            st.rerun()

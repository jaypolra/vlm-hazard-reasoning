"""
ARIA — AI Risk & Incident Analyzer
VLM Reasoning Layer for Industrial Hazard Recognition
CIVS × Purdue University Northwest · SDI Butler Steel Plant

Prompt engineering module — four experimental variants.
This is the core research contribution: structured prompts that give a
general-purpose VLM enough domain context to reason correctly about
SDI Butler plant-specific hazards without any fine-tuning.

Variants:
  1. DIRECT   — plain question, no context (worst-case baseline)
  2. DOMAIN   — domain-specific facility context injected
  3. COT      — chain-of-thought, step-by-step reasoning
  4. MULTI_Q  — multi-question chain, structured output

Each variant supports two modes:
  generic  — Phase 1 open-source safety videos
  sdi      — Phase 2 SDI Butler industrial footage

Usage:
  from prompt_templates import build_prompt
  prompt = build_prompt(variant="cot", mode="sdi", hazard_key="missing_blockers")

Part of the research:
  "Bridging the Domain Gap in Industrial Safety with Vision Language Models"
  CIVS × Purdue University Northwest for SDI Butler Steel Plant
"""

from hazard_library import format_hazard_for_prompt, get_hazard

# ─────────────────────────────────────────────────────────────────────
# FACILITY CONTEXT BLOCK
# Injected into domain-aware prompts for SDI scenarios
# ─────────────────────────────────────────────────────────────────────

SDI_FACILITY_CONTEXT = """
FACILITY CONTEXT:
You are analyzing footage from a steel manufacturing plant (SDI Butler facility).
This is a heavy industrial environment with the following characteristics:
- Heavy mobile equipment (pot haulers, cranes, forklifts) operates in bays
- High-temperature processes create glare and reduced visibility
- Certain zones are designated as RED ZONES (danger zones) — entry is restricted
- Physical blockers (swing gates, barriers) must be present at designated positions for a zone to be considered SAFE
- A RED LIGHT visible on the ground indicates a hazardous operation is in progress — no entry permitted
- Camera quality is limited — some details (swing gate positions, blocker status) may be unclear
- When visibility is insufficient to confirm safety, default to UNSAFE

BLOCKER LOGIC:
  Zone + Person present + Blockers confirmed in place = SAFE
  Zone + Person present + Blockers missing or unconfirmable = UNSAFE
  Zone + Red light visible = UNSAFE regardless of blocker status
"""

GENERIC_FACILITY_CONTEXT = """
FACILITY CONTEXT:
You are analyzing footage from an industrial or construction work environment.
Apply standard workplace safety principles and OSHA regulations when assessing hazards.
"""

# ─────────────────────────────────────────────────────────────────────
# OUTPUT FORMAT BLOCK
# Consistent structured output regardless of variant
# ─────────────────────────────────────────────────────────────────────

OUTPUT_FORMAT = """
Respond in this exact format:

HAZARD_DETECTED: YES / NO / UNCERTAIN
SEVERITY: CRITICAL / HIGH / MEDIUM / LOW / NONE
CONFIDENCE: HIGH / MEDIUM / LOW

WHAT_I_SEE:
[One sentence describing what is visible in the frame]

HAZARD_ANALYSIS:
[2-3 sentences explaining whether the specific hazard is present and why]

SAFE_OR_UNSAFE: SAFE / UNSAFE / UNCONFIRMABLE
REASON: [One sentence — the single most important reason for your assessment]

RECOMMENDED_ACTION:
[One concrete action that should be taken right now]
"""

# ─────────────────────────────────────────────────────────────────────
# VARIANT 1 — DIRECT
# Bare minimum — no context, no guidance
# Establishes the baseline: what does VLM know without any help?
# ─────────────────────────────────────────────────────────────────────

def prompt_direct(mode: str = "generic") -> str:
    """
    Direct prompt — no hazard context, no facility info.
    Tests raw VLM safety knowledge.
    """
    return f"""
Look at this image from a workplace environment.

What safety hazards, if any, do you see?
Is the situation SAFE or UNSAFE?
Explain briefly.

{OUTPUT_FORMAT}
""".strip()


# ─────────────────────────────────────────────────────────────────────
# VARIANT 2 — DOMAIN-SPECIFIC
# Injects facility context + specific hazard definition
# Tests: does domain knowledge improve detection?
# ─────────────────────────────────────────────────────────────────────

def prompt_domain(hazard_key: str, mode: str = "generic") -> str:
    """
    Domain-specific prompt — injects facility context and hazard definition.
    Best for known hazard types where context matters.
    """
    facility_ctx = SDI_FACILITY_CONTEXT if mode == "sdi" else GENERIC_FACILITY_CONTEXT
    hazard_block = format_hazard_for_prompt(hazard_key)
    hazard = get_hazard(hazard_key)

    return f"""
You are an industrial safety expert analyzing a surveillance camera image.

{facility_ctx}

You are specifically looking for the following hazard:
──────────────────────────────────────────
{hazard_block}
──────────────────────────────────────────

Analyze the image with this specific hazard in mind.

{OUTPUT_FORMAT}
""".strip()


# ─────────────────────────────────────────────────────────────────────
# VARIANT 3 — CHAIN OF THOUGHT (CoT)
# Forces step-by-step reasoning before conclusion
# Tests: does structured thinking improve accuracy on complex scenes?
# ─────────────────────────────────────────────────────────────────────

def prompt_cot(hazard_key: str = None, mode: str = "generic") -> str:
    """
    Chain-of-thought prompt — forces step-by-step reasoning.
    Works with or without a specific hazard key.
    """
    facility_ctx = SDI_FACILITY_CONTEXT if mode == "sdi" else GENERIC_FACILITY_CONTEXT

    hazard_section = ""
    if hazard_key:
        hazard_block = format_hazard_for_prompt(hazard_key)
        hazard_section = f"""
You are specifically checking for:
──────────────────────────────────────────
{hazard_block}
──────────────────────────────────────────
"""

    if mode == "sdi":
        steps = """
Think through this step by step:

STEP 1 — PEOPLE
Who is present in the frame? Count them. Where are they positioned?
Are they on foot, in a vehicle, or at a workstation?

STEP 2 — EQUIPMENT
What machinery or vehicles are visible?
Is any equipment actively operating, moving, or in a hazardous state?

STEP 3 — ZONE AND BARRIERS
Are any restricted zones visible?
Are physical blockers (swing gates, barriers, vehicles) present at their designated positions?
Can you clearly confirm blocker status, or is visibility too low?

STEP 4 — ENVIRONMENTAL INDICATORS
Is there a red light visible on the ground?
Are there any warning signs, flashing lights, or other indicators of active operations?

STEP 5 — PROXIMITY AND INTERACTION
Is any person in unsafe proximity to equipment, a hazard zone, or an active process?
Is there a clear physical separation between people and hazards?

STEP 6 — VISIBILITY LIMITATIONS
What cannot you see clearly due to image quality, distance, or occlusion?
How does this uncertainty affect your assessment?

STEP 7 — FINAL ASSESSMENT
Based on steps 1-6, is this situation SAFE, UNSAFE, or UNCONFIRMABLE?
"""
    else:
        steps = """
Think through this step by step:

STEP 1 — PEOPLE
Who is present in the frame? Where are they? What are they doing?

STEP 2 — EQUIPMENT AND ENVIRONMENT
What machinery, tools, or hazardous elements are present?
Is any equipment in an active or dangerous state?

STEP 3 — PPE CHECK
Are workers wearing appropriate personal protective equipment?
What PPE is required in this environment? What is missing?

STEP 4 — PROXIMITY AND POSITION
Is any person in a dangerous position or proximity to a hazard?
Is there an unsafe distance between a worker and a hazard?

STEP 5 — BARRIERS AND CONTROLS
Are safety barriers, guards, or controls in place?
Are OSHA-required safeguards present and functional?

STEP 6 — FINAL ASSESSMENT
Based on steps 1-5, what is the safety status and why?
"""

    return f"""
You are an industrial safety expert analyzing a surveillance camera image.

{facility_ctx}
{hazard_section}
{steps}

After completing all steps, provide your final answer:

{OUTPUT_FORMAT}
""".strip()


# ─────────────────────────────────────────────────────────────────────
# VARIANT 4 — MULTI-QUESTION CHAIN
# Structured Q&A — each answer builds on the previous
# Tests: does constrained questioning reduce hallucination?
# ─────────────────────────────────────────────────────────────────────

def prompt_multi_q(hazard_key: str = None, mode: str = "generic") -> str:
    """
    Multi-question chain — structured Q&A with yes/no anchors.
    Reduces hallucination by constraining each answer before moving on.
    """
    facility_ctx = SDI_FACILITY_CONTEXT if mode == "sdi" else GENERIC_FACILITY_CONTEXT

    hazard_section = ""
    if hazard_key:
        hazard = get_hazard(hazard_key)
        hazard_section = f"\nTarget hazard: {hazard['name']}\n{hazard['description']}\n"

    if mode == "sdi":
        questions = """
Answer each question in order. Keep each answer to 1-2 sentences.

Q1: Are any people (workers, personnel) visible in this frame?
    [YES/NO — if YES, how many and where?]

Q2: Is heavy mobile equipment (crane, pot hauler, forklift) visible?
    [YES/NO — if YES, does it appear to be active/operating?]

Q3: Are physical blockers or swing gates visible at their designated positions?
    [YES/PARTIALLY/NO/CANNOT CONFIRM — explain briefly]

Q4: Is a red light visible on the ground surface?
    [YES/NO/UNCERTAIN]

Q5: Is any person in the zone or in proximity to active equipment without clear separation?
    [YES/NO/UNCERTAIN]

Q6: Can you confirm the safety status from this camera view alone?
    [YES/NO — if NO, what additional information would be needed?]

Q7: Overall assessment:
    [SAFE / UNSAFE / UNCONFIRMABLE — one sentence reason]

Q8: What is the single most important action to take right now?
    [One clear instruction]
"""
    else:
        questions = """
Answer each question in order. Keep each answer to 1-2 sentences.

Q1: Are any workers or people visible in this frame?
    [YES/NO — if YES, describe their location and activity]

Q2: Is there any active machinery, equipment, or hazardous process visible?
    [YES/NO — describe if present]

Q3: Are workers wearing appropriate PPE for this environment?
    [YES/NO/PARTIAL/CANNOT DETERMINE — explain what is present or missing]

Q4: Is any worker in an unsafe position or proximity to a hazard?
    [YES/NO — explain if YES]

Q5: Are required safety controls (guards, barriers, signage) in place?
    [YES/NO/PARTIAL]

Q6: Overall assessment:
    [SAFE / UNSAFE — one sentence reason]

Q7: What is the single most important action to take right now?
    [One clear instruction]
"""

    return f"""
You are an industrial safety expert analyzing a surveillance camera image.

{facility_ctx}
{hazard_section}
{questions}

After answering all questions, provide this summary:

{OUTPUT_FORMAT}
""".strip()


# ─────────────────────────────────────────────────────────────────────
# MASTER BUILDER — single entry point for all prompts
# ─────────────────────────────────────────────────────────────────────

VARIANTS = {
    "direct":   prompt_direct,
    "domain":   prompt_domain,
    "cot":      prompt_cot,
    "multi_q":  prompt_multi_q,
}

def build_prompt(variant: str, mode: str = "generic", hazard_key: str = None) -> str:
    """
    Build a prompt for VLM inference.

    Args:
        variant:    "direct" | "domain" | "cot" | "multi_q"
        mode:       "generic" (Phase 1) | "sdi" (Phase 2)
        hazard_key: key from hazard_library.py (optional for direct)

    Returns:
        str: complete prompt ready to send to any VLM
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(VARIANTS.keys())}")

    fn = VARIANTS[variant]

    if variant == "direct":
        return fn(mode=mode)
    else:
        if not hazard_key:
            raise ValueError(f"Variant '{variant}' requires a hazard_key.")
        return fn(hazard_key=hazard_key, mode=mode)


# ─────────────────────────────────────────────────────────────────────
# EXPERIMENT MATRIX — all combinations for Phase 3
# ─────────────────────────────────────────────────────────────────────

PHASE1_EXPERIMENT_MATRIX = [
    # (variant, mode, hazard_key)
    ("direct",  "generic", None),
    ("domain",  "generic", "ppe_no_helmet"),
    ("cot",     "generic", "ppe_no_helmet"),
    ("multi_q", "generic", "ppe_no_helmet"),

    ("direct",  "generic", None),
    ("domain",  "generic", "fall_from_height"),
    ("cot",     "generic", "fall_from_height"),
    ("multi_q", "generic", "fall_from_height"),

    ("direct",  "generic", None),
    ("domain",  "generic", "forklift_pedestrian"),
    ("cot",     "generic", "forklift_pedestrian"),
    ("multi_q", "generic", "forklift_pedestrian"),
]

PHASE2_EXPERIMENT_MATRIX = [
    # (variant, mode, hazard_key)
    ("direct",  "sdi", None),
    ("domain",  "sdi", "people_with_active_equipment"),
    ("cot",     "sdi", "people_with_active_equipment"),
    ("multi_q", "sdi", "people_with_active_equipment"),

    ("direct",  "sdi", None),
    ("domain",  "sdi", "missing_blockers"),
    ("cot",     "sdi", "missing_blockers"),
    ("multi_q", "sdi", "missing_blockers"),

    ("direct",  "sdi", None),
    ("domain",  "sdi", "blind_spot_vehicle"),
    ("cot",     "sdi", "blind_spot_vehicle"),
    ("multi_q", "sdi", "blind_spot_vehicle"),

    ("direct",  "sdi", None),
    ("domain",  "sdi", "red_light_indicator"),
    ("cot",     "sdi", "red_light_indicator"),
    ("multi_q", "sdi", "red_light_indicator"),
]


if __name__ == "__main__":
    # Quick preview of each variant
    from hazard_library import list_hazards

    print("=" * 60)
    print("VARIANT 1 — DIRECT (generic)")
    print("=" * 60)
    print(build_prompt("direct", mode="generic"))

    print("\n" + "=" * 60)
    print("VARIANT 2 — DOMAIN (SDI, missing blockers)")
    print("=" * 60)
    print(build_prompt("domain", mode="sdi", hazard_key="missing_blockers"))

    print("\n" + "=" * 60)
    print("VARIANT 3 — CHAIN OF THOUGHT (SDI, red light)")
    print("=" * 60)
    print(build_prompt("cot", mode="sdi", hazard_key="red_light_indicator"))

    print("\n" + "=" * 60)
    print("VARIANT 4 — MULTI-QUESTION (SDI, blind spot)")
    print("=" * 60)
    print(build_prompt("multi_q", mode="sdi", hazard_key="blind_spot_vehicle"))

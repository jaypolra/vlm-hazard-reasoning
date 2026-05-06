"""
ARIA — AI Risk & Incident Analyzer
VLM Reasoning Layer for Industrial Hazard Recognition
CIVS × Purdue University Northwest · SDI Butler Steel Plant

Hazard definition library — text-based hazard specifications for VLM prompting.
No retraining needed — add a new hazard by adding a new dict entry.

Two worlds:
  PHASE1_HAZARDS  — generic / open-source scenarios (YouTube baseline)
  PHASE2_HAZARDS  — SDI Butler industrial scenarios (real plant footage)

Each hazard entry defines:
  description      — what the hazard is
  indicators       — visual cues the VLM should look for
  safe_condition   — what makes the situation SAFE
  unsafe_condition — what makes it UNSAFE
  osha_ref         — relevant OSHA standard (for domain prompts)
  severity         — LOW / MEDIUM / HIGH / CRITICAL

Part of the research:
  "Bridging the Domain Gap in Industrial Safety with Vision Language Models"
  CIVS × Purdue University Northwest for SDI Butler Steel Plant
"""

# ─────────────────────────────────────────────────────────────────────
# PHASE 1 — Generic / Open Source Hazards
# These are clear, well-lit, training-video style scenarios
# VLM baseline — should detect these well
# ─────────────────────────────────────────────────────────────────────

PHASE1_HAZARDS = {

    "ppe_no_helmet": {
        "name": "PPE Violation — No Hard Hat",
        "description": "A worker is present in an area requiring head protection but is not wearing a hard hat or helmet.",
        "indicators": [
            "Person visible without hard hat on head",
            "Construction or industrial environment",
            "Other workers nearby may or may not be wearing helmets"
        ],
        "safe_condition": "All workers in the frame are wearing hard hats appropriate for the environment.",
        "unsafe_condition": "One or more workers visible without head protection in a hazardous area.",
        "osha_ref": "OSHA 1926.100 — Head protection required where there is risk of head injury.",
        "severity": "HIGH"
    },

    "ppe_no_vest": {
        "name": "PPE Violation — No High-Visibility Vest",
        "description": "A worker is present in an active work zone without a high-visibility safety vest.",
        "indicators": [
            "Person wearing regular clothing without reflective vest",
            "Active vehicle or machinery movement in area",
            "Warehouse, construction site, or road work environment"
        ],
        "safe_condition": "Worker is wearing a high-visibility vest with reflective strips.",
        "unsafe_condition": "Worker is not wearing a high-visibility vest in an area with active vehicle traffic or machinery.",
        "osha_ref": "OSHA 1926.201 — High-visibility apparel required near vehicle or equipment traffic.",
        "severity": "HIGH"
    },

    "ppe_no_harness": {
        "name": "PPE Violation — No Fall Harness at Height",
        "description": "A worker is performing tasks at elevation without a fall arrest harness or lanyard.",
        "indicators": [
            "Person working on roof, scaffold, elevated platform, or ladder above 6 feet",
            "No visible harness straps on worker",
            "No lanyard connecting worker to anchor point"
        ],
        "safe_condition": "Worker is wearing a visible harness with lanyard attached to a fixed anchor point.",
        "unsafe_condition": "Worker at height with no fall protection equipment visible.",
        "osha_ref": "OSHA 1926.502 — Fall protection required at 6 feet or more in construction.",
        "severity": "CRITICAL"
    },

    "fall_from_height": {
        "name": "Fall from Height — Active or Imminent",
        "description": "A worker is in a position where a fall from an elevated surface has occurred or is likely.",
        "indicators": [
            "Person near unguarded edge of elevated surface",
            "Unstable footing on scaffold, roof, or ladder",
            "Person in mid-fall or fallen from height",
            "Broken or missing guardrails"
        ],
        "safe_condition": "Elevated areas have guardrails, worker has fall arrest harness, or worker is not near edge.",
        "unsafe_condition": "Worker near unguarded edge, on unstable surface, or fall in progress.",
        "osha_ref": "OSHA 1926.502 — Guardrail systems, safety net systems, or personal fall arrest required.",
        "severity": "CRITICAL"
    },

    "machinery_unguarded": {
        "name": "Machinery Danger — Unguarded Moving Parts",
        "description": "Rotating or moving machinery parts are exposed without a guard, creating entanglement or crush risk.",
        "indicators": [
            "Visible rotating shaft, belt, gear, or conveyor with no guard cover",
            "Worker in proximity to moving mechanical parts",
            "Guard removed or missing from machine"
        ],
        "safe_condition": "All moving parts are fully enclosed by guards. Worker is at safe distance.",
        "unsafe_condition": "Moving parts exposed without guarding. Worker is within reach distance.",
        "osha_ref": "OSHA 1910.212 — Machine guarding required for all machines with exposed moving parts.",
        "severity": "CRITICAL"
    },

    "machinery_loto_violation": {
        "name": "Machinery Danger — Lockout/Tagout Violation",
        "description": "A worker is performing maintenance or servicing on a machine that has not been locked out and de-energized.",
        "indicators": [
            "Worker reaching into or working inside machine",
            "No visible lock or tag on energy source",
            "Machine appears energized or capable of movement while being serviced"
        ],
        "safe_condition": "Energy sources locked and tagged. Machine is confirmed de-energized before work begins.",
        "unsafe_condition": "Worker servicing machine without visible lockout device applied.",
        "osha_ref": "OSHA 1910.147 — Control of hazardous energy (lockout/tagout).",
        "severity": "CRITICAL"
    },

    "forklift_pedestrian": {
        "name": "Machinery Danger — Forklift and Pedestrian Proximity",
        "description": "A forklift or powered industrial truck is operating in close proximity to a pedestrian without adequate separation.",
        "indicators": [
            "Forklift and person visible in same area",
            "No physical barrier between pedestrian and forklift path",
            "Forklift in motion or capable of motion"
        ],
        "safe_condition": "Pedestrian in designated walkway with physical barrier. Forklift in vehicle-only lane. Or forklift is stopped with spotter present.",
        "unsafe_condition": "Pedestrian within forklift operating radius without barrier or spotter.",
        "osha_ref": "OSHA 1910.178 — Powered industrial trucks must maintain safe separation from pedestrians.",
        "severity": "HIGH"
    },

    "near_miss_falling_object": {
        "name": "Near Miss — Falling Object Hazard",
        "description": "An object has fallen or is at risk of falling from an elevated position onto a person or work area below.",
        "indicators": [
            "Object falling through frame",
            "Unstable materials on elevated surface near edge",
            "Worker below elevated work area without hard hat",
            "No toe boards or barricade at base of elevated work"
        ],
        "safe_condition": "Elevated areas have toe boards. Below area is barricaded. Workers below wearing hard hats.",
        "unsafe_condition": "Unsecured materials at height. Workers present below without protection.",
        "osha_ref": "OSHA 1926.502(j) — Protection from falling objects required.",
        "severity": "HIGH"
    },

    "near_miss_electrical": {
        "name": "Near Miss — Electrical Hazard",
        "description": "A worker is in close proximity to exposed electrical equipment, live panels, or overhead power lines.",
        "indicators": [
            "Open electrical panel with exposed wiring",
            "Worker approaching or reaching toward live electrical components",
            "No insulated gloves or PPE visible"
        ],
        "safe_condition": "Electrical panels are closed and locked. Worker using insulated PPE. Panel de-energized before access.",
        "unsafe_condition": "Worker near live electrical components without proper PPE or de-energization.",
        "osha_ref": "OSHA 1910.303 — Electrical safety — working near live parts.",
        "severity": "CRITICAL"
    },

    "near_miss_vehicle_pedestrian": {
        "name": "Near Miss — Vehicle and Pedestrian Conflict",
        "description": "A vehicle (car, truck, or industrial vehicle) comes into close proximity with a pedestrian in a shared space.",
        "indicators": [
            "Vehicle and pedestrian in same travel path",
            "No physical separation between vehicle lane and walking area",
            "Pedestrian not visible to driver or not aware of vehicle"
        ],
        "safe_condition": "Physical barriers separate vehicle and pedestrian paths. Clear sightlines. Spotter present.",
        "unsafe_condition": "Vehicle and pedestrian sharing same path without separation or awareness.",
        "osha_ref": "OSHA 1910.178(n) — Safe speeds and pedestrian right-of-way required.",
        "severity": "HIGH"
    },
}


# ─────────────────────────────────────────────────────────────────────
# PHASE 2 — SDI Butler Industrial Scenarios
# Real steel plant footage — complex, low visibility, ambiguous
# VLM stress test — domain gap expected here
# ─────────────────────────────────────────────────────────────────────

PHASE2_HAZARDS = {

    "people_with_active_equipment": {
        "name": "SDI — Personnel in Active Mobile Equipment Zone",
        "description": (
            "Workers are present inside a bay while heavy mobile equipment "
            "(pot haulers, cranes, or forklifts) is actively operating. "
            "The combination of worker presence and active equipment creates "
            "crush, collision, and caught-in hazards."
        ),
        "indicators": [
            "One or more people visible on foot in the bay",
            "Heavy mobile equipment visible and appearing to be in operation",
            "No visible physical separation between personnel and equipment path",
            "Workers not in designated safe waiting areas"
        ],
        "safe_condition": (
            "All personnel are in designated safe areas behind physical barriers. "
            "OR equipment is fully stopped and locked out. "
            "OR a trained spotter is positioned with radio contact."
        ),
        "unsafe_condition": (
            "Personnel are on foot in the active travel path or operating radius "
            "of mobile equipment without physical separation."
        ),
        "osha_ref": "OSHA 1910.178(n)(4) — Sufficient clearance required for pedestrians in aisles where powered trucks operate.",
        "severity": "CRITICAL"
    },

    "missing_blockers": {
        "name": "SDI — Blockers Absent at Designated Positions",
        "description": (
            "Physical safety barriers (blockers) are required to be placed at "
            "designated positions around a hazard zone before personnel may enter. "
            "If blockers are missing, the zone is UNSAFE regardless of other conditions. "
            "In this facility, swing gates and physical equipment serve as blockers."
        ),
        "indicators": [
            "Personnel visible inside or approaching the hazard zone",
            "Designated blocker positions appear empty",
            "Swing gates appear open or absent",
            "No physical barrier visible between personnel and hazard area"
        ],
        "safe_condition": (
            "Physical blockers are visibly placed at all designated positions around the zone. "
            "Swing gates are closed. Personnel entry is permitted only after blocker confirmation."
        ),
        "unsafe_condition": (
            "Personnel are present in or near the zone but one or more designated "
            "blocker positions appear unoccupied or gates appear open."
        ),
        "osha_ref": "OSHA 1910.147 — Energy control / lockout tagout: barriers must be in place before personnel exposure.",
        "severity": "CRITICAL",
        "note": "LOW VISIBILITY WARNING: Swing gates in this facility may not be clearly visible on camera due to distance and image quality. If blocker status cannot be confirmed visually, treat as UNSAFE."
    },

    "blind_spot_vehicle": {
        "name": "SDI — Heavy Equipment at Camera Blind Spot",
        "description": (
            "A heavy mobile vehicle (pot hauler or similar equipment) is positioned "
            "at a location where the camera can only see the entrance or exit of the vehicle, "
            "not its full body or operating position. "
            "The actual hazard zone created by the vehicle cannot be fully assessed from this camera angle."
        ),
        "indicators": [
            "Partial view of large vehicle at edge of camera frame",
            "Front or rear of vehicle visible but body extends out of frame",
            "Personnel possibly present in the unseen area",
            "Vehicle appears to be positioned at bay entrance or transition zone"
        ],
        "safe_condition": (
            "Vehicle is fully visible and confirmed stationary with no personnel nearby. "
            "OR camera coverage from another angle confirms the blind area is clear."
        ),
        "unsafe_condition": (
            "Vehicle partially visible at blind spot. Cannot confirm whether personnel "
            "are present in the unseen zone. Situation must be treated as potentially unsafe."
        ),
        "osha_ref": "OSHA 1910.178(n)(3) — Safe clearance required where vision is obstructed.",
        "severity": "HIGH",
        "note": "CAMERA LIMITATION: This is a known blind spot for this camera. Cross-reference with adjacent camera feeds if available."
    },

    "swing_gate_low_visibility": {
        "name": "SDI — Swing Gate Status Unconfirmable (Low Visibility)",
        "description": (
            "Swing gates serve as primary blockers in this facility. "
            "Due to camera distance, low image quality, or lighting conditions, "
            "the open/closed status of swing gates cannot be reliably determined from the video feed."
        ),
        "indicators": [
            "General area where swing gate should be present appears unclear",
            "Low image contrast or resolution in gate area",
            "Gate structure may or may not be visible",
            "Dark or overexposed areas near gate positions"
        ],
        "safe_condition": (
            "Gate is clearly visible and clearly closed, forming a physical barrier. "
            "OR physical verification by personnel confirms gate status."
        ),
        "unsafe_condition": (
            "Gate status cannot be confirmed from camera feed. "
            "When in doubt, status must default to UNSAFE until physically verified."
        ),
        "osha_ref": "OSHA 1910.147 — Barriers must be confirmed in place, not assumed.",
        "severity": "HIGH",
        "note": (
            "VISUAL LIMITATION ACKNOWLEDGED: This is a known limitation of camera-based monitoring. "
            "The AI system should flag this as 'UNCONFIRMABLE — REQUIRES PHYSICAL CHECK' "
            "rather than assuming safe or unsafe."
        )
    },

    "red_light_indicator": {
        "name": "SDI — Red Ground Light Active (Hazard Operation in Progress)",
        "description": (
            "A red light visible on the ground or low surface level in the bay "
            "indicates that a hazardous operation is currently in progress. "
            "This is a facility-specific safety indicator meaning personnel "
            "should NOT enter the zone while the light is active."
        ),
        "indicators": [
            "Red or orange-red light visible on floor level",
            "Light appears to be a warning indicator rather than ambient lighting",
            "Light is within or near the defined hazard zone"
        ],
        "safe_condition": (
            "Red light is off or not visible, indicating no active hazardous operation. "
            "Zone entry may be permitted based on other safety checks."
        ),
        "unsafe_condition": (
            "Red light is ON. Hazardous operation is in progress. "
            "No personnel should be inside or approaching the zone."
        ),
        "osha_ref": "OSHA 1910.145 — Accident prevention signs and tags: signal word meanings must be followed.",
        "severity": "CRITICAL",
        "note": (
            "FACILITY-SPECIFIC INDICATOR: This red light system is specific to SDI Butler. "
            "This context must be explicitly provided to the VLM — "
            "a generic model will not know what a red floor light means in this facility."
        )
    },
}


# ─────────────────────────────────────────────────────────────────────
# COMBINED — for scripts that iterate over all hazards
# ─────────────────────────────────────────────────────────────────────

ALL_HAZARDS = {**PHASE1_HAZARDS, **PHASE2_HAZARDS}


def get_hazard(key: str) -> dict:
    """Return a hazard definition by key. Raises KeyError if not found."""
    return ALL_HAZARDS[key]


def list_hazards(phase: int = None) -> list:
    """List all hazard keys. Filter by phase (1 or 2) if provided."""
    if phase == 1:
        return list(PHASE1_HAZARDS.keys())
    elif phase == 2:
        return list(PHASE2_HAZARDS.keys())
    return list(ALL_HAZARDS.keys())


def format_hazard_for_prompt(key: str) -> str:
    """Format a hazard definition as a readable block for injection into a prompt."""
    h = get_hazard(key)
    lines = [
        f"HAZARD: {h['name']}",
        f"Description: {h['description']}",
        f"",
        f"Visual indicators to look for:",
    ]
    for ind in h["indicators"]:
        lines.append(f"  - {ind}")
    lines += [
        f"",
        f"SAFE condition: {h['safe_condition']}",
        f"UNSAFE condition: {h['unsafe_condition']}",
        f"Severity level: {h['severity']}",
        f"Regulatory reference: {h['osha_ref']}",
    ]
    if "note" in h:
        lines += [f"", f"IMPORTANT NOTE: {h['note']}"]
    return "\n".join(lines)


if __name__ == "__main__":
    print(f"Phase 1 hazards ({len(PHASE1_HAZARDS)}): {list_hazards(1)}")
    print(f"Phase 2 hazards ({len(PHASE2_HAZARDS)}): {list_hazards(2)}")
    print()
    print("─" * 60)
    print(format_hazard_for_prompt("red_light_indicator"))

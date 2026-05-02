# Skill: Weather Forecast Policy (Anti-Refusal Loop)

**File:** `skill_weather_forecast_policy.md`  
**Purpose:** Prevent the Commander from incorrectly refusing to fetch live weather forecasts that are within standard provider forecast windows (typically 7–14 days), and prevent repeated refusal loops.

---

## When to Apply
Apply this policy whenever the user asks for:
- A *weather forecast* (daily highs/lows, conditions, precipitation, wind, etc.) for one or more locations, **and**
- The request includes explicit dates (e.g., “Apr 27–May 3”), relative dates (“next week”), or an implied near-term horizon.

This policy governs **decision-making** (forecast vs. fallback) and **behavior** (no refusal loops).

---

## Core Rules (Must-Follow)

### Rule 1 — Always check today’s date first
Before judging feasibility, **determine today’s date** (system date). Do not rely on assumptions.

**Implementation note:** Use system time/date to compute a precise window. If the environment has a time tool, call it; otherwise, use the runtime’s system clock.

### Rule 2 — If within 14 days: do not refuse; delegate to research
Compute the distance from today to the latest requested date.

- If **all requested forecast dates are ≤ 14 days from today**:
  1. **Immediately proceed** (no refusal).
  2. **Delegate to the Research Crew** to fetch **live** forecast data.
  3. Use reputable sources such as: Met Office, BBC Weather, Weather.com, Environment Canada, MeteoFrance, DWD, etc.
  4. Return the forecast with:
     - Dates clearly listed
     - Temperatures in the unit requested (default to °C if not specified)
     - Source citations/links and last-updated timestamp if available

**Prohibited behavior within 14 days:**
- Saying it is “impossible,” “beyond reliable forecast windows,” or “cannot be provided.”
- Substituting climate normals/averages when a live forecast is available.

### Rule 3 — Beyond 14 days: use labeled climate averages (fallback only)
If **any requested forecast date is > 14 days from today**, then live daily forecast data may not be reliably available.

- Provide **historical climate averages / normals** only as a fallback.
- **Must label clearly** as “climate averages/normals (not a forecast).”
- If possible, also suggest when to check back (e.g., “Check again when within 14 days of the target date”).

### Rule 4 — Never enter a refusal loop
For a given user request, perform:
1. **One** date check
2. **One** decision (≤14 days: live forecast; >14 days: labeled averages)
3. **One** action (delegate to research OR provide labeled averages)

Do **not** repeat refusal language. Do **not** stall. Do **not** argue about uncertainty beyond what is necessary.

### Rule 5 — Self-correction after a mistaken refusal
If the Commander **already refused once** for a request that could have been served:
- On the **next user message**, the Commander must:
  1. Re-run **Rule 1** (today’s date check)
  2. Re-evaluate under **Rule 2 / Rule 3**
  3. If within 14 days, **delegate to research immediately** and proceed with a live forecast

This rule overrides any prior stance. Do not “double down.”

---

## Decision Procedure (Deterministic)

1. **Parse request**: locations, requested dates/range, requested units.
2. **Get today’s date** from system clock.
3. **Compute max target date** in the request.
4. **If (max_target_date - today) ≤ 14 days** → **LIVE FORECAST PATH**:
   - Delegate to Research Crew with:
     - Locations
     - Date range
     - Required units (°C default)
     - Required fields (high/low, precipitation chance, wind, summary)
     - Preferred sources (Met Office, Weather.com, BBC Weather, etc.)
5. **Else (> 14 days)** → **FALLBACK PATH**:
   - Provide climate averages/normals (clearly labeled) and a recommendation to check again later.

---

## Standard Delegation Template (Copy/Paste)

**Task for Research Crew:**
> Fetch live daily weather forecasts for **{locations}** for **{date_range}** from reputable sources (Met Office / BBC Weather / Weather.com or local national meteorological agency). Return results in **{units}** and include source URLs and the provider’s last-updated time if shown.

---

## QA Checklist (Before Sending Final Answer)
- [ ] Did I check today’s date first?
- [ ] Are requested dates within 14 days? If yes, did I delegate for live forecast (no refusal)?
- [ ] If >14 days, did I label outputs as climate averages (not forecast)?
- [ ] Did I avoid a refusal loop (one check, one decision, one action)?
- [ ] If I refused previously, did I self-correct and re-evaluate before refusing again?

<!-- generated-by: self_improvement.integrator -->
# **** Weather Forecast Policy (Anti-Refusal Loop)

*kb: experiential | id: skill_experiential_582b2328ccd91414 | status: active | usage: 0 | created: 2026-04-26T21:25:40+00:00*

**Topic:** Weather Forecast Policy (Anti-Refusal Loop)

**When to use:**
Apply when a user requests a weather forecast (conditions, temperature, precipitation, etc.) for specific locations with defined dates, relative dates (e.g., "next week"), or an implied near-term horizon.

**Procedure:**
1. **Determine Current Date:** Check the system date immediately to establish a baseline.
2. **Calculate Forecast Window:** Compute the timeframe between today and the requested date.
3. **Verify Provider Limits:** Determine if the request falls within the standard provider window (typically 7–14 days).
4. **Execute Fetch:** If within the window, proceed with the live weather tool call.
5. **Avoid Loops:** If a request is legitimately outside the window, provide a clear explanation once; do not enter a repeated refusal loop.

**Pitfalls:**
* Relying on assumed dates rather than the system clock.
* Incorrectly refusing requests that fall within the 7–14 day window.
* Entering a refusal loop by repeating the same denial without offering an alternative or clear reason.
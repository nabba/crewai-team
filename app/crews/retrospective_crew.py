"""
retrospective_crew.py — Meta-cognitive retrospective analysis.

Runs after primary task execution to analyze what went well, what
didn't, and generate improvement policies for future runs.

Implements the meta-cognitive self-improvement loop from Evers et al. 2025.
"""

import json
import logging
import re
from crewai import Task, Crew, Process
from app.agents.introspector import create_introspector
from app.firebase_reporter import crew_started, crew_completed, crew_failed
from app.memory.chromadb_manager import retrieve_with_metadata
from app.policies.policy_loader import store_policy

logger = logging.getLogger(__name__)


class RetrospectiveCrew:
    def run(self) -> str:
        """Run retrospective analysis on recent execution data."""
        from app.conversation_store import estimate_eta
        task_id = crew_started(
            "self_improvement", "Retrospective analysis",
            eta_seconds=estimate_eta("retrospective"),
        )

        try:
            # Gather execution traces
            traces = self._gather_traces()
            if not traces:
                crew_completed(
                    "self_improvement", task_id, "No traces to analyze"
                )
                return "No recent execution data to analyze."

            introspector = create_introspector()

            analysis_task = Task(
                description=(
                    f"Analyze these execution traces from the AI agent team and "
                    f"generate improvement policies.\n\n"
                    f"## Execution Traces\n{traces}\n\n"
                    f"## Your Task\n"
                    f"1. Identify RECURRING patterns (not one-off issues).\n"
                    f"2. For each pattern, generate a policy in JSON format:\n"
                    f'   {{"trigger": "when X happens", '
                    f'"action": "do Y instead", '
                    f'"evidence": "because Z was observed N times"}}\n\n'
                    f"3. Return a JSON array of 1-5 policies:\n"
                    f'   [{{"trigger": "...", "action": "...", "evidence": "..."}}, ...]\n\n'
                    f"4. Focus on actionable, specific policies — not platitudes.\n"
                    f"5. Check if similar policies already exist before creating duplicates.\n\n"
                    f"Reply with ONLY the JSON array."
                ),
                expected_output="JSON array of improvement policies.",
                agent=introspector,
            )

            crew = Crew(
                agents=[introspector],
                tasks=[analysis_task],
                process=Process.sequential,
                verbose=False,
            )
            raw = str(crew.kickoff()).strip()

            # Parse and store policies
            policies_stored = self._parse_and_store_policies(raw)

            summary = f"Retrospective complete: {policies_stored} new policies generated."
            crew_completed("self_improvement", task_id, summary)
            return summary

        except Exception as exc:
            crew_failed("self_improvement", task_id, str(exc)[:200])
            logger.error(f"Retrospective failed: {exc}")
            return f"Retrospective analysis failed: {str(exc)[:200]}"

    def _gather_traces(self) -> str:
        """Collect recent execution data for analysis.

        Gathers self-reports, reflections, and belief states from
        ChromaDB to provide the Introspector with data to analyze.
        """
        sections = []

        # Self-reports (last 20)
        self_reports = retrieve_with_metadata("self_reports", "task assessment", n=20)
        if self_reports:
            report_lines = []
            for item in self_reports[:15]:
                try:
                    report = json.loads(item["document"])
                    role = report.get("role", "?")
                    conf = report.get("confidence", "?")
                    completeness = report.get("completeness", "?")
                    task = report.get("task_summary", "?")[:60]
                    blockers = report.get("blockers", "")
                    line = f"  [{role}] {task} | confidence={conf} completeness={completeness}"
                    if blockers:
                        line += f" | blockers: {blockers[:60]}"
                    report_lines.append(line)
                except (json.JSONDecodeError, KeyError):
                    continue
            if report_lines:
                sections.append("### Self-Reports\n" + "\n".join(report_lines))

        # Reflections from all agent types
        for role in ["researcher", "coder", "writer", "critic"]:
            reflections = retrieve_with_metadata(
                f"reflections_{role}", "lesson learned", n=10
            )
            if reflections:
                ref_lines = []
                for item in reflections[:5]:
                    try:
                        ref = json.loads(item["document"])
                        task = ref.get("task", "?")[:40]
                        lesson = ref.get("lesson", "?")[:80]
                        went_wrong = ref.get("went_wrong", "")[:60]
                        line = f"  [{role}] {task}: lesson={lesson}"
                        if went_wrong:
                            line += f" | went_wrong: {went_wrong}"
                        ref_lines.append(line)
                    except (json.JSONDecodeError, KeyError):
                        continue
                if ref_lines:
                    sections.append(
                        f"### {role.title()} Reflections\n" + "\n".join(ref_lines)
                    )

        # Belief states
        beliefs = retrieve_with_metadata("scope_beliefs", "agent state", n=10)
        if beliefs:
            belief_lines = []
            for item in beliefs[:8]:
                try:
                    belief = json.loads(item["document"])
                    agent = belief.get("agent", "?")
                    state = belief.get("state", "?")
                    needs = belief.get("needs", [])
                    line = f"  [{agent}] state={state}"
                    if needs:
                        line += f" | needs: {', '.join(needs[:3])}"
                    belief_lines.append(line)
                except (json.JSONDecodeError, KeyError):
                    continue
            if belief_lines:
                sections.append("### Belief States\n" + "\n".join(belief_lines))

        # Proactive trigger notes
        proactive = retrieve_with_metadata("scope_team", "PROACTIVE", n=10)
        if proactive:
            proactive_lines = []
            for item in proactive[:5]:
                doc = item.get("document", "")
                if "[PROACTIVE]" in doc:
                    proactive_lines.append(f"  {doc[:120]}")
            if proactive_lines:
                sections.append(
                    "### Proactive Triggers\n" + "\n".join(proactive_lines)
                )

        if not sections:
            return ""

        return "\n\n".join(sections)

    def _parse_and_store_policies(self, raw: str) -> int:
        """Parse JSON policy array and store in the policies scope.

        Returns the number of policies successfully stored.
        """
        from app.utils import safe_json_parse
        policies, err = safe_json_parse(raw)
        if policies is None:
            logger.warning(f"Retrospective output not valid JSON: {err} | {raw[:100]}")
            return 0

        if not isinstance(policies, list):
            policies = [policies]

        stored = 0
        for policy in policies[:5]:  # Cap at 5 policies per run
            if not isinstance(policy, dict):
                continue
            trigger = policy.get("trigger", "")
            action = policy.get("action", "")
            evidence = policy.get("evidence", "")

            if not trigger or not action:
                continue

            store_policy(
                trigger=trigger,
                action=action,
                evidence=evidence,
            )
            stored += 1
            logger.info(f"Stored policy: {trigger[:60]}")

        return stored

"""pim_crew.py — Personal Information Management crew (email, calendar, tasks)."""

from app.agents.pim_agent import create_pim_agent
from app.crews.base_crew import run_single_agent_crew

PIM_TASK_TEMPLATE = """\
Handle this personal information management task:

{user_input}

You have access to:
- Email (IMAP/SMTP): check inbox, read, send, search, organize, rank.
  - check_email — list/count by recency. Supports hours_back, days_back,
    from_sender, subject_contains, count_only. Use for "how many emails
    today" or "emails from Alice this week".
  - rank_emails — RANK BY IMPORTANCE (not recency). Combines bulk-marker
    analysis (List-Unsubscribe, noreply senders, marketing keywords) with
    personal-marker analysis (direct To:, threaded reply, action keywords)
    and an env-curated allowlist. USE THIS for "top N most important",
    "rank emails", "what should I read first", "important emails today".
  - read_email / search_email — full-content read or query by subject.
  - send_email — confirm details before executing.
  - organize_email — mark read/unread, archive, move.
- Calendar (macOS Calendar): list, create, search, delete events
- Tasks (local database): create, list, update, complete, search
- Kanban tickets (control_plane.tickets, Postgres) — the React dashboard's
  ticket system, distinct from the local-tasks DB above.  Use these tools
  when the user references the Kanban board, mentions tickets, or asks
  to move a ticket / task between workspaces / projects:
  - cp_list_tickets — list tickets in a project (default = active project).
  - cp_search_tickets — search by title/description across all projects.
  - cp_move_ticket — move a ticket to a different project (audit-logged).

  IMPORTANT: "move <task> to workspace X" requests almost always refer
  to the Kanban tickets, not the local tasks.db.  If list_tasks /
  search_tasks come back empty, do NOT report "no tasks found" — try
  cp_search_tickets / cp_list_tickets before concluding the item
  doesn't exist.

TOOL SELECTION FOR EMAIL RANKING:
If the user asks for emails ranked / sorted by importance, use rank_emails,
NOT check_email. check_email returns recent emails (newest first); it has
no notion of importance. rank_emails uses the importance scorer.

CRITICAL: You DO have email/calendar/task tools — they are loaded in your
tool list.  If a tool call fails, report the ACTUAL error message from the
tool.  NEVER respond with "integration is not connected" or "tool not
available" without first calling the tool and getting an error response.
If you are asked about emails/calendar/tasks, USE the tools before
answering.

Determine which tools are needed. Summarize findings concisely.
If the task involves sending email or creating events, confirm the details
in your response before executing.
"""


class PIMCrew:
    def run(self, task_description: str, parent_task_id: str = None, difficulty: int = 5) -> str:
        return run_single_agent_crew(
            crew_name="pim",
            agent_role="pim",
            create_agent_fn=create_pim_agent,
            task_template=PIM_TASK_TEMPLATE,
            task_description=task_description,
            expected_output="Completed PIM task with clear summary of actions taken.",
            parent_task_id=parent_task_id,
            difficulty=difficulty,
        )

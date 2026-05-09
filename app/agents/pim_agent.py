"""pim_agent.py — Personal Information Management agent (email, calendar, tasks)."""

from crewai import Agent
from app.agents._common import optional_tool_group
from app.llm_factory import create_specialist_llm
from app.tools.memory_tool import create_memory_tools
from app.tools.scoped_memory_tool import create_scoped_memory_tools
from app.tools.mem0_tools import create_mem0_tools
from app.souls.loader import compose_backstory


PIM_BACKSTORY = compose_backstory("pim")


def create_pim_agent(force_tier: str | None = None) -> Agent:
    llm = create_specialist_llm(max_tokens=4096, role="writing", force_tier=force_tier)
    memory_tools = create_memory_tools(collection="pim")
    scoped_tools = create_scoped_memory_tools("pim")
    mem0_tools = create_mem0_tools("pim")

    tools: list = [] + memory_tools + scoped_tools + mem0_tools

    # Email tools
    with optional_tool_group('pim', 'email_tools'):
        from app.tools.email_tools import create_email_tools
        email_tools = create_email_tools("pim")
        if email_tools:
            tools.extend(email_tools)

    # Calendar tools
    with optional_tool_group('pim', 'calendar_tools'):
        from app.tools.calendar_tools import create_calendar_tools
        cal_tools = create_calendar_tools("pim")
        if cal_tools:
            tools.extend(cal_tools)

    # Task tools
    with optional_tool_group('pim', 'task_tools'):
        from app.tools.task_tools import create_task_tools
        task_tools = create_task_tools("pim")
        if task_tools:
            tools.extend(task_tools)

    # Control-plane Kanban ticket tools (cp_list_tickets / cp_search_tickets
    # / cp_move_ticket).  Distinct from the SQLite task_tools above:
    # those operate on /app/workspace/tasks.db (no project_id), these
    # operate on control_plane.tickets in Postgres (the React Kanban
    # board, with project_id).  Without these the agent can't fulfil
    # "move that task to workspace X" requests — it would search the
    # wrong store and hallucinate "no tasks found" (live failure
    # 2026-05-09).
    with optional_tool_group('pim', 'cp_tickets'):
        from app.tools.control_plane_tickets_tool import create_cp_tickets_tools
        cp_ticket_tools = create_cp_tickets_tools("pim")
        if cp_ticket_tools:
            tools.extend(cp_ticket_tools)

    # Photos tools (macOS Photos.app via AppleScript through bridge)
    with optional_tool_group('pim', 'photos_tools'):
        from app.tools.photos_tools import create_photos_tools
        photo_tools = create_photos_tools("pim")
        if photo_tools:
            tools.extend(photo_tools)

    # Wiki tools
    with optional_tool_group('pim', 'wiki_tool_registry'):
        from app.tools.wiki_tool_registry import create_wiki_tools
        tools.extend(create_wiki_tools("read", "write"))

    return Agent(
        role="Personal Information Manager",
        goal="Manage email, calendar, and tasks. Summarize, prioritize, and organize.",
        backstory=PIM_BACKSTORY,
        llm=llm,
        tools=tools,
        max_execution_time=300,
        verbose=True,
    )

"""devops_agent.py — DevOps agent (build, test, deploy, CI/CD)."""

from crewai import Agent
from app.agents._common import optional_tool_group
from app.llm_factory import create_specialist_llm
from app.tools.code_executor import execute_code
from app.tools.file_manager import file_manager
from app.tools.web_search import web_search
from app.tools.memory_tool import create_memory_tools
from app.tools.scoped_memory_tool import create_scoped_memory_tools
from app.tools.mem0_tools import create_mem0_tools
from app.souls.loader import compose_backstory


DEVOPS_BACKSTORY = compose_backstory("devops")


def create_devops_agent(force_tier: str | None = None) -> Agent:
    llm = create_specialist_llm(max_tokens=4096, role="coding", force_tier=force_tier)
    memory_tools = create_memory_tools(collection="devops")
    scoped_tools = create_scoped_memory_tools("devops")
    mem0_tools = create_mem0_tools("devops")

    tools = [execute_code, file_manager, web_search] + memory_tools + scoped_tools + mem0_tools

    # Project builder tools
    with optional_tool_group('devops', 'project_builder_tools'):
        from app.tools.project_builder_tools import create_project_builder_tools
        builder_tools = create_project_builder_tools("devops")
        if builder_tools:
            tools.extend(builder_tools)

    # Deployment tools
    with optional_tool_group('devops', 'deployment_tools'):
        from app.tools.deployment_tools import create_deployment_tools
        deploy_tools = create_deployment_tools("devops")
        if deploy_tools:
            tools.extend(deploy_tools)

    # CI/CD generation tools
    with optional_tool_group('devops', 'ci_cd_tools'):
        from app.tools.ci_cd_tools import create_ci_cd_tools
        cicd_tools = create_ci_cd_tools("devops")
        if cicd_tools:
            tools.extend(cicd_tools)

    # Mobile app tools (Expo/React Native, PWA)
    with optional_tool_group('devops', 'mobile_tools'):
        from app.tools.mobile_tools import create_mobile_tools
        mob_tools = create_mobile_tools("devops")
        if mob_tools:
            tools.extend(mob_tools)

    # Bridge tools
    with optional_tool_group('devops', 'bridge_tools'):
        from app.tools.bridge_tools import create_bridge_tools
        bridge_tools = create_bridge_tools("devops")
        if bridge_tools:
            tools.extend(bridge_tools)

    return Agent(
        role="DevOps Engineer",
        goal="Scaffold projects, build, test, package, deploy, and configure CI/CD.",
        backstory=DEVOPS_BACKSTORY,
        llm=llm,
        tools=tools,
        max_execution_time=600,
        verbose=True,
    )

# MCP Capability Expansion Protocol

## Goal
Expand agent team capabilities dynamically by discovering and installing MCP servers when existing tools cannot fulfill a request.

## Workflow
1. **Gap Identification**: When a task requires a capability not present in `web_search`, `code_executor`, or current skills (e.g., interacting with GitHub, Google Calendar, Slack, or a specific database).
2. **Discovery**: Use `mcp_search_servers` with specific keywords (e.g., 'github', 'postgres', 'slack') to find compatible servers.
3. **Evaluation**: Review the server description and required `env_vars` to ensure it matches the task requirement.
4. **Integration**: Call `mcp_add_server` with the correct server name and necessary API keys provided in the context or requested from the user.
5. **Verification**: Use `mcp_list_servers` to verify the tools are active and call the new tool to complete the task.

## Best Practices
- Always check `mcp_list_servers` first to avoid duplicate installations.
- Document the newly added capability in team memory using `team_memory_store`.
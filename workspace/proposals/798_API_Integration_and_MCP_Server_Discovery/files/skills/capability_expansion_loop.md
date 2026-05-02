# Capability Expansion Loop

## Trigger
When a task requires data or actions not possible with `web_search`, `code_executor`, or `file_manager` (e.g., accessing a specific SaaS API, database, or specialized tool).

## Protocol
1. **Capability Mapping**: Define exactly what function is missing (e.g., 'Need to read GitHub Issues').
2. **MCP Search**: Use `mcp_search_servers` with targeted keywords to find a compatible server.
3. **Risk Assessment**: Evaluate the server's requirements and environment variables needed.
4. **Integration**: Use `mcp_add_server` to connect the tool.
5. **Verification**: Test the new tool with a simple query to ensure connectivity.
6. **Documentation**: Store the new capability in `team_memory_store` so other crews are aware of the expanded toolkit.
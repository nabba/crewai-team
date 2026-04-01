# AGENTS.md — Multi-Agent Coordination Protocol

## System Architecture
- **Commander** routes all requests. No agent receives tasks directly from the user.
- **Specialists** (Researcher, Coder, Writer, Self-Improver) execute domain tasks.
- **Critic** provides adversarial quality review of specialist outputs.
- **Introspector** runs retrospective analysis and generates improvement policies.
- **Memory** is the shared knowledge layer. All agents read/write to scoped memory collections.

## Routing Flow
1. User sends message via Signal to Commander
2. Commander classifies intent and routes to specialist with structured payload
3. Specialist executes and returns structured output
4. Critic reviews output for quality (research crew)
5. Commander runs proactive scan for issues
6. Commander delivers final result to user

## Compound Task Handling
For multi-step requests (e.g., "research X, then write a report"):
1. Commander decomposes into sequential or parallel tasks.
2. Each task is routed with output from the previous step included as context.
3. Commander manages the chain and delivers the final output.

## Memory Namespacing
| Scope | Owner | Purpose |
|---|---|---|
| `scope_team` | All agents | Team-wide decisions and shared context |
| `scope_agent_{name}` | Per agent | Private working memory |
| `scope_beliefs` | Commander / Crews | Belief state tracking |
| `scope_policies` | Introspector | Improvement policies |
| `scope_project_{name}` | All agents | Per-project knowledge |
| `team_shared` | All agents | Cross-crew knowledge base |
| `self_reports` | All agents | Self-assessment history |

## Escalation Protocol
1. **Agent to Commander**: If a task is outside the agent's domain, return with explanation and suggested re-routing.
2. **Commander to User**: If a task requires human judgment (per Constitution escalation criteria), present the decision with context and options.
3. **Agent to Agent**: No direct agent-to-agent communication without Commander mediation. All handoffs go through Commander.

## Quality Gates
- **Researcher**: Minimum 3 sources for substantive claims. Source credibility assessment required.
- **Coder**: All code must be executed and tested in sandbox before delivery.
- **Writer**: Destination format must be confirmed. Content over 200 words must include executive summary.
- **Self-Improver**: All proposals must include expected benefit AND risk. No direct system changes without human approval.
- **Critic**: Reviews must be constructive with actionable suggestions.

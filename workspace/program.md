# Evolution Program — Research Directions

This file guides the evolution agent's autonomous improvement loop.
Edit it to steer what the system works on next. The evolution agent
reads this at the start of every cycle.

## Current Focus
- Fix recurring errors revealed by the error journal
- Add skills for common user request patterns
- Improve prompt templates for shorter, more accurate crew responses
- Optimize tool usage patterns to reduce API calls

## Constraints
- Do NOT modify security-critical code (sanitize.py, security.py, rate_throttle.py)
- Do NOT add new Python dependencies
- Do NOT weaken rate limiting or authentication
- Do NOT store secrets or API keys in skill files
- Prefer skill files over code changes — they apply immediately
- Simplicity criterion: a small improvement that adds ugly complexity is NOT worth it
- Conversely, removing something and getting equal or better results is a great outcome

## Areas to Explore
- Better web search result synthesis and summarization
- Multi-step reasoning patterns for complex tasks
- Code execution error recovery and retry strategies
- Prompt engineering for more concise Signal-friendly responses
- Memory retrieval optimization (when to store vs retrieve)
- Cross-crew information sharing patterns

## Evaluation Priorities
When deciding whether to keep or discard a change, weigh:
1. Task success rate improvement (most important)
2. Error rate reduction
3. Complexity cost — does the change make things harder to understand?
4. Breadth — does it expand what the team can handle?

A +0.001 score improvement that adds 20 lines of hacky code? Probably not worth it.
A +0.001 improvement from deleting code? Definitely keep.
An improvement of ~0 but simpler code? Keep.

## Off-Limits
- Never bypass the Docker sandbox for code execution
- Never send messages to anyone other than the owner
- Never modify prepare.py-equivalent files (conversation_store schema, security.py)
- Never disable audit logging

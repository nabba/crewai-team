# Automated Testing and Regression Protocol

## Objective
Ensure code stability and prevent regressions when modifying the system or adding new tools.

## Protocol
1. **Baseline Capture**: Before modifying any existing tool, the coding crew must write a small test script that captures the current correct output.
2. **Unit Test Generation**: Every new tool must be accompanied by a `tests/test_[tool_name].py` file containing:
    - Happy path cases.
    - Edge cases (empty input, malformed JSON, timeouts).
    - Error handling cases (API 404s, 500s).
3. **Execution**: Run tests via `code_executor` before handing off to the writing crew.
4. **Verification**: The self_improvement crew must verify that the test coverage is sufficient before marking a task as complete.
# TDD for AI Coding Crews

## Protocol
1. **Test First**: Before implementing a fix for a bug or adding a feature, the agent must write a Python test script that reproduces the bug or defines the success criteria.
2. **Execution**: Run the test in the `code_executor` to confirm it fails.
3. **Implementation**: Write the production code.
4. **Verification**: Run the test again to confirm it passes.
5. **Persistence**: Store the test script in the `/tests` directory of the workspace to allow future regression checks.

## Validation Criteria
- No code is submitted to the writing crew for documentation unless a corresponding test has passed in the sandbox.
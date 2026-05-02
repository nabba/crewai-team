# Regression Testing & Quality Assurance

## Goal
Ensure code stability across iterations in the Docker sandbox.

## Protocol
1. **Baseline Capture**: Before applying a fix, the `code_executor` must run a script that captures the current output of all primary functions.
2. **Test Case Generation**: Create a `test_suite.py` containing edge cases and expected outputs.
3. **Verification**: After modifying code, run the `test_suite.py`. Any deviation from the expected baseline must be analyzed.
4. **Certification**: A task is only marked complete when all tests in the suite pass.
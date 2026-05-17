# Test-Driven Development (TDD) Workflow

## Objective
Increase the reliability of code produced by the `coding` crew by ensuring functional requirements are validated via automated tests.

## Workflow
1. **Requirement Analysis**: Identify all edge cases and expected outputs.
2. **Test First**: Write a `test_*.py` script using `pytest` or `unittest` that fails on the current state.
3. **Implementation**: Write the minimum amount of code in the `code_executor` to make the test pass.
4. **Refactor**: Optimize the code for performance and readability while keeping tests passing.
5. **Verification**: Run the full test suite and attach the output to the final delivery.
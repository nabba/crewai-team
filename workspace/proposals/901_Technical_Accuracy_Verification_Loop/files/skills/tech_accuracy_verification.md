# Technical Accuracy Verification Loop

## Goal
Ensure 100% alignment between the final written output and the technical implementation.

## Verification Steps
1. **Claim Extraction**: The Writing crew highlights all technical claims (e.g., 'The API supports X', 'The code optimizes Y') in the draft.
2. **Code Validation**: The Coding crew takes these claims and writes a minimal test case in the Docker sandbox to prove/disprove each claim.
3. **Conflict Resolution**: Any claim that fails the test is flagged. The Writer must revise the text based on the actual sandbox output.
4. **Sign-off**: The task is not marked complete until the Coding crew provides a 'Technical Verified' tag.
# Adversarial Fact-Checking Loop

## Problem
High risk of 'hallucination drift' where the writing crew interprets research findings incorrectly.

## Solution
Implement the 'Challenge-Verify' cycle:
1. **Drafting**: Writing crew produces a technical claim.
2. **Challenge**: Research crew is tasked specifically to find a source that contradicts the claim.
3. **Resolution**: If a contradiction is found, the research crew provides the correction. If no contradiction is found after 3 distinct search queries, the claim is marked 'Verified'.
4. **Audit Trail**: All verified claims must include a URL reference in the final document.
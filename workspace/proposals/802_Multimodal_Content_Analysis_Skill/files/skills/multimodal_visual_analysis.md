# Multimodal Visual Analysis Protocol

## Objective
To extract structured information from visual sources where text-only fetching fails (e.g., dashboards, infographics, complex PDFs).

## Workflow
1. **Visual Capture**: Use `browser_screenshot` to capture the target UI element or page.
2. **Contextual Mapping**: Combine the screenshot with `browser_fetch` HTML structure to map visual elements to DOM elements.
3. **Verification**: Cross-reference transcript/text data with visual evidence to ensure accuracy.
4. **Synthesis**: Convert visual patterns (e.g., red/green status indicators) into structured text for the writing crew.
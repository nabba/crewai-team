# PSP Market Intelligence Workflow

## Objective
Transform raw web research on Regional Payment Service Providers (PSPs) into structured, enriched lead lists.

## Workflow Steps
1. **Identification**: Use `web_search` to find regional PSPs based on geographic and sector criteria.
2. **Deep Dive**: Use `web_fetch` on 'About Us', 'Pricing', and 'API Documentation' pages to extract specific capabilities.
3. **Structuring**: Pass extracted text to the `coding` crew to parse into a structured schema (Company Name, HQ, Primary Market, API Availability, Contact Email).
4. **Validation**: Use the `skill__validating_file_existence_before_automation_tasks` to ensure the resulting lead file is correctly saved in the workspace.
5. **Reporting**: Use `automated_pdf_report_generation_with_graphics` to create a summary of the regional landscape.
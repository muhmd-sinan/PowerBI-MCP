# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MCP (Model Context Protocol) server enabling Claude Code to programmatically create, modify, and manage Power BI dashboards from natural language instructions.

## Technical Direction

**Primary approach: PBIP (Power BI Project) file manipulation** — not UI automation.

1. Generate/edit PBIP project files (JSON-based dashboard/page/visual structures)
2. MCP server exposes tools for dataset import, data modeling (DAX), visual creation, layout, and export
3. User opens generated PBIP in Power BI Desktop for rendering

Architecture: `Claude Code → MCP Server → PBIP file manipulation → Power BI Desktop renders result`

## MCP Tool Categories

- **Dataset**: import_dataset, refresh_dataset, list_tables, get_schema
- **Modeling**: create_measure, create_relationship, create_column, run_dax
- **Visualization**: create_dashboard_page, create_card, create_chart, create_table_visual, create_slicer, apply_theme
- **Export**: export_pdf, export_png, save_project, publish_workspace

## Key Decisions

- Avoid UI automation as primary control mechanism
- PBIP format chosen for reliability and version control compatibility
- Analogous to how Blender MCP manipulates Blender programmatically via its API

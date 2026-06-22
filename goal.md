# Goal: Claude Code ↔ Power BI MCP Integration

## Objective

Build an MCP (Model Context Protocol) server that enables Claude Code to control Microsoft Power BI Desktop similarly to how Claude Code can control Blender through an MCP server.

The end goal is for Claude Code to create, modify, and manage Power BI dashboards directly from natural language instructions without requiring manual interaction inside Power BI.

## Success Criteria

A user should be able to type commands such as:

* "Create a sales dashboard from sales.csv"
* "Import this dataset and generate KPI cards for Revenue, Profit, and Orders"
* "Create a line chart showing monthly revenue trends"
* "Add a slicer for Region and Product Category"
* "Generate a complete executive dashboard with insights"
* "Export the dashboard as PDF"

Claude Code should execute these tasks automatically through the MCP server.

---

## Core Capabilities

### 1. Power BI Control Layer

Create a mechanism to control Power BI Desktop programmatically.

Potential approaches:

* Power BI PBIP project manipulation
* Power BI Desktop automation
* PowerShell integration
* UI automation
* Tabular Object Model (TOM)
* XMLA endpoints (where applicable)

---

### 2. Dataset Management

Functions:

* Import CSV files
* Import Excel files
* Import SQL data sources
* Refresh datasets
* Create relationships
* Detect schema automatically

Example MCP Tools:

* import_dataset()
* refresh_dataset()
* create_relationship()

---

### 3. Data Modeling

Functions:

* Create calculated columns
* Create measures
* Create DAX expressions
* Build star schemas
* Manage tables

Example MCP Tools:

* create_measure()
* create_calculated_column()
* create_table()

---

### 4. Dashboard Generation

Functions:

* Create pages
* Create visuals
* Position visuals
* Apply themes
* Create slicers and filters

Supported visuals:

* KPI Cards
* Tables
* Matrix
* Line Charts
* Bar Charts
* Pie Charts
* Area Charts
* Scatter Charts
* Maps

Example MCP Tools:

* create_visual()
* move_visual()
* resize_visual()
* apply_theme()

---

### 5. AI Dashboard Builder

Claude Code should be able to:

1. Analyze dataset structure
2. Identify useful KPIs
3. Suggest measures
4. Select appropriate chart types
5. Generate a complete dashboard automatically

Example:

User:
"Build a professional executive dashboard from this sales dataset."

Claude Code:

* Analyzes columns
* Creates measures
* Designs layout
* Generates visuals
* Produces finished dashboard

---

## MCP Tool Design

### Dataset Tools

* import_dataset
* refresh_dataset
* list_tables
* get_schema

### Modeling Tools

* create_measure
* create_relationship
* create_column
* run_dax

### Visualization Tools

* create_dashboard_page
* create_card
* create_chart
* create_table_visual
* create_slicer
* apply_theme

### Export Tools

* export_pdf
* export_png
* save_project
* publish_workspace

---

## Architecture

Claude Code
↓
MCP Server
↓
Power BI Control Layer
↓
Power BI Desktop / PBIP Project
↓
Generated Dashboard

---

## Recommended Technical Direction

Avoid UI automation as the primary solution.

Preferred approach:

1. Use PBIP project format.
2. Reverse engineer dashboard/page/visual JSON structures.
3. Build an MCP server that edits PBIP files directly.
4. Allow Claude Code to generate and modify dashboard artifacts.
5. Open the generated PBIP in Power BI Desktop for rendering.

This approach is more reliable, version controllable, and closer to how Blender MCP manipulates Blender programmatically.

## Final Deliverable

A production-ready MCP server that allows Claude Code to:

* Create Power BI projects
* Import datasets
* Create measures
* Generate visuals
* Build complete dashboards
* Save and export reports

using natural language commands entirely from Claude Code.

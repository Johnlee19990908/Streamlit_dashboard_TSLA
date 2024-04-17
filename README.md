# Streamlit Dashboard for Coherix Vision System BI Project

## Project Overview

### Problem Statement
The 2K Betamate & RTV dispense process at B2-PAK1-32000 lacks quantified/visual inspection in subsequent stations, leading to potential issues such as missing, thin, or thick beads that deviate from customer specifications.

### Interim Controls
- **2K Betamate:**
  - Ratio checks conducted 4 times per shift to confirm the mix ratio is within specifications.
- **RTV:**
  - Weight shots checked 4 times per shift to ensure accuracy compared to equipment readings.

### Current Challenges
Previously, identifying failures detected by Coherix required manual checks of serial numbers via MOS and individual pack inspections using Tangent PC through VNC viewer. This approach lacked efficiency and statistical process control, hindering process monitoring and improvement efforts.

## Solution Implemented

- Developed a Python-based pipeline for the Dispense Vision system, automating data extraction and processing.
- Deployed 2 SPC dashboards for real-time monitoring of bead parameters, daily FPY (First Pass Yield), and failure heatmaps.
- This solution enables quick identification and analysis of failures, leading to improved process efficiency and significant time savings for process engineers (up to 90% reduction in searching and monitoring time).

## Dashboards Overview

### 32K Coherix Raw Data Dashboard:
- Aggregates data from multiple packs into a single chart, providing a holistic view of process trends and enhancing process control.
- Facilitates comprehensive analysis of height and width for both RTV and 2K simultaneously, streamlining bead measurements.

### 32K FPY and Failure Heatmap & Pareto Dashboard:
- Presents daily FPY, pack zone heatmaps, and Pareto charts of high-failure areas for a comprehensive process overview.
- Date Range feature allows engineers to filter data for specific periods, essential for assessing the impacts of changes and continuous performance enhancement.

"""HiSim sweep visualization package.

Layering rules:
- ``sweep_data`` is pandas-only and never imports rendering libraries.
- ``sweep_figures`` returns ``plotly.graph_objects.Figure`` objects only
  and never writes to disk.
- ``sweep_report`` composes HTML and is the only module aware of jinja2.
- A future dashboard module imports the layers above; it never
  reimplements data or figure logic.
"""

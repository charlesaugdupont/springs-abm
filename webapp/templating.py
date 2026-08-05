"""Shared Jinja2Templates instance - a single place to register custom
filters so every router/page gets them consistently (e.g. tojson, used to
embed simulation result data as JSON for Plotly to consume client-side)."""
import json

from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="webapp/templates")
templates.env.filters["tojson"] = lambda v: json.dumps(v)

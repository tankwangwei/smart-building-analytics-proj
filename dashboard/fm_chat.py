"""'Ask the data': a chat assistant for the facilities team, using Google Gemini.

Gemini answers questions by calling read-only tools over the dashboard's own data (plain Python functions defined in
app.py), so every number comes from the analysis rather than from the model's memory. The conversation history is kept
on our side (stateless `store=False` requests) and replayed unchanged each turn. The dashboard's current view (floor,
timeframe, opening hours) travels with each new question, so the system instruction never changes mid-conversation.

API key, first found wins: Streamlit secrets (GEMINI_API_KEY in .streamlit/secrets.toml), the GEMINI_API_KEY
environment variable, or a key pasted into the app (kept in this browser session only, never written to disk).
Create a key at https://aistudio.google.com (API keys). GEMINI_MODEL in secrets overrides the model.
"""
import inspect
import json
import os
import re

from google import genai

DEFAULT_MODEL = "gemini-3.8-flash"
MAX_TOOL_ROUNDS = 12

SYSTEM_PROMPT = """You are the facilities-management (FM) data assistant for Chamchuri 5, an administrative office \
building at Chulalongkorn University, Bangkok. You answer questions from the FM team about air-conditioning (AC) energy, \
anomalies, comfort and demand forecasts, so they don't have to analyse the data by hand.

Data you can reach (through tools only):
- CU-BEMS 1-minute building data, analysed at 15-minute and hourly resolution. Reporting period 1 Jul – 31 Dec 2019.
- Floors 3, 4, 5 and 7, each with 4 AC units in zones 1, 2, 4 and 5 (zone 3 is stairs, no AC): 16 units in total.
- Each unit's normal pattern was learned from Jul 2018 – Jun 2019.
- Outdoor weather from Open-Meteo; Thai public holidays; the building's opening hours (set in the dashboard).

Key definitions:
- Flags: a 15-minute reading is flagged if it is unusually high for its time slot, runs while the unit is normally off \
or the building is closed (after hours), or stays above normal for 2 hours or more (persistent). Consecutive flags form \
an event. Flags are potential anomalies to check, not diagnosed faults.
- While the building is closed (outside opening hours, closed days, public holidays) normal AC use counts as zero, so \
any AC running then is waste.
- Excess kWh: energy above the unit's normal pattern during flagged periods; excess THB = excess kWh x the tariff.
- Triage score = excess THB x confidence / effort. Confidence (0-100) blends rule agreement, duration, recurrence, \
weather and data quality (and past close-outs). Effort is how hard the fix is: Low = schedule or timer, Medium = \
inspect, High = repair. Priority: High = the items that together make up the first half of the total triage score, \
Medium = up to 80%, Low = the rest.
- Forecast: day-ahead hourly AC demand, made with data up to the midnight before. Public holidays are forecast as zero.
- Costs use the dashboard's tariff (returned by the tools; default 4.2 THB/kWh) unless the user gives another.

How to answer:
- Get every number from a tool. Never estimate or invent figures; if no tool can answer, say what is missing.
- Each question starts with the dashboard's current view in brackets. Use that scope and timeframe unless the user \
asks about something else; pass dates as YYYY-MM-DD and keep them within 2019-07-01 to 2019-12-31 (the forecast \
tool also covers 2020-01-01).
- Lead with the answer and its key number, then a few short bullets or a small table. Name units like \
"Floor 7 · Zone 1 · AC4", give units (kWh, kW, °C, THB) and say which period the figures cover.
- When useful, end with one concrete action for the FM team (e.g. check a timer, inspect a filter).
- Be clear about uncertainty: savings are estimates; flags are leads to check."""

_JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean"}


def _secret(streamlit_secrets, name):
    try:
        return streamlit_secrets.get(name)
    except Exception:  # no secrets.toml at all
        return None


def get_api_key(streamlit_secrets):
    """Key from Streamlit secrets or the environment; None if neither is set."""
    return (_secret(streamlit_secrets, "GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY"))


def get_model(streamlit_secrets):
    return _secret(streamlit_secrets, "GEMINI_MODEL") or os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL


def function_declaration(fn):
    """A Gemini function declaration built from a Python function's signature and Google-style docstring."""
    doc = inspect.getdoc(fn) or ""
    summary, _, args_text = doc.partition("Args:")
    arg_docs = {}
    for name, text in re.findall(r"^\s*(\w+):\s*(.+(?:\n(?!\s*\w+:).+)*)", args_text, flags=re.MULTILINE):
        arg_docs[name] = " ".join(text.split())
    properties, required = {}, []
    for name, param in inspect.signature(fn).parameters.items():
        properties[name] = {"type": _JSON_TYPES.get(param.annotation, "string"), "description": arg_docs.get(name, "")}
        if param.default is inspect.Parameter.empty:
            required.append(name)
    return {"type": "function", "name": fn.__name__, "description": " ".join(summary.split()),
            "parameters": {"type": "object", "properties": properties, "required": required}}


def _run_tool(tools_by_name, name, arguments):
    fn = tools_by_name.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool '{name}'."})
    try:
        return fn(**(arguments or {}))
    except TypeError as error:  # wrong or missing arguments: tell the model so it can retry
        return json.dumps({"error": f"Bad arguments for {name}: {error}"})


def ask(api_key, model, history, question, tools):
    """Run one question through Gemini with the dashboard tools.

    history: the conversation so far, as interaction steps (replayed unchanged). Returns
    (answer_text, new_history_entries, tools_used). Raises the SDK's errors for the caller to show.
    """
    client = genai.Client(api_key=api_key)
    tools_by_name = {fn.__name__: fn for fn in tools}
    declarations = [function_declaration(fn) for fn in tools]
    turn = [{"type": "user_input", "content": [{"type": "text", "text": question}]}]
    tools_used = []
    for _ in range(MAX_TOOL_ROUNDS):
        interaction = client.interactions.create(
            model=model,
            system_instruction=SYSTEM_PROMPT,
            tools=declarations,
            input=history + turn,
            store=False,  # stateless: we keep and resend the history ourselves
        )
        calls = []
        for step in interaction.steps or []:
            turn.append(step.model_dump())
            if step.type == "function_call":
                calls.append(step)
        if not calls:
            text = (interaction.output_text or "").strip()
            return text or "No answer came back. Please try rephrasing the question.", turn, tools_used
        for call in calls:
            tools_used.append(call.name)
            turn.append({"type": "function_result", "call_id": call.id, "name": call.name,
                         "result": _run_tool(tools_by_name, call.name, call.arguments)})
    return ("I looked at the data but ran out of steps before answering. Try a more specific question.",
            turn, tools_used)


def error_message(error):
    """A plain-language message for an API error."""
    status = getattr(error, "status_code", None) or getattr(error, "code", None)
    if status in (401, 403):
        return "The Gemini API key was rejected. Check that it is correct and enabled in Google AI Studio."
    if status == 429:
        return ("Too many requests (the free tier allows a limited number per minute and per day). Wait a minute "
                "and try again.")
    if status == 404:
        return "The Gemini model wasn't found. Set GEMINI_MODEL in .streamlit/secrets.toml to a current model name."
    if status == 400:
        return (f"The request was rejected ({getattr(error, 'message', error)}). Clearing the conversation usually "
                "fixes this.")
    if status and status >= 500:
        return f"The Gemini API had a server error ({status}). Try again shortly."
    if "connect" in type(error).__name__.lower() or "NoResponse" in type(error).__name__:
        return "Couldn't reach the Gemini API. Check the internet connection."
    return f"Something went wrong: {error}"

# llm_client.py
import os
import re
import traceback
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Optional, Tuple, Dict, Any

# Try to import new OpenAI SDK client. Provide graceful fallback to older API.
try:
    from openai import OpenAI as OpenAIClient
    NEW_OPENAI_SDK = True
except Exception:
    NEW_OPENAI_SDK = False

# try to import legacy top-level module for older versions (0.x)
try:
    import openai as legacy_openai
    LEGACY_OPENAI = True
except Exception:
    LEGACY_OPENAI = False

# Read API key from file config/openai_key.txt
def _read_key():
    p = os.path.join("config", "openai_key.txt")
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as fh:
                k = fh.read().strip()
                return k if k else None
        except Exception:
            pass

    # Fallback: check environment variable
    return os.getenv("OPENAI_API_KEY")

OPENAI_API_KEY = _read_key()

# Initialize client object for new SDK if available
_client = None
if NEW_OPENAI_SDK and OPENAI_API_KEY:
    try:
        _client = OpenAIClient(api_key=OPENAI_API_KEY)
    except Exception:
        _client = None

def llm_available() -> bool:
    """
    True if we have either the new OpenAI client configured or the legacy module available.
    """
    if OPENAI_API_KEY is None:
        return False
    if NEW_OPENAI_SDK and _client is not None:
        return True
    if LEGACY_OPENAI:
        return True
    return False

# Safe prompt builder import; fallback if prompts.py not present
try:
    from prompts import build_parser_prompt
except Exception:
    def build_parser_prompt(user_query: str, known_categories: Optional[list] = None) -> str:
        known = ", ".join(known_categories[:40]) if known_categories else "none"
        return (
            "You are a strict XML parser for short banking queries. "
            "Return EXACTLY one XML block <result>...</result> on the first line. No extra commentary.\n\n"
            "Parse into fields: intent (TOP_CATEGORIES, SPEND_ON_CATEGORY, SPEND_TOTAL_PERIOD, TOP_MERCHANTS, RECURRING, INCOME, BALANCE, UNKNOWN),\n"
            "category, n (optional integer), months (optional integer), month (optional integer), years (optional integer), plot (true/false), explain (short explanation).\n\n"
            f"Known categories: {known}\n\nUser query:\n{user_query}\n\n"
            "If unsure, return intent=UNKNOWN and empty category. Example output:\n"
            "<result><intent>SPEND_ON_CATEGORY</intent><category>groceries</category><months>3</months><plot>false</plot><explain>I'll fetch monthly spend for groceries over the last 3 months.</explain></result>\n"
        )

# Resilient _call_chat with fallback + logging
def _call_chat(messages: list, model: str = "gpt-3.5-turbo", max_tokens: int = 240, temperature: float = 0.0) -> str:
    """
    Call chat completion using the available client.
    - Tries new OpenAI SDK path first (wrapped in try/except).
    - Falls back to legacy openai if available.
    - On error writes a short diagnostic to /tmp and raises RuntimeError.
    """
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    # Try new SDK first (if configured)
    if NEW_OPENAI_SDK and _client is not None:
        try:
            resp = _client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens, temperature=temperature)
            return resp.choices[0].message.content
        except Exception as e_new:
            # diagnostic for new SDK failure
            errpath = f"/tmp/spendly_openai_newsdk_err_{ts}.log"
            try:
                with open(errpath, "w", encoding="utf-8") as fh:
                    fh.write("New SDK call failed:\n")
                    fh.write(f"{type(e_new).__name__}: {str(e_new)}\n")
                    fh.write("Messages (truncated):\n" + str(messages)[:2000])
            except Exception:
                pass
            # fall through to legacy attempt

    # Legacy openai path
    if LEGACY_OPENAI:
        try:
            if OPENAI_API_KEY:
                try:
                    legacy_openai.api_key = OPENAI_API_KEY
                except Exception:
                    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
            resp = legacy_openai.ChatCompletion.create(model=model, messages=messages, max_tokens=max_tokens, temperature=temperature)
            return resp.choices[0].message.content
        except Exception as e_legacy:
            errpath = f"/tmp/spendly_openai_legacy_err_{ts}.log"
            try:
                with open(errpath, "w", encoding="utf-8") as fh:
                    fh.write("Legacy SDK call failed:\n")
                    fh.write(traceback.format_exc())
                    fh.write("\nMessages (truncated):\n" + str(messages)[:2000])
            except Exception:
                pass
            raise RuntimeError(f"Both new & legacy OpenAI calls failed. See logs: {errpath}")

    # No client available
    raise RuntimeError("No OpenAI client available (new SDK or legacy openai).")

# simple_chat wrapper expected by responder.generate_guided_advice()
def simple_chat(messages: list, model: str = "gpt-3.5-turbo", max_tokens: int = 240, temperature: float = 0.2) -> str:
    """
    Minimal wrapper returning assistant text for a simple messages list.
    Keeps the interface tiny so `responder.py` can call it for short rewrites.
    """
    try:
        return _call_chat(messages, model=model, max_tokens=max_tokens, temperature=temperature)
    except Exception as e:
        return f"LLM call failed: {type(e).__name__}: {e}"

# parse XML into typed dict
def parse_llm_xml(xml_text: str) -> Tuple[Optional[Dict[str,Any]], Optional[str]]:
    """
    Extract first <result>...</result> block and parse fields into a dict.
    Returns (parsed_dict, explanation_text) or (None, None) on parse failure.
    """
    m = re.search(r"(<result[\s\S]*?</result>)", xml_text, re.DOTALL)
    if not m:
        return None, None
    block = m.group(1)
    try:
        root = ET.fromstring(block)
    except Exception:
        return None, None

    def txt(tag):
        el = root.find(tag)
        return el.text.strip() if (el is not None and el.text is not None) else None

    parsed: Dict[str, Any] = {}
    parsed["intent"] = (txt("intent") or "UNKNOWN").strip().upper()
    parsed["category"] = txt("category") or None
    parsed["category_confidence"] = (txt("category_confidence") or None)

    def to_int(x):
        try:
            return int(x)
        except Exception:
            return None

    parsed["n"] = to_int(txt("n"))
    parsed["months"] = to_int(txt("months"))
    parsed["month"] = to_int(txt("month"))
    parsed["years"] = to_int(txt("years"))
    parsed["plot"] = True if (txt("plot") or "").strip().lower() in ("true", "1", "yes") else False
    parsed["explain"] = txt("explain") or ""
    return parsed, parsed.get("explain", "")

# main entry: ask LLM with examples and return typed dict + explanation (diagnostics)
def llm_parse_query_xml(user_query: str, known_categories: Optional[list] = None) -> Tuple[Optional[Dict[str,Any]], Optional[str]]:
    """
    Ask the LLM for compact XML. On failure, returns (None, explanation_string).
    Also writes raw LLM output + diagnostics to /tmp for inspection.
    """
    if not llm_available():
        return None, "OpenAI key not configured or client missing."

    prompt = build_parser_prompt(user_query, known_categories=known_categories)
    messages = [
        {"role": "system", "content": "You are a compact strict parser that outputs a single XML block on the first line."},
        {"role": "user", "content": prompt}
    ]
    try:
        raw = _call_chat(messages, model="gpt-3.5-turbo", max_tokens=240, temperature=0.0)
    except Exception as e:
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        errpath = f"/tmp/spendly_llm_err_{ts}.log"
        try:
            with open(errpath, "w", encoding="utf-8") as fh:
                fh.write("Exception calling LLM:\n")
                fh.write(traceback.format_exc())
                fh.write("\nPrompt:\n" + prompt)
        except Exception:
            pass
        return None, f"LLM call failed: {type(e).__name__}: {e}. See {errpath}"

    # write raw to /tmp for easy debugging
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    rawpath = f"/tmp/spendly_llm_raw_{ts}.log"
    try:
        with open(rawpath, "w", encoding="utf-8") as fh:
            fh.write(raw)
    except Exception:
        pass

    parsed, explain = parse_llm_xml(raw)
    if parsed is None:
        diag = f"LLM returned unparsable response. Raw saved to {rawpath}. Raw (first 800 chars):\n{raw[:800]}"
        return None, diag

    # If category present and known_categories provided, canonicalize
    if parsed.get("category") and known_categories:
        cat = parsed["category"].strip()
        found = None
        for c in known_categories:
            if c and c.lower() == cat.lower():
                found = c
                break
        if not found:
            for c in known_categories:
                if c and (cat.lower() in c.lower() or c.lower() in cat.lower()):
                    found = c
                    break
        if found:
            parsed["category"] = found
            parsed["category_confidence"] = parsed.get("category_confidence") or "high"
        else:
            parsed["category_confidence"] = parsed.get("category_confidence") or "low"

    return parsed, explain or f"Parsed OK. Raw logged to {rawpath}"

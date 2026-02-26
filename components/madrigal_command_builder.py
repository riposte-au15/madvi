import re
from datetime import datetime
from langflow.custom.custom_component.component import Component
from langflow.io import MessageTextInput, Output
from langflow.schema.message import Message
from madrigalWeb.madrigalWeb import MadrigalData

DEFAULT_MADRIGAL_URL = "https://cedar.openmadrigal.org/"

# ------------------------
# Helpers
# ------------------------

def _to_text(x):
    if isinstance(x, Message):
        return x.text or ""
    return (x or "")

def _extract_first_url(s: str) -> str:
    s = (s or "").strip()
    m = re.search(r"https?://[^\s\"']+", s)
    return m.group(0) if m else ""

def _normalize_url(url: str) -> str:
    url = (url or "").strip().strip('"').strip("'")
    extracted = _extract_first_url(url)
    if extracted:
        url = extracted
    if not url:
        url = DEFAULT_MADRIGAL_URL
    url = re.sub(r"(https?://[^/]+)(/.*)?$", r"\1/", url)
    if not url.endswith("/"):
        url += "/"
    return url

def _clean_instrument_name(instr: str) -> str:
    instr = (instr or "").strip()
    instr = re.sub(r"\[[^\]]*\]", "", instr)  # remove [1961-2025]
    instr = re.sub(r"\s+", " ", instr).strip()
    return instr

def _tokenize(s: str):
    s = re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()
    return [t for t in s.split() if t]

def _expand_aliases(tokens):
    aliases = {
        # common shorthand
        "is": ["incoherent", "scatter", "radar"],
        "iscat": ["incoherent", "scatter", "radar"],
        "uhf": ["uhf"],
        "vhf": ["vhf"],
        "hf": ["hf"],
        "radar": ["radar"],

        # common beginner intent words
        "tec": ["total", "electron", "content"],
    }
    expanded = set(tokens)
    for t in list(tokens):
        if t in aliases:
            expanded.update(aliases[t])
    return expanded

def _site_style_normalize(name: str) -> str:
    s = (name or "").lower()
    s = re.sub(r"\[[^\]]*\]", "", s)

    s = s.replace("incoherent scatter radar", "radar")
    s = s.replace("incoherent scatter", "")
    s = s.replace("incoherent", "")
    s = s.replace("scatter", "")
    s = s.replace("is radar", "radar")
    s = re.sub(r"\bis\b", "", s)

    s = re.sub(r"\s+", " ", s).strip()
    return s

def _extract_inst(inst):
    code = getattr(inst, "code", None)
    name = getattr(inst, "name", None)
    if code is not None and name is not None:
        try:
            return int(code), str(name)
        except Exception:
            pass
    try:
        return int(inst[0]), str(inst[1])
    except Exception:
        return None, None

def _strip_ordinal(s: str) -> str:
    # "1st"->"1", "2nd"->"2", "3rd"->"3", "4th"->"4"
    return re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", (s or ""), flags=re.IGNORECASE)

def _parse_date_to_mmddyyyy(s: str) -> str:
    """
    Accept:
      - MM/DD/YYYY
      - YYYY-MM-DD
      - YYYY/MM/DD
      - "January 1st 2025"
      - "Jan 1 2025"
    Return:
      - MM/DD/YYYY
    """
    s = (s or "").strip().strip('"').strip("'")
    if not s:
        return ""

    s = _strip_ordinal(s)
    s = re.sub(r"\s+", " ", s).strip()

    # MM/DD/YYYY
    if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}", s):
        mm, dd, yyyy = s.split("/")
        return f"{int(mm):02d}/{int(dd):02d}/{int(yyyy):04d}"

    # YYYY-MM-DD or YYYY/MM/DD
    m = re.fullmatch(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", s)
    if m:
        yyyy, mm, dd = m.groups()
        return f"{int(mm):02d}/{int(dd):02d}/{int(yyyy):04d}"

    # Month name formats
    for fmt in (
        "%B %d %Y",     # January 1 2025
        "%b %d %Y",     # Jan 1 2025
        "%B %d, %Y",    # January 1, 2025
        "%b %d, %Y",    # Jan 1, 2025
    ):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%m/%d/%Y")
        except Exception:
            pass

    # obvious typo year like 20215
    if re.search(r"\d{5,}", s):
        return ""

    return ""

def _safe_shell_quote(s: str) -> str:
    s = (s or "")
    s = s.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{s}"'

def _normalize_format(fmt_in: str) -> str:
    fmt_in = (fmt_in or "").strip()
    if not fmt_in:
        return ""
    fmt_map = {
        "hdf5": "Hdf5",
        "ascii": "ascii",
        "netcdf4": "netCDF4",
    }
    return fmt_map.get(fmt_in.lower(), fmt_in)

def _normalize_parms_freeform(parms_raw: str) -> str:
    """
    IMPORTANT: We DO NOT guess unknown Madrigal codes.
    We only normalize:
      - comma/space separated codes
      - common aliases like "electron content" -> TEC
    """
    s = (parms_raw or "").strip()
    if not s:
        return ""

    s_low = s.lower()

    # Safe aliases (only when unambiguous)
    if "electron content" in s_low or "total electron content" in s_low or s_low.strip() == "tec":
        return "TEC"

    # If user provides codes like "TEC, GDALT" or "TEC GDALT"
    # normalize to comma-separated uppercase tokens without spaces.
    parts = re.split(r"[,\s]+", s)
    parts = [p.strip().upper() for p in parts if p.strip()]

    # If the “tokens” still contain non-code-looking strings (like "ELECTRON", "DENSITY")
    # we treat it as freeform request and ask user for Madrigal parm code instead of guessing.
    # Heuristic: codes typically contain letters/numbers/underscore and are short.
    if any(len(p) > 12 or not re.fullmatch(r"[A-Z0-9_]+", p) for p in parts):
        return "__NEEDS_EXACT_MADRIGAL_CODES__"

    return ",".join(parts)

# ------------------------
# Component
# ------------------------

class MadrigalCommandBuilder(Component):
    display_name = "Madrigal Command Builder"
    description = "Builds website-style globalIsprint.py commands and resolves instrument names to inst IDs (beginner-proof)."
    icon = "terminal"
    name = "MadrigalCommandBuilder"
    trace_type = "tool"

    inputs = [
        MessageTextInput(name="url", display_name="Madrigal URL", tool_mode=True),
        MessageTextInput(name="instrument", display_name="Instrument (name or id)", tool_mode=True),
        MessageTextInput(name="parms", display_name="Parms (codes or safe aliases like TEC)", tool_mode=True),
        MessageTextInput(name="startDate", display_name="Start Date", tool_mode=True),
        MessageTextInput(name="endDate", display_name="End Date", tool_mode=True),
        MessageTextInput(name="format", display_name="Format (Hdf5|ascii|netCDF4)", tool_mode=True),
        MessageTextInput(name="output_path", display_name="Output path", tool_mode=True),
        MessageTextInput(name="user_fullname", display_name="User full name", tool_mode=True),
        MessageTextInput(name="user_email", display_name="User email", tool_mode=True),
        MessageTextInput(name="user_affiliation", display_name="User affiliation", tool_mode=True),

        # New: lets user say "skip" once and proceed with defaults
        MessageTextInput(name="skip_optional", display_name="Skip optional info? (true/false)", tool_mode=True),

        MessageTextInput(name="debug", display_name="Debug (true/false)", tool_mode=True),
    ]

    outputs = [
        Output(display_name="Command", name="command", method="build_command"),
    ]

    async def build_command(self) -> Message:
        # ---- Inputs ----
        url_raw = _to_text(self.url).strip()
        url = _normalize_url(url_raw)

        instrument_raw = _to_text(self.instrument).strip()
        instrument_clean = _clean_instrument_name(instrument_raw)

        parms_raw = _to_text(self.parms).strip()
        parms_norm = _normalize_parms_freeform(parms_raw)

        start_raw = _to_text(self.startDate).strip()
        end_raw = _to_text(self.endDate).strip()
        startDate = _parse_date_to_mmddyyyy(start_raw)
        endDate = _parse_date_to_mmddyyyy(end_raw)

        fmt_raw = _to_text(self.format).strip()
        fmt = _normalize_format(fmt_raw)

        output_path = (_to_text(self.output_path).strip() or "/tmp")

        user_fullname = (_to_text(self.user_fullname).strip() or "")
        user_email_raw = (_to_text(self.user_email).strip() or "")
        m_email = re.search(r"[\w\.\-+]+@[\w\.\-]+\.\w+", user_email_raw)
        user_email = (m_email.group(0) if m_email else "")
        user_affiliation = (_to_text(self.user_affiliation).strip() or "")

        skip_optional = (_to_text(self.skip_optional).strip().lower() == "true")
        debug = (_to_text(self.debug).strip().lower() == "true")

        # ---- Required questions first (minimal) ----
        if not instrument_clean:
            return Message(text="Which Madrigal instrument should I use? (Example: 'Millstone Hill IS Radar' or inst id like 30)")

        if (start_raw and not startDate) or (end_raw and not endDate):
            return Message(
                text=(
                    "I couldn't parse your date range.\n"
                    "Use either:\n"
                    "  - 2025-01-01 to 2025-01-31\n"
                    "or\n"
                    "  - January 1 2025 to January 31 2025\n"
                )
            )

        if not startDate or not endDate:
            return Message(text="What date range do you want? (Example: 2025-01-01 to 2025-01-31)")

        if not fmt:
            return Message(text="Which output format do you want? Reply with: HDF5, ASCII, or netCDF4.")

        # Parameters:
        if not parms_norm:
            return Message(text="What Madrigal parameter codes do you want? If you mean Total Electron Content, reply: TEC.")
        if parms_norm == "__NEEDS_EXACT_MADRIGAL_CODES__":
            return Message(
                text=(
                    "Madrigal needs short parameter CODES (not freeform words).\n"
                    "If you meant Total Electron Content, reply: TEC.\n"
                    "Otherwise, tell me the exact Madrigal parameter code(s) you want (comma-separated).\n"
                )
            )

        # ---- Pre-flight optional info (grandma friendly) ----
        # Ask ONCE before generating, unless user explicitly skips.
        if not skip_optional:
            missing = []
            if not user_fullname:
                missing.append("full name")
            if not user_email:
                missing.append("email")
            if not user_affiliation:
                missing.append("affiliation")

            if missing:
                return Message(
                    text=(
                        "Before I generate the runnable command, do you want to include any optional info?\n"
                        f"Missing: {', '.join(missing)}\n\n"
                        "You can reply with any of these (or multiple):\n"
                        "  - name: Your Name\n"
                        "  - email: you@domain.com\n"
                        "  - affiliation: Your Org\n"
                        "  - output: /your/path\n\n"
                        "Or reply: skip\n"
                    )
                )

        # If user skipped, fill defaults safely
        if not user_fullname:
            user_fullname = "Unknown User"
        if not user_email:
            user_email = "unknown@example.com"
        if not user_affiliation:
            user_affiliation = "None"

        # ---- Resolve instrument ----
        inst_debug = []
        needle_norm = ""
        try:
            if instrument_clean.isdigit():
                inst_id = int(instrument_clean)
            else:
                md = MadrigalData(url)
                inst_list = md.getAllInstruments()

                needle_ui = instrument_clean.lower()
                needle_norm = _site_style_normalize(instrument_clean)
                tokens = _expand_aliases(set(_tokenize(instrument_clean)))

                # Special: if user explicitly wrote "IS radar" (or "incoherent scatter radar")
                explicit_is_radar = ("is radar" in needle_ui) or ("incoherent scatter radar" in needle_ui) or ("isradar" in needle_ui.replace(" ", ""))

                candidates = []
                for inst in inst_list:
                    _id, name = _extract_inst(inst)
                    if _id is None or not name:
                        continue

                    name_l = name.lower()
                    name_tokens = set(_tokenize(name_l))
                    score = 0

                    # Very strong preference when user explicitly wants IS Radar
                    if explicit_is_radar and ("is radar" in name_l):
                        score += 2000

                    # Strong preference: "is"+"radar" appears in both
                    if ("is" in tokens) and ("radar" in tokens) and ("is" in name_l) and ("radar" in name_l):
                        score += 700

                    # Token overlap
                    overlap = tokens.intersection(name_tokens)
                    if overlap:
                        score += 200 + 35 * len(overlap)

                    # Substring matches
                    if needle_ui and needle_ui in name_l:
                        score += 180
                    if needle_norm and needle_norm in name_l:
                        score += 220

                    if score > 0:
                        candidates.append((score, _id, name))

                candidates.sort(reverse=True, key=lambda x: x[0])

                if not candidates:
                    return Message(
                        text=(
                            f"I couldn't resolve the instrument '{instrument_raw}' at {url}.\n"
                            "Reply with a numeric inst id (e.g., 30) or a more specific name.\n"
                        )
                    )

                top = candidates[0]
                second = candidates[1] if len(candidates) > 1 else None
                inst_id = top[1]
                inst_debug = candidates[:10]

                # Ask only if genuinely close
                if second and (top[0] - second[0]) < 120:
                    msg = [
                        f"Your instrument name is ambiguous: '{instrument_raw}'",
                        f"Site: {url}",
                        "Top candidates (score - id - name):",
                    ]
                    for s, i, n in inst_debug:
                        msg.append(f"  {s} - {i} - {n}")
                    msg.append("Reply with the numeric instrument ID you want (e.g., 30).")
                    return Message(text="\n".join(msg))

        except Exception as e:
            return Message(
                text=(
                    "I hit an error connecting to Madrigal to resolve the instrument.\n"
                    f"Site: {url}\n"
                    f"Error: {type(e).__name__}: {e}\n"
                    "Try again, or reply with a numeric instrument ID.\n"
                )
            )

        # ---- Build command (website style) ----
        parms_part = f" --parms={parms_norm}" if parms_norm else ""

        cmd = (
            "globalIsprint.py --verbose "
            f"--url={url}"
            f"{parms_part} "
            f"--output={_safe_shell_quote(output_path)} "
            f"--user_fullname={_safe_shell_quote(user_fullname)} "
            f"--user_email={_safe_shell_quote(user_email)} "
            f"--user_affiliation={_safe_shell_quote(user_affiliation)} "
            f'--startDate="{startDate}" --endDate="{endDate}" '
            f"--inst={inst_id} --format={fmt}"
        )

        if debug:
            dbg = [
                "DEBUG:",
                f"  url_raw: {url_raw}",
                f"  url_normalized: {url}",
                f"  instrument_raw: {instrument_raw}",
                f"  instrument_clean: {instrument_clean}",
                f"  instrument_normalized: {needle_norm}",
                f"  resolved_inst_id: {inst_id}",
                f"  parms_raw: {parms_raw}",
                f"  parms_normalized: {parms_norm}",
                f"  start_raw: {start_raw}",
                f"  end_raw: {end_raw}",
                f"  startDate_mmddyyyy: {startDate}",
                f"  endDate_mmddyyyy: {endDate}",
                f"  format_raw: {fmt_raw}",
                f"  format_norm: {fmt}",
                f"  skip_optional: {skip_optional}",
                "  top_candidates:",
            ]
            for s, i, n in inst_debug:
                dbg.append(f"    {s} - {i} - {n}")
            cmd = cmd + "\n\n" + "\n".join(dbg)

        self.status = cmd
        return Message(text=cmd)
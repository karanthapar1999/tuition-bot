"""
WhatsApp AI Tuition Bot - Production Ready
==========================================

Features:
- Smart language detection and switching
- Context-aware responses (uses last 3 non-math messages for math queries)
- Multilingual support (English, Hindi, Hinglish, Tamil, Bengali, Telugu, Kannada, Malayalam, Gujarati, Marathi)
- LaTeX rendering with composite images
- Validity gates to prevent hallucinations
- Image processing with OCR for homework questions
- Homework-only filter (blocks non-academic questions)
- Adaptive teaching based on student level

Language Logic:
1. Non-math messages: Reply in the EXACT language of current message
2. Math-only messages: Reply in language of last 3 non-math messages
3. Supports natural mixing (Hinglish)

Author: Gotham AI
Version: 1.6 - Adaptive Teaching
"""

import os
import json
import hashlib
import requests
import base64
import io
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import re

# Analytics
try:
    from mixpanel import Mixpanel
    MIXPANEL_TOKEN = os.getenv('MIXPANEL_TOKEN', '')
    if MIXPANEL_TOKEN:
        mp = Mixpanel(MIXPANEL_TOKEN)
        MIXPANEL_ENABLED = True
        print("✅ Mixpanel analytics enabled")
    else:
        mp = None
        MIXPANEL_ENABLED = False
        print("⚠️  Mixpanel token not found - analytics disabled")
except ImportError:
    mp = None
    MIXPANEL_ENABLED = False
    print("⚠️  Mixpanel not installed - analytics disabled")

# Error tracking
try:
    import sentry_sdk
    SENTRY_DSN = os.getenv('SENTRY_DSN', '')
    if SENTRY_DSN:
        sentry_sdk.init(
            dsn=SENTRY_DSN,
            traces_sample_rate=0.1,  # 10% performance monitoring
            profiles_sample_rate=0.1,  # 10% profiling
            environment="production",
            release="tuition-bot@1.6"
        )
        print("✅ Sentry error tracking enabled")
    else:
        print("⚠️  Sentry DSN not found - error tracking disabled")
except ImportError:
    print("⚠️  Sentry not installed - error tracking disabled")

# Image processing (with graceful fallback)
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image, ImageOps, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️  Warning: Pillow/matplotlib not available. LaTeX rendering disabled.")

# ============================================
# CONFIGURATION
# ============================================
API_KEY = os.getenv("OPENAI_API_KEY", "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")

if not API_KEY:
    raise ValueError("API key not found")

if not os.getenv("OPENAI_API_KEY"):
    print("⚠️  WARNING: Using hardcoded API key for testing. Set OPENAI_API_KEY environment variable for production!")

DEBUG_MODE = True  # Set to False in production

API_URL = "https://api.openai.com/v1/chat/completions"

# Storage: Use persistent disk if available, fallback to local
if Path("/data").exists():
    STORAGE_DIR = Path("/data/bot_storage")
    print("✅ Using persistent disk storage: /data/bot_storage")
else:
    STORAGE_DIR = Path("bot_storage")
    print("⚠️  Using ephemeral storage: bot_storage/ (data will be lost on restart!)")

STORAGE_DIR.mkdir(exist_ok=True)

MAX_HISTORY = 15
MAX_MESSAGE_LENGTH = 1500
MAX_IMAGES_PER_DAY = 20
IMAGE_HISTORY_MARKER = "<image>"

# ============================================
# DATA MODEL
# ============================================
@dataclass
class UserPreferences:
    user_id: str
    name: Optional[str] = None
    language: Optional[str] = None
    language_code: Optional[str] = None
    script: Optional[str] = None
    language_confirmed: bool = False
    message_history: List[Dict] = None
    image_count_today: int = 0
    image_count_reset_date: str = ""
    last_known_language: str = ""

    def __post_init__(self):
        if self.message_history is None:
            self.message_history = []
        if not self.image_count_reset_date:
            self.image_count_reset_date = datetime.now().date().isoformat()

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

# ============================================
# STORAGE HELPERS
# ============================================
def get_user_id(phone: str) -> str:
    """Generate consistent user ID from phone number"""
    return hashlib.sha256(phone.encode()).hexdigest()[:16]

def load_user_preferences(user_id: str) -> UserPreferences:
    """Load user preferences from disk"""
    filepath = STORAGE_DIR / f"{user_id}.json"
    if filepath.exists():
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return UserPreferences.from_dict(data)
        except:
            pass
    return UserPreferences(user_id=user_id)

def save_user_preferences(prefs: UserPreferences):
    """Atomic save of user preferences"""
    filepath = STORAGE_DIR / f"{prefs.user_id}.json"
    temp_filepath = filepath.with_suffix('.tmp')
    with open(temp_filepath, 'w', encoding='utf-8') as f:
        json.dump(prefs.to_dict(), f, indent=2, ensure_ascii=False)
    temp_filepath.replace(filepath)

# ============================================
# ANALYTICS TRACKING
# ============================================
def track_event(event_name: str, user_id: str, properties: dict = None):
    """Track analytics event to Mixpanel"""
    if not MIXPANEL_ENABLED or not mp:
        return
    
    try:
        props = properties or {}
        props['timestamp'] = datetime.now().isoformat()
        mp.track(user_id, event_name, props)
        
        if DEBUG_MODE:
            print(f"📊 Tracked: {event_name} for user {user_id[:8]}...")
    except Exception as e:
        if DEBUG_MODE:
            print(f"Analytics error: {e}")

def update_user_profile(user_id: str, properties: dict):
    """Update user profile in Mixpanel"""
    if not MIXPANEL_ENABLED or not mp:
        return
    
    try:
        mp.people_set(user_id, properties)
        if DEBUG_MODE:
            print(f"📊 Updated profile for user {user_id[:8]}...")
    except Exception as e:
        if DEBUG_MODE:
            print(f"Profile update error: {e}")


# ============================================
# IMAGE PROCESSING
# ============================================
def image_bytes_to_data_uri(img_bytes: bytes) -> str:
    """Convert image bytes to base64 data URI for GPT-4 Vision"""
    if PIL_AVAILABLE:
        try:
            im = Image.open(io.BytesIO(img_bytes))
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=85, optimize=True)
            b = buf.getvalue()
        except Exception as e:
            if DEBUG_MODE:
                print(f"Image processing error: {e}, using original bytes")
            b = img_bytes
    else:
        b = img_bytes
    b64 = base64.b64encode(b).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def check_image_rate_limit(prefs: UserPreferences) -> tuple:
    """Check if user has exceeded daily image limit"""
    today = datetime.now().date().isoformat()
    if prefs.image_count_reset_date != today:
        prefs.image_count_today = 0
        prefs.image_count_reset_date = today
        save_user_preferences(prefs)
    if prefs.image_count_today >= MAX_IMAGES_PER_DAY:
        return False, f"You've reached your daily limit of {MAX_IMAGES_PER_DAY} images. Resets tomorrow!"
    return True, ""

def increment_image_count(prefs: UserPreferences):
    """Increment daily image counter"""
    prefs.image_count_today += 1
    save_user_preferences(prefs)

# ============================================
# LATEX DETECTION & RENDERING
# ============================================
RENDER_TAG_RE = re.compile(r'^\s*<render:(text|image)>\s*$', re.I)

def parse_render_tag_and_body(s: str):
    """Extract render mode and body from GPT response"""
    lines = s.splitlines()
    mode = None
    body_lines = []
    for i, ln in enumerate(lines):
        m = RENDER_TAG_RE.match(ln.strip())
        if m:
            mode = m.group(1).lower()
            body_lines = lines[i+1:]
            break
        if ln.strip():
            body_lines = lines
            break
    body = "\n".join(body_lines).strip()
    body = re.sub(r'</render:(text|image)>', '', body, flags=re.IGNORECASE)
    return mode, body

def contains_latex(s: str) -> bool:
    """Check if string contains LaTeX math"""
    return bool(re.search(r'\$\$[\s\S]*?\$\$', s) or 
                re.search(r'\$[^\$]+?\$', s) or 
                re.search(r'\\\[[\s\S]*?\\\]', s))

def should_render_as_image(gpt_body: str) -> bool:
    """Determine if response should be rendered as image based on LaTeX complexity"""
    if not contains_latex(gpt_body):
        return False

    double_blocks = re.findall(r'\$\$([\s\S]*?)\$\$', gpt_body)
    single_blocks = re.findall(r'\$([^\$]+?)\$', gpt_body)
    bracket_blocks = re.findall(r'\\\[([\s\S]*?)\\\]', gpt_body)
    all_blocks = double_blocks + single_blocks + bracket_blocks

    if len(all_blocks) >= 2:
        return True

    complexity = [r'\frac', r'\int', r'\sum', r'\lim', r'\sqrt', r'\matrix', r'\cases']
    return any(any(k in b for k in complexity) for b in all_blocks)

def latex_to_unicode(latex: str) -> str:
    """Convert simple inline LaTeX to unicode/plain text"""
    latex = re.sub(r'\\text\{([^}]*)\}', r'\1', latex)
    
    latex = latex.replace(r'\,', ' ')
    latex = latex.replace(r'\!', '')
    latex = latex.replace(r'\:', ' ')
    latex = latex.replace(r'\;', ' ')
    latex = latex.replace(r'\quad', ' ')
    latex = latex.replace(r'\qquad', '  ')
    latex = latex.replace(r'\ ', ' ')
    
    def frac_to_slash(match):
        num = match.group(1)
        denom = match.group(2)
        num_clean = num.strip('{}')
        denom_clean = denom.strip('{}')
        return f"({num_clean}/{denom_clean})"

    latex = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', frac_to_slash, latex)
    
    # Use caret notation for superscripts instead of unicode (font compatibility)
    # Keep ^{...} or ^x as is - just remove curly braces
    latex = re.sub(r'\^\{([^}]*)\}', r'^\1', latex)
    # Subscripts: keep _{...} or _x format
    latex = re.sub(r'_\{([^}]*)\}', r'_\1', latex)

    latex = latex.replace(r'\sqrt', '√')
    latex = latex.replace(r'\pm', '±')
    latex = latex.replace(r'\times', '×')
    latex = latex.replace(r'\div', '÷')
    latex = latex.replace(r'\leq', '≤')
    latex = latex.replace(r'\geq', '≥')
    latex = latex.replace(r'\neq', '≠')
    latex = latex.replace(r'\approx', '≈')
    latex = latex.replace(r'\infty', '∞')
    latex = latex.replace(r'\pi', 'π')
    latex = latex.replace(r'\theta', 'θ')
    latex = latex.replace(r'\alpha', 'α')
    latex = latex.replace(r'\beta', 'β')
    latex = latex.replace(r'\gamma', 'γ')
    latex = latex.replace(r'\delta', 'δ')
    latex = latex.replace(r'\sum', '∑')
    latex = latex.replace(r'\int', '∫')
    latex = latex.replace(r'\cdot', '·')
    latex = latex.replace(r'\rightarrow', '→')
    latex = latex.replace(r'\leftarrow', '←')
    latex = latex.replace(r'\Rightarrow', '⇒')
    latex = latex.replace(r'\Leftarrow', '⇐')
    
    latex = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', latex)
    latex = re.sub(r'\\[a-zA-Z]+', '', latex)
    latex = latex.replace('{', '').replace('}', '')
    
    return latex

MATH_RE = re.compile(r'\$\$[\s\S]*?\$\$|\$[^\$]+?\$')

def split_into_segments(text: str):
    """Split text into text and DISPLAY math segments, converting inline to unicode"""
    segments = []
    last_idx = 0

    def replace_inline(match):
        latex = match.group(1)
        return latex_to_unicode(latex)

    stashed_doubles = []
    
    def stash_double(match):
        stashed_doubles.append(match.group(0))
        return f"@@DOUBLE{len(stashed_doubles)-1}@@"
    
    text = re.sub(r'\$\$([\s\S]*?)\$\$', stash_double, text)
    
    text = re.sub(r'\\\((.*?)\\\)', replace_inline, text)
    text = re.sub(r'\$([^\$]+?)\$', replace_inline, text)
    
    def unstash_double(match):
        idx = int(match.group(1))
        return stashed_doubles[idx]
    
    text = re.sub(r'@@DOUBLE(\d+)@@', unstash_double, text)

    for match in re.finditer(r'\$\$([\s\S]*?)\$\$', text):
        if match.start() > last_idx:
            text_chunk = text[last_idx:match.start()].strip()
            if text_chunk:
                segments.append({"kind": "text", "content": text_chunk})
        segments.append({"kind": "latex", "content": match.group(1).strip()})
        last_idx = match.end()

    if last_idx < len(text):
        text_chunk = text[last_idx:].strip()
        if text_chunk:
            segments.append({"kind": "text", "content": text_chunk})

    return segments

def preprocess_latex_for_mathtext(latex: str) -> str:
    """Convert LaTeX to matplotlib mathtext-compatible syntax"""
    latex = re.sub(r'\\text\{([^}]*)\}', r'\1', latex)

    latex = re.sub(r'_([^{])', r'_{\1}', latex)
    latex = re.sub(r'\^([^{])', r'^{\1}', latex)
    latex = latex.replace(r'\,', ' ')
    latex = latex.replace(r'\!', '')
    latex = latex.replace(r'\:', ' ')
    latex = latex.replace(r'\;', ' ')
    latex = latex.replace(r'\quad', ' ')
    latex = latex.replace(r'\qquad', ' ')
    latex = re.sub(r'\\[Bb]igg?g?[lmr]?', '', latex)
    latex = latex.replace(r'\left', '')
    latex = latex.replace(r'\right', '')
    return latex.strip()

UNSUPPORTED = [r'\begin{aligned}', r'\begin{array}', r'\cases', r'\matrix',
               r'\overbrace', r'\underbrace', r'\stackrel']

def has_unsupported_latex(latex: str) -> bool:
    """Check if LaTeX contains constructs mathtext can't handle"""
    return any(construct in latex for construct in UNSUPPORTED)

def wrap_text_lines(draw, text, font, max_width):
    """Wrap text to fit within max_width"""
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        test_line = (current_line + " " + word).strip()
        if draw.textlength(test_line, font=font) <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    return lines if lines else [text]

def render_composite_image(body: str, max_width=800, max_height=3500) -> Optional[bytes]:
    """Render a composite image with text and display math stacked vertically"""
    if not PIL_AVAILABLE:
        return None

    try:
        # Pre-process: Force step separation with blank lines
        body = re.sub(r'\s*(Step \d+:)', r'\n\n\1', body).strip()
        
        segments = split_into_segments(body)
        if not segments:
            return None

        rendered_parts = []
        padding = 20
        line_height = 30
        usable_width = max_width - 2 * padding

        for seg in segments:
            if seg["kind"] == "latex":
                latex_clean = preprocess_latex_for_mathtext(seg["content"])

                if has_unsupported_latex(latex_clean):
                    if DEBUG_MODE:
                        print(f"Skipping unsupported LaTeX: {latex_clean[:50]}...")
                    continue

                fig, ax = plt.subplots(figsize=(0.1, 0.1))
                ax.axis('off')
                ax.text(0.5, 0.5, f"${latex_clean}$", fontsize=24,
                       ha='center', va='center', usetex=False, transform=ax.transAxes)

                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                           pad_inches=0.05, facecolor='white')
                plt.close(fig)

                buf.seek(0)
                img = Image.open(buf).convert("RGB")

                if img.width > usable_width:
                    scale = usable_width / img.width
                    new_size = (int(img.width * scale), int(img.height * scale))
                    img = img.resize(new_size, Image.LANCZOS)

                rendered_parts.append({"type": "image", "content": img})
            else:
                rendered_parts.append({"type": "text", "content": seg["content"].split('\n')})

        try:
            import matplotlib.font_manager as fm
            font_path = fm.findfont(fm.FontProperties(family='DejaVu Sans'))
            font = ImageFont.truetype(font_path, 20)
        except:
            try:
                font = ImageFont.truetype("Arial", 20)
            except:
                font = ImageFont.load_default()

        temp_img = Image.new("RGB", (1, 1))
        temp_draw = ImageDraw.Draw(temp_img)

        wrapped_parts = []
        total_height = padding

        for part in rendered_parts:
            if part["type"] == "image":
                wrapped_parts.append(part)
                total_height += part["content"].height + padding
            else:
                wrapped_lines = []
                for line in part["content"]:
                    wrapped_lines.extend(wrap_text_lines(temp_draw, line, font, usable_width))
                wrapped_parts.append({"type": "text", "content": wrapped_lines})
                total_height += len(wrapped_lines) * line_height + padding

        if total_height > max_height:
            if DEBUG_MODE:
                print(f"Image too tall ({total_height}px > {max_height}px), falling back to text")
            return None

        canvas = Image.new("RGB", (max_width, total_height), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        y_offset = padding

        for part in wrapped_parts:
            if part["type"] == "image":
                img = part["content"]
                x_offset = (max_width - img.width) // 2
                canvas.paste(img, (x_offset, y_offset))
                y_offset += img.height + padding
            else:
                for line in part["content"]:
                    draw.text((padding, y_offset), line, fill=(0, 0, 0), font=font)
                    y_offset += line_height
                y_offset += padding // 2

        out_buf = io.BytesIO()
        canvas.save(out_buf, format="PNG")
        return out_buf.getvalue()

    except Exception as e:
        if DEBUG_MODE:
            print(f"Composite render failed: {e}")
            import traceback
            traceback.print_exc()
        return None

# ============================================
# LANGUAGE DETECTION
# ============================================
def detect_current_message_language(message: str) -> str:
    """
    Detect language of the CURRENT message only (not from history).
    This is critical for proper language switching.
    UPDATED: Now supports Bengali, Telugu, Kannada, Malayalam, Gujarati, Marathi
    """
    # Check for Devanagari script (Hindi/Marathi)
    if any('\u0900' <= c <= '\u097F' for c in message):
        return "Hindi"
    # Check for Tamil script  
    elif any('\u0B80' <= c <= '\u0BFF' for c in message):
        return "Tamil"
    # Check for Bengali script
    elif any('\u0980' <= c <= '\u09FF' for c in message):
        return "Bengali"
    # Check for Telugu script
    elif any('\u0C00' <= c <= '\u0C7F' for c in message):
        return "Telugu"
    # Check for Kannada script
    elif any('\u0C80' <= c <= '\u0CFF' for c in message):
        return "Kannada"
    # Check for Malayalam script
    elif any('\u0D00' <= c <= '\u0D7F' for c in message):
        return "Malayalam"
    # Check for Gujarati script
    elif any('\u0A80' <= c <= '\u0AFF' for c in message):
        return "Gujarati"
    
    # For Latin script, check for Hinglish markers
    hinglish_markers = {
        # Core markers (high confidence)
        'nahi', 'hai', 'kya', 'ka', 'ki', 'ke', 'ko', 'mein', 'samjhao', 
        'batao', 'karo', 'ye', 'kaise', 'matlab', 'thik', 'achha', 'haan', 
        'ji', 'aur', 'mera', 'nam', 'apka', 'tum', 'tumhara', 'main', 'hun',
        # Common additions
        'bhai', 'dost', 'acha', 'sahi', 'theek', 'yaar', 'arre', 'bhagwan',
        'kuch', 'koi', 'kab', 'kahan', 'kyun', 'aise', 'waise', 'he', 'hain',
        'abhi', 'phir', 'fir', 'bas', 'sir', 'madam', 'tha', 'thi',
        'kya', 'kyaa', 'kese', 'jao', 'aao', 'lao',
        'samajh', 'samjha', 'samjho', 'dekh', 'dekho', 'bata', 'bolo',
        # Common misspellings/abbreviations
        'nhi', 'haa', 'mtlb', 'haii', 'kyu', 
        'kyuu', 'kaha', 'kha', 'yr', 'bht', 'smjh', 'smjha',
        # Question words
        'kaun', 'kitna', 'kitne', 'kaunsa', 'kaisa', 'kesi',
        # Verbs/actions (avoid ambiguous words)
        'chalo', 'karo', 'dekho', 'suno', 'batao'
    }
    
    words = message.lower().split()
    # Strip common punctuation from words before checking
    cleaned_words = [word.strip('.,!?;:()[]{}"\'-') for word in words]
    if any(word in hinglish_markers for word in cleaned_words):
        return "Hinglish"
    
    # Default to English for Latin script
    return "English"


def is_neutral_filler(message: str) -> bool:
    """
    Check if message is a neutral filler word/phrase that should use conversation context.
    These are language-ambiguous words that don't indicate a language switch.
    """
    # Strip punctuation and lowercase
    cleaned = message.lower().strip('.,!?;:()[]{}"\'-').strip()
    
    neutral_words = {
        # Acknowledgements
        'ok', 'okay', 'k', 'kk', 'hmm', 'hm', 'uh', 'um', 'ah',
        # Continuations
        'go on', 'continue', 'next', 'and', 'then', 'so',
        # Agreements
        'yes', 'yep', 'yeah', 'yup', 'sure', 'right', 'correct',
        # Negations
        'no', 'nope', 'nah',
        # Single emoji or very short
        '👍', '👎', '✓', '✔', '❌'
    }
    
    return cleaned in neutral_words

def is_math_only(message: str) -> bool:
    """
    Check if message contains only mathematical expressions with no actual words.
    Math symbols, numbers, variables, and operators are allowed.
    """
    # Remove all math symbols, numbers, operators, and common math notation
    cleaned = re.sub(r'[0-9+\-*/=(){}[\]<>≤≥≠±×÷√∫∑∏∂∆∇πθαβγδεζηθλμνξπρστφχψω\^\s.,!?;:]', '', message)
    
    # Remove single-letter variables (x, y, z, a, b, c, etc.)
    cleaned = re.sub(r'\b[a-zA-Z]\b', '', cleaned)
    
    # Remove LaTeX delimiters
    cleaned = cleaned.replace('$$', '').replace('$', '').replace('\\', '')
    
    # If nothing is left, it's math-only
    return len(cleaned.strip()) == 0

def detect_language_from_history(prefs: UserPreferences) -> str:
    """
    Detect target language from last 3 non-math user messages.
    Returns detected language or falls back to user preference.
    """
    recent_non_math = [
        m for m in reversed(prefs.message_history[-10:])
        if m.get('role') == 'user' and not m.get('is_math_only', False)
    ][:3]
    
    if not recent_non_math:
        return prefs.language or "English"
    
    # Detect language of most recent non-math message
    most_recent = recent_non_math[0]['content']
    
    # Skip image markers
    if most_recent.startswith('[Image:'):
        if len(recent_non_math) > 1:
            most_recent = recent_non_math[1]['content']
        else:
            return prefs.language or "English"
    
    return detect_current_message_language(most_recent)

def build_context_with_language_anchors(prefs: UserPreferences) -> Tuple[List[Dict], str]:
    """
    Build conversation context with explicit language markers for context.
    Returns (recent_messages, language_context_block)
    """
    # Get last N messages
    recent = prefs.message_history[-MAX_HISTORY:] if prefs.message_history else []
    
    # Find last 3 non-math user messages for language context
    non_math_messages = [
        m for m in reversed(recent)
        if m.get('role') == 'user' and not m.get('is_math_only', False)
    ][:3]
    
    if not non_math_messages:
        return recent, ""
    
    # Build language context block
    examples = []
    for msg in reversed(non_math_messages):  # Reverse back to chronological
        content = msg['content']
        if not content.startswith('[Image:'):
            lang = detect_current_message_language(content)
            examples.append(f'- Student said: "{content[:60]}..." → Language: {lang}')
    
    if not examples:
        return recent, ""
    
    context = f"""
LANGUAGE CONTEXT FROM RECENT MESSAGES:
{chr(10).join(examples)}

Use this EXACT same language style in your reply.
"""
    
    return recent, context

# ============================================
# HOMEWORK FILTER (NEW)
# ============================================
def is_homework_related(message: str) -> bool:
    """
    Check if a question is homework/academic related.
    Returns True if homework-related, False otherwise.
    """
    # Strip and lowercase
    msg_lower = message.lower().strip()
    
    # Very short messages are likely greetings or acknowledgments - allow them
    if len(msg_lower.split()) <= 2:
        return True
    
    # Check for obvious non-homework keywords
    non_homework_keywords = [
        'girlfriend', 'boyfriend', 'dating', 'relationship', 'love', 'crush',
        'meaning of life', 'purpose of life', 'why are we here', 'existence',
        'depression', 'suicide', 'kill myself',
        'how to make money', 'get rich', 'bitcoin', 'stock market',
        'recipe', 'cooking', 'food', 'restaurant',
        'movie', 'game', 'sports', 'music', 'song',
        'joke', 'funny', 'meme',
        'weather', 'news', 'politics',
        'health', 'doctor', 'medicine', 'symptom',
        'job', 'career', 'interview', 'resume',
        'social media', 'instagram', 'facebook', 'tiktok',
        'fashion', 'clothing', 'outfit'
    ]
    
    # Check if message contains non-homework keywords
    for keyword in non_homework_keywords:
        if keyword in msg_lower:
            return False
    
    # Check for academic keywords (allow these)
    academic_keywords = [
        'solve', 'calculate', 'find', 'prove', 'derive', 'explain',
        'equation', 'formula', 'theorem', 'proof',
        'homework', 'assignment', 'problem', 'question', 'exercise',
        'math', 'science', 'physics', 'chemistry', 'biology',
        'history', 'geography', 'english', 'grammar',
        'integrate', 'differentiate', 'simplify', 'factor',
        'angle', 'triangle', 'circle', 'square', 'area', 'volume',
        'velocity', 'acceleration', 'force', 'energy',
        'atom', 'molecule', 'reaction', 'element',
    ]
    
    # If contains academic keywords, it's homework
    for keyword in academic_keywords:
        if keyword in msg_lower:
            return True
    
    # Check if it contains math expressions or numbers with operators
    has_math = bool(re.search(r'\d+\s*[+\-*/=^]\s*\d+', msg_lower))
    if has_math:
        return True
    
    # Check for question patterns that suggest academic queries
    question_patterns = [
        r'what is\s+\w+',
        r'how (to|do|does|did)',
        r'why (is|are|does|did)',
        r'when (is|are|does|did)',
        r'where (is|are|does|did)',
        r'can you (explain|show|tell|help)',
        r'please (explain|show|tell|help)',
    ]
    
    for pattern in question_patterns:
        if re.search(pattern, msg_lower):
            # This could be academic - default to allowing it
            return True
    
    # Default: allow the message (give benefit of doubt)
    return True

# ============================================
# GPT HELPER
# ============================================
def call_gpt_for_text(system_prompt: str, user_prompt: str, 
                      temp: float = 0, max_tok: int = 100, 
                      default: str = "") -> str:
    """Call GPT for simple text generation"""
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temp,
        "max_tokens": max_tok
    }
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    
    try:
        response = requests.post(API_URL, headers=headers, json=data, timeout=10)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"GPT Error: {e}")
        return default

STYLE_DELIMITERS = {
    'bold_start': '\u200B*\u200B', 'bold_end': '\u200B*\u200B',
    'italic_start': '\u200B_\u200B', 'italic_end': '\u200B_\u200B',
    'code_start': '\u200B`\u200B', 'code_end': '\u200B`\u200B'
}

def protect_math_then_style(body: str) -> str:
    """Apply style guard while protecting LaTeX math from being corrupted"""
    stash = []

    def _stash(m):
        stash.append(m.group(0))
        return f'@@MATH{len(stash)-1}@@'

    masked = MATH_RE.sub(_stash, body)
    masked = apply_style_guard(masked)

    def _unstash(m):
        return stash[int(m.group(1))]

    return re.sub(r'@@MATH(\d+)@@', _unstash, masked)

def apply_style_guard(text: str) -> str:
    """Remove MCQ-style formatting"""
    text = re.sub(r'(?im)^\s*options:\s*', '', text)
    text = re.sub(r'(?im)^\s*(\([a-z]\)|[a-z]\)|\d+[.)]|[ivxlcdm]+[.)]|[•\-–—])\s*', '', text)
    text = re.sub(r'\(\s*[a-z]\s*\)\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+,', ', ', text)
    text = re.sub(r'[ \t]{2,}', ' ', text).strip()
    return text

# ============================================
# NAME DETECTION HELPERS
# ============================================
LATIN_GREETS = {"hi", "hello", "hey", "namaste", "vanakkam", "hola", "bonjour", "ciao", "hallo", "ok", "okay", "yes", "no", "haan", "nahi"}
DEVANAGARI_GREETS = {"नमस्ते", "नमस्कार", "प्रणाम", "हाँ", "नहीं"}
TAMIL_GREETS = {"வணக்கம்", "வணகம்", "ஆம்", "இல்லை"}

def looks_like_name(token: str, script_hint: Optional[str] = None) -> bool:
    """Check if a token looks like a person's name"""
    t = token.strip()
    if not t.replace("'", "").replace("-", "").isalpha():
        return False
    
    if script_hint == "Devanagari" and t in DEVANAGARI_GREETS:
        return False
    if script_hint == "Tamil" and t in TAMIL_GREETS:
        return False
    
    if t.lower() in LATIN_GREETS:
        return False
    
    if len(t) < 2 or len(t) > 20:
        return False
    
    return True

# ============================================
# JSON PARSER
# ============================================
def parse_json_response(raw: str, retry_on_fail: bool = True) -> Dict:
    """Parse JSON from GPT response with fallback handling"""
    raw = raw.strip()
    
    for start_char in ['```json', '```']:
        if raw.startswith(start_char):
            raw = raw[len(start_char):].strip()
            break
    
    if raw.endswith('```'):
        raw = raw[:-3].strip()
    
    for char in ['{', '[']:
        idx = raw.find(char)
        if idx > 0:
            raw = raw[idx:]
            break
    
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        try:
            pass
        except:
            pass

    coerced = raw.replace("True", "true").replace("False", "false")
    try:
        return json.loads(coerced)
    except Exception:
        if retry_on_fail:
            print(f"JSON parse error. Raw: {raw[:200]}...")
        return {}

def strip_quotes(s: str) -> str:
    """Remove surrounding quotes from string"""
    s = s.strip()
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        return s[1:-1].strip()
    return s

# ============================================
# ONBOARDING ANALYZER
# ============================================
def analyze_first_message(message: str) -> Dict:
    """Analyze first message to detect language, script, and name"""
    if len(message) > MAX_MESSAGE_LENGTH:
        message = message[:MAX_MESSAGE_LENGTH] + "..."

    system_prompt = """You are a deterministic analyzer. You must ONLY return a single valid JSON object. Do not include any extra text. If uncertain, set fields to null and "confidence":"low". Never invent names."""

    user_prompt = f"""Analyze this first message from a student:
"{message}"

Task:
1) Detect the SCRIPT first (by actual characters used, not meaning)
2) Then detect the language
3) Extract a first name if present
4) Produce a greeting in the SAME SCRIPT

STEP 1 - SCRIPT DETECTION (look at actual characters):
- Check each character in the message
- If ALL letters are a-z, A-Z (Latin alphabet) → Script: "Latin"
- If contains Hindi/Devanagari characters (हिंदी) → Script: "Devanagari"  
- If contains Tamil characters (தமிழ்) → Script: "Tamil"
- If contains Bengali characters (বাংলা) → Script: "Bengali"
- If contains Telugu characters (తెలుగు) → Script: "Telugu"
- If contains Kannada characters (ಕನ್ನಡ) → Script: "Kannada"
- If contains Malayalam characters (മലയാളം) → Script: "Malayalam"
- If contains Gujarati characters (ગુજરાતી) → Script: "Gujarati"

STEP 2 - LANGUAGE DETECTION (after knowing script):
- If Script is "Latin":
  * Words like: mera, naam, hai, kya, main, hun, he, ka, ki, ke → Language: "Hinglish"
  * Words like: my, is, what, I, you, am, the, this → Language: "English"
- If Script is "Devanagari":
  * Language: "Hindi" (or "Marathi" if context suggests)
- If Script is "Tamil":
  * Language: "Tamil"
- If Script is "Bengali":
  * Language: "Bengali"
- If Script is "Telugu":
  * Language: "Telugu"
- If Script is "Kannada":
  * Language: "Kannada"
- If Script is "Malayalam":
  * Language: "Malayalam"
- If Script is "Gujarati":
  * Language: "Gujarati"

CRITICAL: "mera nam karan" uses LATIN alphabet (a-z), so script MUST be "Latin" and language MUST be "Hinglish"

Confidence rules:
- Single short greeting ("hi", "hello") → "low"
- Clear Hinglish markers (mera, naam, hai, kya, etc.) → "high"
- Devanagari/Tamil/Bengali/Telugu/Kannada/Malayalam/Gujarati script with words → "high"
- Clear English sentences → "high"  
- Otherwise → "medium"

Return ONLY JSON:
{{
  "detected_language": "<English|Hindi|Hinglish|Tamil|Bengali|Telugu|Kannada|Malayalam|Gujarati|Marathi|... or null>",
  "language_code": "<en|hi|hi_latin|ta|bn|te|kn|ml|gu|mr|... or null>",
  "script": "<Latin|Devanagari|Tamil|Bengali|Telugu|Kannada|Malayalam|Gujarati|... or null>",
  "name": "<FirstName or null>",
  "confidence": "<high|medium|low>",
  "greeting": "<same-script greeting asking their name, or null>"
}}"""

    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0,
        "max_tokens": 150
    }

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    try:
        response = requests.post(API_URL, headers=headers, json=data, timeout=10)
        response.raise_for_status()
        result = response.json()
        content = result['choices'][0]['message']['content']
        return parse_json_response(content)
    except Exception as e:
        print(f"Error analyzing message: {e}")
        return {"confidence": "low"}

# ============================================
# GREETING GENERATORS
# ============================================
def generate_greeting(language: str, script: str = "Latin") -> str:
    """Generate appropriate greeting in user's language"""
    if language == "Hinglish" or (language == "Hindi" and script == "Latin"):
        return "Namaste! Main aapka AI Tuition Teacher hun. Aapka naam kya hai?"
    elif language == "Hindi" and script == "Devanagari":
        return "नमस्ते! मैं आपका AI ट्यूशन टीचर हूं। आपका नाम क्या है?"
    elif language == "Tamil":
        return "வணக்கம்! நான் உங்கள் AI பயிற்சி ஆசிரியர். உங்கள் பெயர் என்ன?"
    elif language == "English":
        return "Hello! I'm your AI Tuition Teacher. What's your name?"

    system_prompt = "Return ONLY the translated sentence. No quotes, no commentary."
    user_prompt = f"""Translate exactly this sentence into {language} (same tone):
"Hello! I'm your AI Tuition Teacher. What's your name?" """

    return call_gpt_for_text(system_prompt, user_prompt,
                          default="Hello! I'm your AI Tuition Teacher. What's your name?")

def generate_welcome(name: str, language: str, script: str = "Latin") -> str:
    """Generate welcome message in user's language"""
    if language == "Hinglish" or (language == "Hindi" and script == "Latin"):
        return f"Welcome {name}! Main aapke homework mein madad ke liye ready hun. Koi bhi sawal puchiye - text mein type karo ya photo bhejo! 📸"
    elif language == "Hindi" and script == "Devanagari":
        return f"स्वागत है {name}! मैं आपके होमवर्क में मदद के लिए तैयार हूं। कोई भी सवाल पूछें - टेक्स्ट में लिखें या फोटो भेजें! 📸"
    elif language == "Tamil":
        return f"வரவேற்கிறேன் {name}! நான் உங்கள் வீட்டுப்பாடத்தில் உதவ தயாராக இருக்கிறேன். எந்த கேள்வியும் கேளுங்கள் - எழுதலாம் அல்லது புகைப்படம் அனுப்பலாம்! 📸"
    elif language == "English":
        return f"Welcome {name}! I'm ready to help with your homework. Ask me any question - you can type it or send a photo! 📸"

    system_prompt = "Return ONLY the translated sentence. No quotes, no commentary."
    user_prompt = f"""Translate this sentence into {language}, preserving the name {name}:
"Welcome {name}! I'm ready to help with your homework. Ask me any question - you can type it or send a photo!" """

    return call_gpt_for_text(system_prompt, user_prompt,
                          default=f"Welcome {name}! I'm ready to help with your homework. Ask me any question - you can type it or send a photo! 📸")

# ============================================
# ONBOARDING FLOW
# ============================================
def handle_onboarding(user_message: str, prefs: UserPreferences) -> Dict:
    """Handle onboarding flow to capture name and language"""
    if not prefs.name and not prefs.language:
        analysis = analyze_first_message(user_message)
        
        # Track first message
        track_event('first_message_sent', prefs.user_id, {
            'detected_language': analysis.get('detected_language'),
            'confidence': analysis.get('confidence'),
            'has_name': bool(analysis.get('name'))
        })
        
        if DEBUG_MODE:
            print(f"DEBUG ONBOARDING: Analysis result: {analysis}")

        detected_lang = analysis.get("detected_language")
        detected_script = analysis.get("script")
        name = analysis.get("name")
        custom_greeting = analysis.get("greeting")
        confidence = analysis.get("confidence")

        if confidence in ("high", "medium") and detected_lang:
            prefs.language = detected_lang
            prefs.language_code = analysis.get("language_code", "en")
            prefs.script = detected_script or "Latin"
            
            if DEBUG_MODE:
                print(f"DEBUG ONBOARDING: Set language to {prefs.language}, script to {prefs.script}")

            if name:
                prefs.name = name
                prefs.language_confirmed = True
                
                # Track name provided
                track_event('name_provided', prefs.user_id, {
                    'name': name,
                    'language': prefs.language,
                    'script': prefs.script
                })
                
                # Update user profile
                update_user_profile(prefs.user_id, {
                    '$name': name,
                    'language': prefs.language,
                    'script': prefs.script,
                    'language_code': prefs.language_code
                })
                
                prefs.message_history.append({"role": "user", "content": user_message})
                welcome = generate_welcome(name, prefs.language, prefs.script)
                prefs.message_history.append({"role": "assistant", "content": welcome})
                save_user_preferences(prefs)
                return {"text": welcome}
            elif custom_greeting:
                prefs.message_history.append({"role": "user", "content": user_message})
                prefs.message_history.append({"role": "assistant", "content": custom_greeting})
                save_user_preferences(prefs)
                return {"text": custom_greeting}
            else:
                prefs.message_history.append({"role": "user", "content": user_message})
                greeting = generate_greeting(prefs.language, prefs.script)
                prefs.message_history.append({"role": "assistant", "content": greeting})
                save_user_preferences(prefs)
                return {"text": greeting}
        else:
            prefs.language = "unknown"
            trilingual = """Hello! I'm your AI Tuition Teacher. What's your name?

नमस्ते! मैं आपका AI ट्यूशन टीचर हूं। आपका नाम क्या है?

வணக்கம்! நான் உங்கள் AI பயிற்சி ஆசிரியர். உங்கள் பெயர் என்ன?"""
            
            prefs.message_history.append({"role": "user", "content": user_message})
            prefs.message_history.append({"role": "assistant", "content": trilingual})
            save_user_preferences(prefs)

            return {"text": trilingual}

    if not prefs.name:
        if len(user_message.split()) > 1 and prefs.language == "unknown":
            analysis = analyze_first_message(user_message)
            
            if DEBUG_MODE:
                print(f"DEBUG ONBOARDING (name extraction): Analysis result: {analysis}")
            
            name = analysis.get("name")

            if analysis.get("confidence") in ("high", "medium") and analysis.get("detected_language"):
                prefs.language = analysis.get("detected_language")
                prefs.language_code = analysis.get("language_code", "en")
                prefs.script = analysis.get("script", "Latin")
                prefs.language_confirmed = True
                
                if DEBUG_MODE:
                    print(f"DEBUG ONBOARDING: Set language to {prefs.language}, script to {prefs.script}")

            if name:
                prefs.name = name
                if prefs.language == "unknown":
                    prefs.language = "English"
                    prefs.language_code = "en"
                    prefs.script = "Latin"
                    prefs.language_confirmed = True
                    
                    if DEBUG_MODE:
                        print(f"DEBUG ONBOARDING: Language still unknown, defaulting to English")
                prefs.message_history.append({"role": "user", "content": user_message})
                welcome = generate_welcome(prefs.name, prefs.language, prefs.script)
                
                if DEBUG_MODE:
                    print(f"DEBUG ONBOARDING: Generating welcome with language={prefs.language}, script={prefs.script}")
                
                prefs.message_history.append({"role": "assistant", "content": welcome})
                save_user_preferences(prefs)
                return {"text": welcome}

        words = user_message.strip().split()
        if len(words) == 1 and looks_like_name(words[0], prefs.script or "Latin"):
            if (prefs.script or "Latin") == "Latin":
                prefs.name = words[0].capitalize()
            else:
                prefs.name = words[0]

            if prefs.language == "unknown":
                prefs.language = "English"
                prefs.language_code = "en"
                prefs.script = "Latin"
                prefs.language_confirmed = True
                
                if DEBUG_MODE:
                    print(f"DEBUG ONBOARDING: Single word name, defaulting to English")

            prefs.message_history.append({"role": "user", "content": user_message})
            welcome = generate_welcome(prefs.name, prefs.language, prefs.script)
            
            if DEBUG_MODE:
                print(f"DEBUG ONBOARDING: Generating welcome (single name) with language={prefs.language}, script={prefs.script}")
            
            prefs.message_history.append({"role": "assistant", "content": welcome})
            save_user_preferences(prefs)
            return {"text": welcome}

        if prefs.language == "unknown":
            trilingual = """Hello! I'm your AI Tuition Teacher. What's your name?

नमस्ते! मैं आपका AI ट्यूशन टीचर हूं। आपका नाम क्या है?

வணக்கம்! நான் உங்கள் AI பயிற்சி ஆசிரியர். உங்கள் பெயர் என்ன?"""
            
            prefs.message_history.append({"role": "user", "content": user_message})
            prefs.message_history.append({"role": "assistant", "content": trilingual})
            save_user_preferences(prefs)

            return {"text": trilingual}
        else:
            greeting = generate_greeting(prefs.language, prefs.script)
            prefs.message_history.append({"role": "user", "content": user_message})
            prefs.message_history.append({"role": "assistant", "content": greeting})
            save_user_preferences(prefs)
            return {"text": greeting}

    return {"text": "Please send your homework question!"}

# ============================================
# CORE TUTORING WITH SMART LANGUAGE SWITCHING
# ============================================
def call_gpt_tutor(user_message: str, prefs: UserPreferences) -> Dict:
    """
    Core tutoring function with context-aware language switching.
    
    Logic:
    - If user message has words: Reply in language of current message
    - If user message is pure math: Reply in language of last 3 non-math messages
    """
    # HOMEWORK FILTER CHECK
    if not is_homework_related(user_message):
        # Get user's language
        target_lang = detect_language_from_history(prefs)
        
        # Generate response in user's language
        if target_lang == "Hinglish":
            return {"text": "Main sirf homework aur padhai mein help kar sakta hun. Koi homework question hai?"}
        elif target_lang == "Hindi":
            return {"text": "मैं केवल होमवर्क और पढ़ाई में मदद कर सकता हूं। कोई होमवर्क का सवाल है?"}
        elif target_lang == "Tamil":
            return {"text": "நான் ஹோம்வொர்க் மற்றும் படிப்பில் மட்டுமே உதவ முடியும். ஏதாவது ஹோம்வொர்க் கேள்வி உள்ளதா?"}
        else:
            return {"text": "I can only help with homework and academic questions. Do you have a homework question?"}
    
    # Truncate if too long
    if len(user_message) > MAX_MESSAGE_LENGTH:
        user_message = user_message[:MAX_MESSAGE_LENGTH] + "..."
    
    current_is_math = is_math_only(user_message)
    
    # Track message sent
    track_event('message_sent', prefs.user_id, {
        'is_math_only': current_is_math,
        'language': prefs.language,
        'message_length': len(user_message),
        'has_name': bool(prefs.name)
    })
    
    # Detect language of CURRENT message if it has words
    current_message_language = None
    if not current_is_math:
        current_message_language = detect_current_message_language(user_message)
    
    if DEBUG_MODE:
        print(f"DEBUG: Processing message: '{user_message[:50]}...'")
        print(f"DEBUG: Is math-only: {current_is_math}")
        if current_message_language:
            print(f"DEBUG: Current message language: {current_message_language}")

    # Build context with language anchors
    recent_messages, language_context = build_context_with_language_anchors(prefs)
    
    # CRITICAL: Only inject context if current message is math-only
    context_block = language_context if (current_is_math and language_context) else ""
    
    # Detect target language from history
    target_language = detect_language_from_history(prefs)
    save_user_preferences(prefs)

    if DEBUG_MODE:
        print(f"DEBUG: Target language: {target_language}")
        print(f"DEBUG: Context included: {bool(context_block)}")

    # Add explicit current language override if message has words
    current_lang_override = ""
    if current_message_language:
        current_lang_override = f"""
═══════════════════════════════════════════════════════════════
🚨 ABSOLUTE CRITICAL OVERRIDE - READ THIS FIRST 🚨
═══════════════════════════════════════════════════════════════
The CURRENT message from the student is in: {current_message_language.upper()}

YOU MUST RESPOND 100% IN {current_message_language.upper()} ONLY.
- Use {current_message_language} for ALL explanatory text
- Use {current_message_language} for ALL step descriptions  
- Use {current_message_language} for ALL words between LaTeX formulas
- IGNORE all other language examples below
- IGNORE the language from previous conversation history
- THIS OVERRIDES EVERYTHING ELSE IN THIS PROMPT

WRONG: Using any language other than {current_message_language}
RIGHT: Using ONLY {current_message_language} for all text
═══════════════════════════════════════════════════════════════
"""
    
    system_prompt = f"""{current_lang_override}

You are **Gotham Sir**, a patient but rigorous WhatsApp tuition teacher for {prefs.name or 'the student'}.

ADAPTIVE TEACHING - CRITICAL:

**Your PRIMARY job is to TEACH, not just solve. Adapt your approach based on the student's level.**

STEP 1 - INFER STUDENT LEVEL from the question complexity:
- Elementary (ages 6-11): Basic arithmetic (2+2, 5×3), simple fractions (1/2), counting, basic shapes
- Middle School (ages 11-14): Algebra basics (2x+3=7), percentages, area/perimeter, simple equations
- High School (ages 14-18): Advanced algebra, quadratic equations, trigonometry, calculus basics, physics
- College/Advanced (18+): Advanced calculus, differential equations, linear algebra, quantum physics, proofs

STEP 2 - ADJUST YOUR TEACHING STYLE based on inferred level:

For ELEMENTARY students:
- Use simple, everyday language with concrete examples
- "Let's think of this like sharing candies among friends..."
- Break into tiny steps with encouragement
- Avoid technical terms

For MIDDLE SCHOOL students:
- Introduce proper terminology gently
- "You know that 2+3=5. Now let's see why 2x+3x works the same way..."
- Connect to real scenarios (money, sports, cooking)
- Show the "why" behind rules

For HIGH SCHOOL students:
- Use proper mathematical terminology
- Explain concepts AND intuition together
- "This integral represents the area under the curve. Here's why that matters..."
- Show connections between topics

For COLLEGE/ADVANCED students:
- Assume mathematical maturity
- Focus on deep understanding and implications
- "This is a classic application of the Fundamental Theorem. Notice how..."
- Reference theorems and advanced techniques

STEP 3 - TEACHING DEPTH (adjust based on complexity):

SIMPLE questions (basic arithmetic, simple factoring):
- 2-3 brief steps with explanations
- "This is straightforward: we just need to..."

MODERATE questions (solve equations, basic integration):
- 4-5 detailed steps
- "Here's what we're doing and why at each stage..."

COMPLEX questions (derivations, multi-step proofs, physics problems):
- 6+ comprehensive steps
- Explain the STRATEGY first before solving
- "Let's think about our approach: we'll use X because..."
- Connect to broader mathematical concepts
- "This technique is powerful because it helps us..."

**REMEMBER: Don't just show steps - TEACH the underlying concepts. Help students understand WHY, not just HOW.**

LANGUAGE RULE - FOLLOW EXACTLY:
1. If current message HAS WORDS → Reply in the SAME language as those words. Ignore everything else.
2. If current message is PURE MATH (no words) → Look at context below and reply in THAT EXACT language/style.

{context_block}

EXAMPLE (when current is math-only):
- Student's recent messages show: "mera naam karan", "main samajh gaya"
- Current: "4x-7y"  
- You MUST reply in Hinglish mixing pattern: "Yeh ek expression hai. Isse solve karne ke liye..."
- NEVER switch to pure English when context shows Hindi/Hinglish!

RENDER DECISION - Understanding WhatsApp's Technical Limitations:

WhatsApp is a PLAIN TEXT messenger with these hard technical constraints:
1. NO vertical fraction display - (x³/3) appears inline, not stacked
2. NO proper superscript/subscript stacking - ∫₀¹ renders as ∫01 with misaligned numbers
3. Unicode limits DON'T stack above/below symbols - they appear inline and look broken
4. Complex expressions become parenthesis soup - (((x-3)/(x+2))^2) is unreadable
5. Font inconsistencies - superscript ¹²³ may render in different font than ⁴⁵⁶⁷⁸⁹⁰
6. NO LaTeX, NO MathML, NO HTML formatting support

**CRITICAL: Do NOT avoid proper mathematical notation just to stay in text mode!**
**If a problem is clearer with proper formulas, USE proper LaTeX notation AND render as image.**

**Decision based on PROBLEM TYPE (not symbols you happen to use):**

ALWAYS use `<render:image>` for these problem types:
- Integration problems (use proper integral notation with limits, not workarounds)
- Limit problems (use proper limit notation, not "lim (x → 0)")
- Derivative problems (use proper notation)
- Problems with fractions in solution (use proper fractions, not (a/b))
- Summation/product problems (use proper sigma/pi notation with bounds)
- Multi-step derivations (2+ equations)
- Physics problems with complex notation
- ANY problem where proper mathematical notation would be significantly clearer

Use `<render:text>` ONLY for genuinely simple cases:
- Basic arithmetic: 2+2, 15×3
- Simple linear equations: 2x+5=11, x=3
- Single simple expressions: x², 3x-7

**Rule of thumb: If you're tempted to write √ instead of proper root notation, 
or (a/b) instead of proper fractions, or "lim" instead of proper limits, 
YOU SHOULD BE USING IMAGE MODE.**

**When uncertain → ALWAYS choose `<render:image>`**
**First line MUST be exactly: `<render:text>` or `<render:image>`**

• **CRITICAL FORMATTING:**
  - For `<render:image>`: Use ONLY $$...$$ for display math (double dollars)
  - For `<render:text>`: Write math naturally without any $ delimiters (will be auto-converted)
  - **ABSOLUTELY FORBIDDEN: NEVER use \\[ \\] or $...$ (single dollar) - these break rendering**

• **CRITICAL - LaTeX RESTRICTIONS:**
  ✗ NEVER use \\text{{}} - write labels in plain text outside $$
  ✗ NEVER use \\begin{{aligned}}, \\begin{{array}}, \\cases, \\matrix
  ✗ NEVER use \\overbrace, \\underbrace, \\stackrel
  ✗ Example: Write "Area: $$3 \\times 17$$" NOT "$$\\text{{Area}} = 3 \\times 17$$"

VALIDITY GATE - CRITICAL:
• **NEVER HALLUCINATE MISSING INFORMATION. If unclear, ASK.**
• Check if message provides ALL necessary information:
  ✓ Expression only (no "=") → Cannot solve. Can only simplify.
  ✓ Unclear/ambiguous → Need clarification.

• **LOGIC GATE - Validate before answering:**
  ✓ Check if terms used correctly (radius for circles/spheres, not squares)
  ✓ Check if question is physically/mathematically possible
  ✓ If illogical/contradictory, STOP and ask for clarification

• **CRITICAL EXAMPLES:**
  - User: "4x-7y" → Bot: "This is an expression. To solve for x or y, I'd need it to equal something. Do you have an equation?"
  - User: "radius of a square" → Bot: "Squares have side lengths, not radii. Did you mean: 1) A sphere (3D) or 2) A circle (2D)?"

TUTORING PEDAGOGY - CRITICAL:
• **NEVER give direct final answers immediately**
• **ALWAYS adapt complexity to student level** (infer from question difficulty)
• **TEACH concepts, not just procedures**
• **ALWAYS break problems into teaching steps with explanations**
• **NUMBER YOUR STEPS: Step 1, Step 2, Step 3, etc.**
• **PUT A BLANK LINE BETWEEN EACH STEP FOR READABILITY**

Your explanation structure:
• Step 1: Understand - "What are we trying to find?"
• Step 2: Plan - "Here's the concept/method we'll use and why..."
• Step 3-N: Execute - Apply step-by-step with clear explanations
• Final Step: Verify - "Let's check if this makes sense..."
• Closing: Learn - "The key idea here is..." or "Remember this for next time..."

Between steps, explain WHY we're doing each action.
• Write in flowing natural language - NO section headers, NO bold markdown (**text**), NO bullets in your response
• Explain step-by-step using connecting words like "pehle" (first), "phir" (then), "ab" (now), "isliye" (therefore)
• **IMPORTANT: Adjust explanation depth based on question complexity**

For simple questions: Be concise
For complex questions: Be thorough - explain the reasoning, strategy, and intuition

STYLE: No lists, friendly full sentences."""

    messages = [{"role": "system", "content": system_prompt}]

    # Add recent conversation history (without the is_math_only tags)
    messages.extend([
        {"role": msg['role'], "content": msg['content']}
        for msg in recent_messages
    ])

    messages.append({"role": "user", "content": user_message})

    data = {
        "model": "gpt-4o-mini",
        "messages": messages,
        "temperature": 0.35,
        "max_tokens": 700
    }

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    try:
        # Increased timeout from 20s to 40s and added retry logic
        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = requests.post(API_URL, headers=headers, json=data, timeout=40)
                response.raise_for_status()
                result = response.json()
                gpt_response = result['choices'][0]['message']['content']
                break  # Success, exit retry loop
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    if DEBUG_MODE:
                        print(f"Timeout on attempt {attempt + 1}, retrying...")
                    import time
                    time.sleep(2)  # Wait 2 seconds before retry
                    continue
                else:
                    raise  # Final attempt failed, raise the timeout error

        mode, body = parse_render_tag_and_body(gpt_response)
        
        # Convert \[...\] to $$...$$ automatically
        body = re.sub(r'\\\[([\s\S]*?)\\\]', r'$$\1$$', body)
        
        body = protect_math_then_style(body)

        # Check for complex math FIRST
        if should_render_as_image(body):
            mode = "image"
        elif mode not in ("text", "image"):
            mode = "text"

        if mode == "image" and not contains_latex(body):
            mode = "text"

        if DEBUG_MODE:
            print(f"DEBUG: Render mode: {mode}")

        # Convert inline LaTeX to unicode
        body = re.sub(r'\\\((.*?)\\\)', lambda m: latex_to_unicode(m.group(1)), body)

        # Strip ALL LaTeX delimiters in text mode
        if mode == "text":
            def replace_math(match):
                return latex_to_unicode(match.group(1))

            body = re.sub(r'\$\$(.*?)\$\$', replace_math, body)
            body = re.sub(r'\$([^\$]+?)\$', replace_math, body)

        if "new question" in body.lower() and "starting" in body.lower():
            prefs.message_history = []

        # Save with tags
        prefs.message_history.append({
            "role": "user", 
            "content": user_message[:MAX_MESSAGE_LENGTH],
            "is_math_only": current_is_math
        })
        prefs.message_history.append({
            "role": "assistant", 
            "content": body
        })

        if len(prefs.message_history) > MAX_HISTORY:
            prefs.message_history = prefs.message_history[-MAX_HISTORY:]

        save_user_preferences(prefs)

        result_dict = {"text": body}

        # Only render image if mode is "image" AND there's still LaTeX
        if mode == "image" and PIL_AVAILABLE and contains_latex(body):
            img_bytes = render_composite_image(body)

            if img_bytes:
                # Always use simple caption - steps are in the image
                clean_text = "See the step-by-step solution in the image above."

                result_dict = {
                    "text": clean_text,
                    "image_bytes": img_bytes,
                    "send_as": "image_with_caption"
                }

        return result_dict

    except requests.exceptions.Timeout:
        print(f"GPT Tutor Timeout Error after {max_retries} attempts")
        
        # Track timeout error
        track_event('error_timeout', prefs.user_id, {
            'error_type': 'openai_timeout',
            'max_retries': max_retries,
            'language': prefs.language
        })
        
        # Return language-appropriate timeout message
        target_lang = detect_language_from_history(prefs)
        if target_lang == "Hinglish":
            return {"text": "OpenAI thoda slow hai abhi. Please thoda wait karke phir se try karo!"}
        elif target_lang == "Hindi":
            return {"text": "OpenAI अभी थोड़ा धीमा है। कृपया थोड़ी देर प्रतीक्षा करें और फिर से प्रयास करें!"}
        elif target_lang == "Tamil":
            return {"text": "OpenAI இப்போது சற்று மெதுவாக உள்ளது. தயவுசெய்து சற்று காத்திருந்து மீண்டும் முயற்சிக்கவும்!"}
        else:
            return {"text": "OpenAI is running a bit slow right now. Please wait a moment and try again!"}
    except Exception as e:
        print(f"GPT Tutor Error: {e}")
        
        # Track general error
        track_event('error_gpt_tutor', prefs.user_id, {
            'error_type': type(e).__name__,
            'error_message': str(e)[:200],  # First 200 chars
            'language': prefs.language
        })
        
        # Return language-appropriate error message
        target_lang = detect_language_from_history(prefs)
        if target_lang == "Hinglish":
            return {"text": "Kuch technical problem aa gayi. Please thodi der baad try karo!"}
        elif target_lang == "Hindi":
            return {"text": "कुछ तकनीकी समस्या आ गई है। कृपया थोड़ी देर बाद प्रयास करें!"}
        elif target_lang == "Tamil":
            return {"text": "சில தொழில்நுட்ப சிக்கல் ஏற்பட்டுள்ளது. தயவுசெய்து சிறிது நேரம் கழித்து முயற்சிக்கவும்!"}
        else:
            return {"text": "I'm having a technical issue. Please try again in a moment!"}

# ============================================
# IMAGE TUTORING WITH SMART LANGUAGE SWITCHING
# ============================================
def call_gpt_tutor_image(image_data_uri: str, caption_text: str, prefs: UserPreferences) -> Dict:
    """
    Image tutoring function with context-aware language switching.
    
    Logic:
    - If caption has words: Reply in language of caption
    - If no caption: Reply in language of last 3 non-math messages
    """
    # Track image sent
    track_event('image_sent', prefs.user_id, {
        'has_caption': bool(caption_text and len(caption_text.strip()) > 0),
        'caption_length': len(caption_text) if caption_text else 0,
        'language': prefs.language
    })
    
    # Build context with language anchors
    _, language_context = build_context_with_language_anchors(prefs)
    
    # CRITICAL: Only inject context if caption is empty/ambiguous OR is a neutral filler
    has_words_in_caption = bool(caption_text and len(caption_text.strip()) > 0)
    is_filler_caption = is_neutral_filler(caption_text) if caption_text else False
    context_block = "" if (has_words_in_caption and not is_filler_caption) else language_context
    
    # Detect caption language if it has words AND is not a neutral filler
    caption_language = None
    if has_words_in_caption and not is_filler_caption:
        caption_language = detect_current_message_language(caption_text)
    
    # Detect target language
    target_language = detect_language_from_history(prefs)

    if DEBUG_MODE:
        print(f"DEBUG: Processing image with caption: '{caption_text}'")
        print(f"DEBUG: Caption language: {caption_language}")
        print(f"DEBUG: Target language: {target_language}")
        print(f"DEBUG: Is math-only: False")
        print(f"DEBUG: Is neutral filler: {is_filler_caption}")
        print(f"DEBUG: Context included: {bool(context_block)}")
    
    # Build language override
    caption_lang_override = ""
    if caption_language:
        caption_lang_override = f"""
🚨🚨🚨 MANDATORY INSTRUCTION - HIGHEST PRIORITY 🚨🚨🚨
RESPOND ONLY IN: {caption_language.upper()}
🚨🚨🚨 END MANDATORY INSTRUCTION 🚨🚨🚨
"""
    
    system_prompt = f"""{caption_lang_override}

You are **Gotham Sir**, a patient but rigorous WhatsApp tuition teacher for {prefs.name or 'the student'}.

ADAPTIVE TEACHING - CRITICAL:

**Your PRIMARY job is to TEACH, not just solve. Adapt your approach based on the student's level.**

STEP 1 - INFER STUDENT LEVEL from the question complexity:
- Elementary (ages 6-11): Basic arithmetic (2+2, 5×3), simple fractions (1/2), counting, basic shapes
- Middle School (ages 11-14): Algebra basics (2x+3=7), percentages, area/perimeter, simple equations
- High School (ages 14-18): Advanced algebra, quadratic equations, trigonometry, calculus basics, physics
- College/Advanced (18+): Advanced calculus, differential equations, linear algebra, quantum physics, proofs

STEP 2 - ADJUST YOUR TEACHING STYLE based on inferred level:

For ELEMENTARY students:
- Use simple, everyday language with concrete examples
- "Let's think of this like sharing candies among friends..."
- Break into tiny steps with encouragement
- Avoid technical terms

For MIDDLE SCHOOL students:
- Introduce proper terminology gently
- "You know that 2+3=5. Now let's see why 2x+3x works the same way..."
- Connect to real scenarios (money, sports, cooking)
- Show the "why" behind rules

For HIGH SCHOOL students:
- Use proper mathematical terminology
- Explain concepts AND intuition together
- "This integral represents the area under the curve. Here's why that matters..."
- Show connections between topics

For COLLEGE/ADVANCED students:
- Assume mathematical maturity
- Focus on deep understanding and implications
- "This is a classic application of the Fundamental Theorem. Notice how..."
- Reference theorems and advanced techniques

STEP 3 - TEACHING DEPTH (adjust based on complexity):

SIMPLE questions (basic arithmetic, simple factoring):
- 2-3 brief steps with explanations
- "This is straightforward: we just need to..."

MODERATE questions (solve equations, basic integration):
- 4-5 detailed steps
- "Here's what we're doing and why at each stage..."

COMPLEX questions (derivations, multi-step proofs, physics problems):
- 6+ comprehensive steps
- Explain the STRATEGY first before solving
- "Let's think about our approach: we'll use X because..."
- Connect to broader mathematical concepts
- "This technique is powerful because it helps us..."

**REMEMBER: Don't just show steps - TEACH the underlying concepts. Help students understand WHY, not just HOW.**

LANGUAGE RULE - FOLLOW EXACTLY:
1. If caption HAS WORDS (not just "ok"/"yes"/"this") → Reply in the SAME language as the caption.
2. If NO caption OR caption is neutral filler ("ok", "yes", "this") → Look at context below and reply in THAT EXACT language/style.

{context_block}

RENDER DECISION - WhatsApp's Technical Limitations:

WhatsApp plain text has hard constraints:
- NO vertical fractions or stacked notation
- Unicode limits appear inline, not above/below symbols
- Complex expressions = unreadable parenthesis nests
- NO LaTeX/MathML/HTML support

**CRITICAL: Use proper mathematical notation! Don't avoid LaTeX to stay in text mode.**

**Base decision on PROBLEM TYPE:**

ALWAYS use `<render:image>` for:
- Integration (use proper integral notation)
- Limits (use proper limit notation)
- Fractions (use proper fraction notation, not (a/b))
- Summations/products with bounds
- Multi-step derivations
- ANY problem where LaTeX notation is clearer

Use `<render:text>` ONLY for:
- Basic arithmetic, simple equations
- Single simple expressions

**If proper notation (roots, fractions, limits, integrals) would be clearer, use it AND choose image mode.**
**When uncertain → use `<render:image>`**
**First line MUST be: `<render:text>` or `<render:image>`**

• **CRITICAL FORMATTING:**
  - For `<render:image>`: Use ONLY $$...$$ (double dollars)
  - For `<render:text>`: No $ delimiters
  - **FORBIDDEN: NEVER use \\[ \\] or $...$ (single dollar)**

• **CRITICAL - LaTeX RESTRICTIONS:**
  ✗ NEVER use \\text{{}}, \\begin{{aligned}}, \\cases, \\matrix, \\overbrace

IMAGE GATE - CRITICAL:
• **NEVER HALLUCINATE. If unclear, ASK.**
• Check if image + caption provide ALL info:
  ✓ Expression only (no "=") → Cannot solve
  ✓ Blurry → Need clearer photo
  ✓ Shape without dimensions → Need measurements
  ✓ Multiple problems → Which one?

• **LOGIC GATE:**
  ✓ Validate terms used correctly
  ✓ Check if physically/mathematically possible
  ✓ If illogical, ask for clarification

• **Examples:**
  - Photo: "4x-4" alone → "I see 4x-4. Want me to simplify, or is there an equation?"
  - Blurry → "Too blurry. Retake with better lighting on flat surface?"
  - Triangle without dimensions → "I see a triangle! What are the side lengths?"
  - Caption: "radius of a square" → "Squares have side lengths. Did you mean sphere or circle?"

TUTORING PEDAGOGY - CRITICAL:
• **NEVER give direct final answers immediately**
• **ALWAYS adapt complexity to student level** (infer from question difficulty)
• **TEACH concepts, not just procedures**
• **ALWAYS break problems into teaching steps with explanations**
• **NUMBER YOUR STEPS: Step 1, Step 2, Step 3, etc.**
• **PUT A BLANK LINE BETWEEN EACH STEP FOR READABILITY**

Your explanation structure:
• Step 1: Understand - "What are we trying to find?"
• Step 2: Plan - "Here's the concept/method we'll use and why..."
• Step 3-N: Execute - Apply step-by-step with clear explanations
• Final Step: Verify - "Let's check if this makes sense..."
• Closing: Learn - "The key idea here is..." or "Remember this for next time..."

Between steps, explain WHY we're doing each action.
• Write in flowing natural language - NO section headers, NO bold markdown (**text**), NO bullets in your response
• Explain step-by-step using connecting words like "pehle" (first), "phir" (then), "ab" (now), "isliye" (therefore)
• **IMPORTANT: Adjust explanation depth based on question complexity**

For simple questions: Be concise
For complex questions: Be thorough - explain the reasoning, strategy, and intuition

STYLE: No lists, friendly tone"""

    messages = [{"role": "system", "content": system_prompt}]

    if prefs.message_history:
        messages.extend(prefs.message_history[-MAX_HISTORY:])

    user_content = []

    if caption_text:
        user_content.append({"type": "text", "text": caption_text})
    else:
        user_content.append({"type": "text", "text": "Use this photo as the question or show me your working."})

    user_content.append({"type": "image_url", "image_url": {"url": image_data_uri, "detail": "high"}})

    messages.append({"role": "user", "content": user_content})

    data = {
        "model": "gpt-4o",
        "messages": messages,
        "temperature": 0.35,
        "max_tokens": 700
    }

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    try:
        # Increased timeout to 50s for image processing (GPT-4o with images is slower)
        # Added retry logic
        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = requests.post(API_URL, headers=headers, json=data, timeout=50)
                response.raise_for_status()
                j = response.json()
                gpt_response = (j.get("choices") or [{}])[0].get("message", {}).get("content")
                break  # Success, exit retry loop
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    if DEBUG_MODE:
                        print(f"Image timeout on attempt {attempt + 1}, retrying...")
                    import time
                    time.sleep(2)  # Wait 2 seconds before retry
                    continue
                else:
                    raise  # Final attempt failed, raise the timeout error

        if not gpt_response:
            return {"text": "I couldn't read that image. Please resend a clearer photo with good lighting on a flat surface."}

        mode, body = parse_render_tag_and_body(gpt_response)
        
        body = re.sub(r'\\\[([\s\S]*?)\\\]', r'$$\1$$', body)
        body = protect_math_then_style(body)

        if should_render_as_image(body):
            mode = "image"
        elif mode not in ("text", "image"):
            mode = "text"

        if mode == "image" and not contains_latex(body):
            mode = "text"

        body = re.sub(r'\\\((.*?)\\\)', lambda m: latex_to_unicode(m.group(1)), body)

        if mode == "text":
            def replace_math(match):
                return latex_to_unicode(match.group(1))

            body = re.sub(r'\$\$(.*?)\$\$', replace_math, body)
            body = re.sub(r'\$([^\$]+?)\$', replace_math, body)

        if "new question" in body.lower() and "starting" in body.lower():
            prefs.message_history = []

        history_text = caption_text if caption_text else IMAGE_HISTORY_MARKER
        prefs.message_history.append({"role": "user", "content": f"[Image: {history_text}]"})
        prefs.message_history.append({"role": "assistant", "content": body})

        if len(prefs.message_history) > MAX_HISTORY:
            prefs.message_history = prefs.message_history[-MAX_HISTORY:]

        save_user_preferences(prefs)

        result_dict = {"text": body}

        if mode == "image" and PIL_AVAILABLE and contains_latex(body):
            img_bytes = render_composite_image(body)

            if img_bytes:
                # FIXED: Always use simple message when rendering as image
                clean_text = "See the step-by-step solution in the image above."

                result_dict = {
                    "text": clean_text,
                    "image_bytes": img_bytes,
                    "send_as": "image_with_caption"
                }

        return result_dict

    except requests.exceptions.Timeout:
        print(f"GPT Image Tutor Timeout Error after {max_retries} attempts")
        
        # Track image timeout error
        track_event('error_image_timeout', prefs.user_id, {
            'error_type': 'image_processing_timeout',
            'has_caption': bool(caption_text),
            'language': prefs.language
        })
        
        # Return language-appropriate timeout message
        target_lang = detect_language_from_history(prefs)
        if target_lang == "Hinglish":
            return {"text": "Photo process hone mein time lag raha hai. Ek chota ya clear photo try karo!"}
        elif target_lang == "Hindi":
            return {"text": "फोटो प्रोसेस होने में समय लग रहा है। एक छोटी या स्पष्ट फोटो भेजें!"}
        elif target_lang == "Tamil":
            return {"text": "புகைப்படம் செயலாக்க நேரம் அதிகமாகிறது. ஒரு சிறிய அல்லது தெளிவான புகைப்படத்தை முயற்சிக்கவும்!"}
        else:
            return {"text": "The image is taking too long to process. Try a smaller or clearer photo!"}
    except Exception as e:
        print(f"GPT Image Tutor Error: {e}")
        
        # Track image processing error
        track_event('error_image_processing', prefs.user_id, {
            'error_type': type(e).__name__,
            'error_message': str(e)[:200],
            'has_caption': bool(caption_text),
            'language': prefs.language
        })
        
        # Return language-appropriate error message
        target_lang = detect_language_from_history(prefs)
        if target_lang == "Hinglish":
            return {"text": "Photo read nahi ho payi. Better lighting aur focus ke saath phir se try karo!"}
        elif target_lang == "Hindi":
            return {"text": "फोटो पढ़ नहीं पाया। बेहतर प्रकाश और फोकस के साथ फिर से प्रयास करें!"}
        elif target_lang == "Tamil":
            return {"text": "புகைப்படத்தை படிக்க முடியவில்லை. சிறந்த வெளிச்சம் மற்றும் கவனத்துடன் மீண்டும் முயற்சிக்கவும்!"}
        else:
            return {"text": "I'm having trouble reading that photo. Try again with better lighting and focus!"}

# ============================================
# MAIN BOT CLASS
# ============================================
class TuitionBot:
    """Main bot class handling user interactions"""
    
    def __init__(self, phone_number: str):
        self.user_id = get_user_id(phone_number)
        self.prefs = load_user_preferences(self.user_id)

    def process_message(self, user_message: str) -> Dict:
        """Process text message from user"""
        user_message = user_message.strip()
        if not user_message:
            return {"text": "Please send a message!"}

        if not self.prefs.name or not self.prefs.language_confirmed:
            return handle_onboarding(user_message, self.prefs)

        response = call_gpt_tutor(user_message, self.prefs)
        return response

    def process_image(self, image_bytes: bytes, caption_text: str = "") -> Dict:
        """Process image message from user"""
        if not image_bytes:
            return {"text": "I didn't receive the photo. Please try again."}

        allowed, limit_msg = check_image_rate_limit(self.prefs)
        if not allowed:
            return {"text": limit_msg}

        if not self.prefs.name:
            self.prefs.name = "Student"
            self.prefs.language = "English"
            self.prefs.language_code = "en"
            self.prefs.language_confirmed = True
            save_user_preferences(self.prefs)

        try:
            data_uri = image_bytes_to_data_uri(image_bytes)
        except Exception as e:
            print(f"Image conversion error: {e}")
            return {"text": "I couldn't process that image format. Try sending a JPG or PNG."}

        increment_image_count(self.prefs)

        reply = call_gpt_tutor_image(data_uri, caption_text, self.prefs)
        return reply

    def clear_history(self):
        """Clear conversation history"""
        self.prefs.message_history = []
        save_user_preferences(self.prefs)
        return f"History cleared. Student: {self.prefs.name}, Language: {self.prefs.language}"

# ============================================
# BOT MANAGER
# ============================================
class BotManager:
    """Manages multiple bot instances (one per user)"""
    
    def __init__(self):
        self.bots = {}

    def get_bot(self, phone: str) -> TuitionBot:
        """Get or create bot instance for phone number"""
        if phone not in self.bots:
            self.bots[phone] = TuitionBot(phone)
        return self.bots[phone]

    def process_message(self, phone: str, message: str) -> Dict:
        """Process text message"""
        bot = self.get_bot(phone)
        return bot.process_message(message)

    def process_image(self, phone: str, image_bytes: bytes, caption_text: str = "") -> Dict:
        """Process image message"""
        bot = self.get_bot(phone)
        return bot.process_image(image_bytes, caption_text)

# Global bot manager instance
bot_manager = BotManager()

# ============================================
# PUBLIC API
# ============================================
def process_message(phone: str, message: str) -> Dict:
    """
    Public API: Process text message
    
    Args:
        phone: User's phone number (e.g., "+919876543210")
        message: Text message from user
        
    Returns:
        Dict with 'text' key, optionally 'image_bytes' and 'send_as' keys
    """
    return bot_manager.process_message(phone, message)

def process_image(phone: str, image_bytes: bytes, caption_text: str = "") -> Dict:
    """
    Public API: Process image message
    
    Args:
        phone: User's phone number
        image_bytes: Image data as bytes
        caption_text: Optional caption text
        
    Returns:
        Dict with 'text' key, optionally 'image_bytes' and 'send_as' keys
    """
    return bot_manager.process_image(phone, image_bytes, caption_text)

# ============================================
# TESTING
# ============================================
def interactive_chat():
    """Interactive chat for testing"""
    import random

    phone = f"+91{random.randint(7000000000, 9999999999)}"

    print("\n" + "="*60)
    print("🎓 WhatsApp Tuition Bot - Production Ready v1.6")
    print("="*60)
    print("\n✅ FEATURES:")
    print("• Smart language switching (detects current message)")
    print("• Context-aware (uses last 3 non-math for math queries)")
    print("• Multilingual (English, Hindi, Hinglish, Tamil, Bengali, Telugu, Kannada, Malayalam, Gujarati, Marathi)")
    print("• LaTeX rendering with composite images")
    print("• Validity gates (no hallucinations)")
    print("• Homework-only filter (blocks non-academic questions)")
    print("• Adaptive teaching based on student level")
    print("\n📝 LANGUAGE LOGIC:")
    print("• Non-math messages → Reply in CURRENT message language")
    print("• Math-only messages → Reply in language of last 3 non-math")
    print("\n🎯 ADAPTIVE TEACHING:")
    print("• Infers student level from question complexity")
    print("• Adjusts language and depth accordingly")
    print("• Elementary → Simple language, concrete examples")
    print("• College → Advanced terminology, deep concepts")
    print("\n⌨️  COMMANDS: 'quit', 'reset', 'clear', 'info', 'image'")
    print("="*60)

    while True:
        msg = input("\n💬 You: ").strip()

        if not msg:
            continue

        if msg.lower() == 'quit':
            print("Goodbye!")
            break

        if msg.lower() == 'reset':
            bot_manager.bots[phone] = TuitionBot(phone)
            print("✅ Complete reset")
            continue

        if msg.lower() == 'clear':
            bot = bot_manager.get_bot(phone)
            result = bot.clear_history()
            print(f"✅ {result}")
            continue

        if msg.lower() == 'info':
            bot = bot_manager.get_bot(phone)
            print(f"\n📊 Current State:")
            print(f"  Name: {bot.prefs.name or 'Not set'}")
            print(f"  Language: {bot.prefs.language or 'Not set'}")
            print(f"  Script: {bot.prefs.script or 'Not set'}")
            print(f"  History: {len(bot.prefs.message_history)} messages")
            print(f"  Images today: {bot.prefs.image_count_today}/{MAX_IMAGES_PER_DAY}")
            
            non_math = [m for m in bot.prefs.message_history 
                       if m.get('role') == 'user' and not m.get('is_math_only', False)]
            print(f"  Non-math messages: {len(non_math)}")
            continue

        if msg.lower() == 'image':
            image_path = input("Image file path: ").strip()
            caption = input("Caption (optional): ").strip()

            try:
                with open(image_path, 'rb') as f:
                    img_bytes = f.read()
                response = process_image(phone, img_bytes, caption)
                print(f"\n🤖 Bot: {response['text']}")

                if 'image_bytes' in response:
                    out_path = "math_solution.png"
                    with open(out_path, "wb") as f:
                        f.write(response['image_bytes'])
                    print(f"📐 Solution image saved to {out_path}")
            except FileNotFoundError:
                print(f"❌ File not found: {image_path}")
            except Exception as e:
                print(f"❌ Error: {e}")
            continue

        response = process_message(phone, msg)
        print(f"\n🤖 Bot: {response['text']}")

        if 'image_bytes' in response:
            out_path = "math_solution.png"
            with open(out_path, "wb") as f:
                f.write(response['image_bytes'])
            print(f"📐 Solution image saved to {out_path}")

if __name__ == "__main__":
    interactive_chat()

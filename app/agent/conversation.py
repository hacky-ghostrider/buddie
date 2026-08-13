"""Lightweight conversational / domain routing for Buddie.

Deterministic intent classification runs *before* RAG or employee tools so
greetings, goodbyes, acknowledgements, and nonsense never trigger retrieval.

This module classifies by *category* (greeting, goodbye, thanks, ack, …),
not by an exhaustive list of exact phrases. Casual prefixes must not hide
real employee / policy intent.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class IntentRoute(str, Enum):
    """High-level route selected before tool planning."""

    CONVERSATION = "conversation"
    EMPLOYEE = "employee"
    KNOWLEDGE = "knowledge"
    HYBRID = "hybrid"
    UNKNOWN = "unknown"
    UNSUPPORTED = "unsupported"
    EMPTY = "empty"
    VERIFY_ID = "verify_id"
    PASS_THROUGH = "pass_through"


class ConversationKind(str, Enum):
    """Subtype for the conversation route."""

    GREETING = "greeting"
    THANKS = "thanks"
    GOODBYE = "goodbye"
    IDENTITY = "identity"
    CAPABILITIES = "capabilities"
    CASUAL = "casual"
    ACK = "ack"


VERIFY_PROMPT = (
    "Before I access your employee information, I need to verify your "
    "employee ID.\n\n"
    "Please enter your employee ID, for example E-1101."
)

VERIFY_FAILED = (
    "That employee ID couldn't be verified.\n\n"
    "Employee IDs should follow the format E-1101.\n"
    "Please recheck and try again."
)

EMPLOYEE_ID_FORMAT_PROMPT = (
    "Please enter your employee ID in the format E-1101."
)

UNKNOWN_FALLBACK = (
    "I'm not sure what you're asking. I can help with leave, holidays, "
    "benefits, employee information, and company policies."
)

# Keep a single user-facing fallback for misunderstood / out-of-domain input.
UNSUPPORTED_FALLBACK = UNKNOWN_FALLBACK

EMPTY_FALLBACK = "How can I help you today?"

GENERATION_FAILURE = (
    "I'm having trouble generating a response right now. Please try again."
)

TOOL_FAILURE = (
    "I'm having trouble completing that right now. Please try again."
)

RAG_FAILURE = (
    "I'm having trouble accessing company knowledge right now. Please try again."
)

_EMPLOYEE_ID_RE = re.compile(r"^E-\d{4}$", re.IGNORECASE)
_EMPLOYEE_ID_FIND_RE = re.compile(r"\bE-\d{4}\b", re.IGNORECASE)
_DIGITS_ONLY_RE = re.compile(r"^\d+$")
_PUNCT_ONLY_RE = re.compile(r"^[\W_]+$", re.UNICODE)

# Short phrases that submit an employee id (not business questions mentioning one).
_ID_SUBMISSION_HINTS = (
    "employee id",
    "employee_id",
    "emp id",
    "my id",
    "my employee",
    "id is",
    "id:",
    "id =",
)

# ---------------------------------------------------------------------------
# Lightweight routing normalization (never used as the downstream question)
# ---------------------------------------------------------------------------

# Very common short aliases / typos — keep tiny; do not grow into a dictionary.
_TOKEN_ALIASES: dict[str, str] = {
    "helo": "hello",
    "helloo": "hello",
    "hii": "hi",
    "hiii": "hi",
    "heyya": "hey",
    "hiya": "hi",
    "thx": "thanks",
    "thanx": "thanks",
    "ty": "thanks",
    "byee": "bye",
    "byebye": "bye",
    "cya": "see ya",
    "ttyl": "talk later",
    "gud": "good",
    "gm": "good morning",
    "whats": "what is",
    "meny": "many",
    "ok": "okay",
    "k": "okay",
    "yep": "yes",
    "yup": "yes",
    "nope": "no",
}

_FILLER_TOKENS = frozenset(
    {
        "cool",
        "okay",
        "ok",
        "alright",
        "allright",
        "sure",
        "yeah",
        "yep",
        "yup",
        "right",
        "well",
        "so",
        "just",
        "please",
        "buddie",
        "buddy",
        "bot",
        "assistant",
    }
)

_GREETING_TOKENS = frozenset(
    {
        "hi",
        "hello",
        "hey",
        "hiya",
        "howdy",
        "morning",
        "afternoon",
        "evening",
        "yo",
        "sup",
    }
)

_GOODBYE_PHRASES = (
    "talk to you later",
    "catch you later",
    "see you later",
    "see you soon",
    "have a good day",
    "have a nice day",
    "have a great day",
    "i'm leaving",
    "im leaving",
    "gotta go",
    "got to go",
    "good bye",
    "goodbye",
    "see ya",
    "see you",
    "bye bye",
    "bye",
    "later",
    "ttyl",
    "cya",
)

_THANKS_PHRASES = (
    "thank you",
    "thanks a lot",
    "thanks so much",
    "much appreciated",
    "appreciate it",
    "appreciate that",
    "thanks",
    "thank",
    "thx",
    "thanx",
    "ty",
)

_ACK_TOKENS = frozenset(
    {
        "cool",
        "okay",
        "ok",
        "alright",
        "sure",
        "yes",
        "no",
        "yep",
        "yup",
        "nope",
        "hmm",
        "hm",
        "huh",
        "got it",
        "understood",
        "makes sense",
        "sounds good",
        "perfect",
        "great",
        "nice",
        "awesome",
        "roger",
        "noted",
    }
)

_IDENTITY_HINTS = (
    "who are you",
    "what are you",
    "what is buddie",
    "what's buddie",
    "who is buddie",
    "your name",
)

_CAPABILITY_HINTS = (
    "what can you do",
    "how can you help",
    "what can i ask",
    "what do you do",
    "how do you help",
    "what are you able to",
)

_CASUAL_PHRASES = (
    "how are you doing",
    "how are you",
    "are you there",
    "you there",
    "can you help me",
    "can you help",
)

# Signals that the message still contains a real employee / knowledge ask.
_BUSINESS_HINTS = (
    "leave",
    "vacation",
    "pto",
    "holiday",
    "holidays",
    "benefit",
    "benefits",
    "policy",
    "policies",
    "handbook",
    "payroll",
    "salary",
    "paycheck",
    "attendance",
    "absent",
    "manager",
    "department",
    "profile",
    "employee",
    "hr ",
    " hr",
    "remote",
    "work from home",
    "parental",
    "maternity",
    "paternity",
    "expense",
    "reimbursement",
    "onboarding",
    "probation",
    "notice period",
    "dress code",
    "code of conduct",
    "security",
    "carry forward",
    "carry-forward",
    "rollover",
    "sick",
    "balance",
    "history",
    "pending",
    "timesheet",
    "designation",
    "how many",
    "how much",
    "what is",
    "what's",
    "whats",
    "show me",
    "show my",
    "tell me",
    "can i",
    "do i have",
    "days do i",
    "upcoming",
)

_KNOWLEDGE_HINTS = (
    "policy",
    "policies",
    "handbook",
    "benefits",
    "benefit",
    "work remotely",
    "remote work",
    "work from home",
    "parental leave",
    "maternity",
    "paternity",
    "security requirement",
    "security policy",
    "code of conduct",
    "expense policy",
    "reimbursement",
    "dress code",
    "notice period",
    "onboarding",
    "probation",
    "leave policy",
    "vacation policy",
    "sick leave policy",
    "pto policy",
    "company provide",
    "does the company",
    "employee handbook",
    "hr policy",
    "can employees",
)

_UNSUPPORTED_HINTS = (
    "weather",
    "capital of",
    "mars",
    "recipe",
    "stock price",
    "bitcoin",
    "who won the",
    "sports score",
    "movie",
    "joke",
    "write a poem",
    "translate ",
)


@dataclass(frozen=True)
class IntentDecision:
    """Result of lightweight intent classification."""

    route: IntentRoute
    kind: ConversationKind | None = None
    response: str | None = None
    employee_id: str | None = None


def normalize_message(text: str) -> str:
    """Collapse whitespace for display / light matching."""
    cleaned = (text or "").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def normalize_for_routing(text: str) -> str:
    """Normalize a message for intent matching only.

    Downstream planning still receives the original user text. This path
    strips edge punctuation, lowercases, applies tiny alias rewrites, and
    collapses elongated letters (``hiii`` → ``hi``) without building a
    giant typo dictionary.
    """
    cleaned = normalize_message(text).lower()
    if not cleaned:
        return ""

    # Drop emoji / most symbols but keep word characters and spaces.
    cleaned = re.sub(r"[^\w\s'-]", " ", cleaned, flags=re.UNICODE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return ""

    # Collapse 3+ repeated letters → 1 (byee → bye, hiii → hi).
    cleaned = re.sub(r"(.)\1{2,}", r"\1", cleaned)
    # Collapse remaining doubles on short tokens later via aliases / token map.

    tokens = cleaned.split()
    rewritten: list[str] = []
    for token in tokens:
        token = token.strip("'-_")
        if not token:
            continue
        # Soft collapse of doubled letters for short tokens (heello → helo → hello).
        collapsed = re.sub(r"(.)\1+", r"\1", token) if len(token) <= 8 else token
        mapped = _TOKEN_ALIASES.get(token) or _TOKEN_ALIASES.get(collapsed) or token
        if " " in mapped:
            rewritten.extend(mapped.split())
        else:
            rewritten.append(mapped)

    # "gud morning" style already mapped token-wise.
    return " ".join(rewritten).strip()


def looks_like_employee_id(text: str) -> bool:
    """True only for explicit ``E-####`` ids — bare digits are not enough."""
    cleaned = normalize_message(text).upper().replace(" ", "")
    return bool(_EMPLOYEE_ID_RE.match(cleaned))


def extract_employee_id(text: str) -> str | None:
    """Return a normalized ``E-####`` when the message is an ID submission.

    Accepts bare ids (``E-1101``) and short submission phrases such as
    ``employee ID is E-1101`` / ``my employee id is E-1101``.

    Bare digits (``123``) and business questions that merely mention an id
    never match — those stay on the conversational / planner path.
    """
    cleaned = normalize_message(text)
    if not cleaned:
        return None

    compact = cleaned.upper().replace(" ", "")
    if _EMPLOYEE_ID_RE.match(compact):
        return compact

    match = _EMPLOYEE_ID_FIND_RE.search(cleaned)
    if not match:
        return None

    eid = match.group(0).upper()
    lowered = cleaned.lower()

    # Explicit "employee id is E-####" style submissions.
    if any(hint in lowered for hint in _ID_SUBMISSION_HINTS):
        # Keep real HR questions on the planner path even if they cite an id.
        if has_business_intent(cleaned):
            remainder = _EMPLOYEE_ID_FIND_RE.sub(" ", cleaned)
            remainder = re.sub(r"[^\w\s]", " ", remainder, flags=re.UNICODE)
            remainder_tokens = [
                tok
                for tok in remainder.lower().split()
                if tok
                and tok
                not in {
                    "employee",
                    "id",
                    "emp",
                    "my",
                    "is",
                    "the",
                    "here",
                    "please",
                    "it",
                    "a",
                }
            ]
            business_markers = (
                "leave",
                "vacation",
                "balance",
                "history",
                "policy",
                "holiday",
                "benefit",
                "pending",
                "how",
                "show",
                "what",
                "can",
                "days",
            )
            if any(
                any(marker in tok for marker in business_markers)
                for tok in remainder_tokens
            ):
                return None
        return eid

    # Tiny filler around a bare id: "here E-1101 please"
    remainder = _EMPLOYEE_ID_FIND_RE.sub(" ", cleaned)
    remainder = re.sub(r"[^\w\s]", " ", remainder, flags=re.UNICODE)
    tokens = [tok for tok in remainder.lower().split() if tok]
    filler = {
        "my",
        "is",
        "the",
        "employee",
        "id",
        "emp",
        "here",
        "please",
        "it",
        "a",
        "this",
    }
    if len(cleaned) <= 48 and tokens and all(tok in filler for tok in tokens):
        return eid
    return None


def is_knowledge_question(text: str) -> bool:
    """Heuristic: company / policy knowledge that should use RAG."""
    lowered = normalize_message(text).lower()
    return any(hint in lowered for hint in _KNOWLEDGE_HINTS)


def is_unsupported_domain(text: str) -> bool:
    """Heuristic: clearly outside Buddie's employee domain."""
    lowered = normalize_message(text).lower()
    return any(hint in lowered for hint in _UNSUPPORTED_HINTS)


def is_nonsense(text: str) -> bool:
    """True for keyboard-smash / non-linguistic input."""
    cleaned = normalize_message(text)
    if not cleaned:
        return False
    if _PUNCT_ONLY_RE.match(cleaned):
        return True
    if _DIGITS_ONLY_RE.match(cleaned):
        return False  # handled separately

    letters = re.findall(r"[a-zA-Z]", cleaned)
    alnum = re.findall(r"[a-zA-Z0-9]", cleaned)
    if not letters and alnum:
        # e.g. randomword123 handled below; pure symbols already caught.
        pass

    # Single token keyboard smash / low-vowel blobs.
    if " " not in cleaned and len(cleaned) >= 4:
        alpha = "".join(ch for ch in cleaned if ch.isalpha())
        if len(alpha) >= 4:
            vowels = sum(1 for ch in alpha.lower() if ch in "aeiou")
            if vowels <= 1:
                return True
            unique = len(set(alpha.lower()))
            if unique <= 3 and len(alpha) >= 5:
                return True

    # token + digits mash like hello123xyz / randomword123
    if re.fullmatch(r"[a-zA-Z]+\d+[a-zA-Z0-9]*", cleaned) and len(cleaned) >= 8:
        return True
    if re.fullmatch(r"[a-zA-Z]{6,}\d+", cleaned):
        return True

    if len(letters) >= 8 and " " not in cleaned:
        unique = len(set(ch.lower() for ch in cleaned if ch.isalpha()))
        if unique <= 4:
            return True
    return False


def has_business_intent(text: str) -> bool:
    """True when the message still asks for employee / policy / hybrid help."""
    routed = normalize_for_routing(text)
    if not routed:
        return False
    # Strip leading conversational wrappers, then inspect the remainder.
    remainder = _strip_conversational_wrappers(routed)
    if not remainder:
        return False
    # Pure identity / capability / casual small-talk is not business.
    if any(hint in remainder for hint in _IDENTITY_HINTS):
        return False
    if any(hint in remainder for hint in _CAPABILITY_HINTS):
        return False
    if any(phrase in remainder for phrase in _CASUAL_PHRASES):
        return False
    if any(hint in remainder for hint in _BUSINESS_HINTS):
        return True
    return False


def _strip_conversational_wrappers(routed: str) -> str:
    """Remove greeting / thanks / goodbye / filler shells; keep the core ask."""
    text = routed
    # Remove known multi-word goodbye / thanks phrases first.
    for phrase in sorted(_GOODBYE_PHRASES + _THANKS_PHRASES, key=len, reverse=True):
        text = text.replace(phrase, " ")
    tokens = text.split()
    while tokens and (
        tokens[0] in _GREETING_TOKENS
        or tokens[0] in _FILLER_TOKENS
        or tokens[0] in {"good", "thanks", "thank", "bye"}
    ):
        # Keep "good morning" handled as greeting elsewhere; here drop "good".
        tokens.pop(0)
    while tokens and (
        tokens[-1] in _FILLER_TOKENS
        or tokens[-1] in {"bye", "thanks", "thank", "buddie", "buddy"}
    ):
        tokens.pop()
    return " ".join(tokens).strip()


def _contains_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


def _is_greeting_only(routed: str) -> bool:
    if not routed:
        return False
    if routed in {"good morning", "good afternoon", "good evening"}:
        return True
    tokens = routed.split()
    if not tokens:
        return False
    # "hey how are you" — casual/greeting combo without business.
    if tokens[0] in _GREETING_TOKENS and not has_business_intent(routed):
        # Allow short greeting + optional name/filler / how-are-you.
        if len(tokens) <= 6 and not _contains_phrase(routed, _GOODBYE_PHRASES):
            # If leftover is only fillers / how are you / there — greeting.
            remainder = _strip_conversational_wrappers(routed)
            if not remainder or remainder in {
                "there",
                "how are you",
                "how are you doing",
                "buddie",
                "buddy",
            }:
                return True
            if all(tok in _FILLER_TOKENS | {"there", "how", "are", "you", "doing"} for tok in remainder.split()):
                return True
            # "hey buddie" already covered; longer non-business with greeting head.
            if len(tokens) <= 4 and all(
                tok in _GREETING_TOKENS | _FILLER_TOKENS | {"there", "good"}
                for tok in tokens
            ):
                return True
    return False


def _is_goodbye_only(routed: str) -> bool:
    if not routed or has_business_intent(routed):
        return False
    if not _contains_phrase(routed, _GOODBYE_PHRASES):
        return False
    # "bye the way ..." still has business → filtered by has_business_intent.
    remainder = _strip_conversational_wrappers(routed)
    if not remainder:
        return True
    # Pure filler + goodbye leftovers.
    return all(
        tok in _FILLER_TOKENS | {"thanks", "thank", "that", "all", "thats", "that's"}
        for tok in remainder.split()
    )


def _is_thanks_only(routed: str) -> bool:
    if not routed or has_business_intent(routed):
        return False
    if not _contains_phrase(routed, _THANKS_PHRASES):
        return False
    if _contains_phrase(routed, _GOODBYE_PHRASES):
        return False  # "thanks bye" → goodbye
    remainder = _strip_conversational_wrappers(routed)
    if not remainder:
        return True
    return all(
        tok
        in _FILLER_TOKENS
        | {
            "that",
            "thats",
            "that's",
            "helpful",
            "a",
            "lot",
            "so",
            "much",
            "for",
            "the",
            "help",
            "info",
            "information",
        }
        for tok in remainder.split()
    )


def _is_ack_only(routed: str) -> bool:
    if not routed or has_business_intent(routed):
        return False
    if _contains_phrase(routed, _GOODBYE_PHRASES) or _contains_phrase(
        routed, _THANKS_PHRASES
    ):
        return False
    if routed in _ACK_TOKENS:
        return True
    tokens = routed.split()
    if len(tokens) <= 3 and all(
        tok in _ACK_TOKENS | _FILLER_TOKENS | {"it", "that", "got", "makes", "sense", "sounds", "good"}
        for tok in tokens
    ):
        # Require at least one ack signal.
        joined = " ".join(tokens)
        return any(
            ack == joined or ack in tokens or joined.startswith(ack)
            for ack in _ACK_TOKENS
        )
    return False


def _prior_assistant_text(metadata: dict[str, Any] | None) -> str:
    if not metadata:
        return ""
    for key in (
        "last_assistant_message",
        "prior_assistant_message",
        "last_answer",
    ):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    history = metadata.get("conversation_history") or metadata.get("messages")
    if isinstance(history, list):
        for item in reversed(history):
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "").lower()
            if role in {"assistant", "buddie", "ai"}:
                content = item.get("content") or item.get("message") or ""
                if isinstance(content, str) and content.strip():
                    return content.strip().lower()
    return ""


def _ack_follows_helpful_answer(metadata: dict[str, Any] | None) -> bool:
    prior = _prior_assistant_text(metadata)
    if not prior:
        return False
    # After a real answer / verify success, short acks read as thanks.
    helpful_markers = (
        "vacation",
        "leave",
        "days",
        "holiday",
        "policy",
        "benefit",
        "verified",
        "remaining",
        "balance",
        "here",
        "found",
        "according",
        "handbook",
    )
    if any(marker in prior for marker in helpful_markers):
        return True
    return len(prior) > 40


def classify_intent(
    question: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> IntentDecision:
    """Classify a user message into a Buddie route.

    Conversational and unsupported intents short-circuit before RAG/tools.
    Business intents return ``PASS_THROUGH`` so the planner builds the plan.

    Args:
        question: Raw user text (kept intact for downstream planning).
        metadata: Optional session context (e.g. last assistant message).
    """
    cleaned = normalize_message(question)
    if not cleaned:
        return IntentDecision(route=IntentRoute.EMPTY, response=EMPTY_FALLBACK)

    # Bare digits are NOT employee ids — stay on the graceful fallback path.
    # Only E-#### (or short ID-submission phrases) enter verification.
    if _DIGITS_ONLY_RE.match(cleaned):
        return IntentDecision(
            route=IntentRoute.UNKNOWN,
            response=UNKNOWN_FALLBACK,
        )

    eid = extract_employee_id(cleaned)
    if eid:
        return IntentDecision(route=IntentRoute.VERIFY_ID, employee_id=eid)

    routed = normalize_for_routing(cleaned)

    # Meaningful business ask wins over casual wrappers ("hey, …", "cool, …").
    if has_business_intent(cleaned):
        logger.debug("Business intent preserved: preview=%r", cleaned[:80])
        return IntentDecision(route=IntentRoute.PASS_THROUGH)

    if _is_goodbye_only(routed):
        return IntentDecision(
            route=IntentRoute.CONVERSATION,
            kind=ConversationKind.GOODBYE,
            response=_goodbye_response(cleaned),
        )

    if _is_thanks_only(routed):
        return IntentDecision(
            route=IntentRoute.CONVERSATION,
            kind=ConversationKind.THANKS,
            response=_thanks_response(cleaned),
        )

    if _is_greeting_only(routed) or routed in {
        "good morning",
        "good afternoon",
        "good evening",
    }:
        return IntentDecision(
            route=IntentRoute.CONVERSATION,
            kind=ConversationKind.GREETING,
            response=_greeting_response(cleaned),
        )

    if any(hint in routed for hint in _IDENTITY_HINTS):
        return IntentDecision(
            route=IntentRoute.CONVERSATION,
            kind=ConversationKind.IDENTITY,
            response=(
                "I'm Buddie, your AI Employee Assistant. I can help you "
                "with everyday employee questions such as leave, holidays, "
                "benefits, and company policies."
            ),
        )

    if any(hint in routed for hint in _CAPABILITY_HINTS):
        return IntentDecision(
            route=IntentRoute.CONVERSATION,
            kind=ConversationKind.CAPABILITIES,
            response=(
                "I can help with things like leave, holidays, benefits, "
                "company policies, and your employee information."
            ),
        )

    if any(phrase in routed for phrase in _CASUAL_PHRASES) and not has_business_intent(
        cleaned
    ):
        return IntentDecision(
            route=IntentRoute.CONVERSATION,
            kind=ConversationKind.CASUAL,
            response=_casual_response(cleaned),
        )

    if _is_ack_only(routed):
        if _ack_follows_helpful_answer(metadata):
            return IntentDecision(
                route=IntentRoute.CONVERSATION,
                kind=ConversationKind.THANKS,
                response=_thanks_response(cleaned),
            )
        return IntentDecision(
            route=IntentRoute.CONVERSATION,
            kind=ConversationKind.ACK,
            response=_ack_response(cleaned),
        )

    if is_nonsense(cleaned) or _PUNCT_ONLY_RE.match(cleaned):
        return IntentDecision(route=IntentRoute.UNKNOWN, response=UNKNOWN_FALLBACK)

    if is_unsupported_domain(cleaned):
        return IntentDecision(
            route=IntentRoute.UNSUPPORTED,
            response=UNSUPPORTED_FALLBACK,
        )

    logger.debug("Intent left for business planner: preview=%r", cleaned[:80])
    return IntentDecision(route=IntentRoute.PASS_THROUGH)


def _greeting_response(original: str) -> str:
    lowered = original.strip().lower()
    if lowered.startswith("good morning") or "good morning" in normalize_for_routing(
        original
    ):
        return (
            "Good morning! 👋 I'm Buddie, your AI Employee Assistant. "
            "How can I help you today?"
        )
    if lowered.startswith("good afternoon"):
        return (
            "Good afternoon! 👋 I'm Buddie, your AI Employee Assistant. "
            "How can I help you today?"
        )
    if lowered.startswith("good evening"):
        return (
            "Good evening! 👋 I'm Buddie, your AI Employee Assistant. "
            "How can I help you today?"
        )
    if lowered.startswith("hello") or normalize_for_routing(original).startswith(
        "hello"
    ):
        return "Hello! 👋 How can I help you today?"
    if lowered.startswith("hey") or normalize_for_routing(original).startswith("hey"):
        return "Hey! 👋 How can I help you today?"
    return (
        "Hi! 👋 I'm Buddie, your AI Employee Assistant.\n"
        "How can I help you today?"
    )


def _thanks_response(original: str) -> str:
    del original
    return "You're welcome! 😊"


def _goodbye_response(original: str) -> str:
    del original
    return "Bye! 👋 Have a great day!"


def _ack_response(original: str) -> str:
    lowered = original.strip().lower()
    if lowered in {"yes", "yep", "yup", "sure"}:
        return "Great — what would you like help with?"
    if lowered in {"no", "nope"}:
        return "No problem. I'm here if you need anything else."
    if "hmm" in lowered or lowered in {"hm", "huh"}:
        return (
            "Take your time. I can help with leave, holidays, benefits, "
            "or company policies."
        )
    return "Got it! Let me know if there's anything else I can help with."


def _casual_response(original: str) -> str:
    lowered = original.strip().lower()
    if "how are you" in lowered:
        return (
            "I'm doing well, thanks for asking! "
            "How can I help with your employee questions today?"
        )
    if "are you there" in lowered or "you there" in lowered:
        return "Yes, I'm here! Ask me about leave, holidays, benefits, or policies."
    return (
        "Of course — I can help with leave, holidays, benefits, "
        "employee information, and company policies. What do you need?"
    )


def sanitize_user_facing_answer(answer: str) -> str:
    """Strip developer / offline implementation details from chat answers."""
    text = (answer or "").strip()
    if not text:
        return text

    patterns = (
        r"\n*\(Offline extractive answer from document.*?\)\s*",
        r"\n*—?\s*set OPENAI_API_KEY.*",
        r"OPENAI_API_KEY",
    )
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)

    text = re.sub(r"[A-Za-z]:\\[^\s`\]]+", "[document]", text)
    text = re.sub(r"/?(?:home|Users|tmp|var)/[^\s`\]]+", "[document]", text)
    text = re.sub(r"::p\d+::c\d+", "", text)

    cleaned = text.strip()
    if not cleaned:
        return GENERATION_FAILURE
    return cleaned


__all__ = [
    "IntentRoute",
    "ConversationKind",
    "IntentDecision",
    "VERIFY_PROMPT",
    "VERIFY_FAILED",
    "EMPLOYEE_ID_FORMAT_PROMPT",
    "UNKNOWN_FALLBACK",
    "UNSUPPORTED_FALLBACK",
    "EMPTY_FALLBACK",
    "GENERATION_FAILURE",
    "TOOL_FAILURE",
    "RAG_FAILURE",
    "normalize_message",
    "normalize_for_routing",
    "looks_like_employee_id",
    "extract_employee_id",
    "is_knowledge_question",
    "is_unsupported_domain",
    "is_nonsense",
    "has_business_intent",
    "classify_intent",
    "sanitize_user_facing_answer",
]

"""
Safety Guard — synchronous pre-LLM filter.

Runs BEFORE the classifier on every query. No LLM calls, no network calls,
pure local computation. Must complete in well under 10ms.

Strategy:
  1. Normalize input (lowercase, strip whitespace)
  2. Check if the query is educational (asking ABOUT a topic, not requesting action)
  3. Check against category-specific harmful patterns (intent + topic)
  4. Each blocked category returns a distinct professional refusal message

The educational detector uses "question-about" patterns (what is, explain, how does,
define, etc.) to whitelist queries that mention harmful topics but are clearly
seeking knowledge, not requesting illegal action.

The harmful detector looks for combinations of action intent (help me, how can I,
I want to) with topic-specific keywords. This catches "help me trade on insider
info" but passes "what is insider trading".

Tradeoff: Some edge cases near the boundary will be over-blocked. For example,
"I need to know the earnings before tomorrow's announcement" — this contains
action intent + insider topic. We err on the side of blocking.
"""
from __future__ import annotations

import re
from src.core.models import SafetyVerdict


# ---------------------------------------------------------------------------
# Category-specific refusal messages (must be distinct per assignment rules)
# ---------------------------------------------------------------------------

_REFUSAL_MESSAGES: dict[str, str] = {
    "insider_trading": (
        "I cannot assist with trading on material non-public information. "
        "Insider trading is illegal under securities law in most jurisdictions, "
        "including SEC Rule 10b-5 (US) and the Criminal Justice Act 1993 (UK). "
        "If you have questions about insider trading regulations, feel free to "
        "ask in an educational context."
    ),
    "market_manipulation": (
        "I cannot help with market manipulation schemes such as pump-and-dump, "
        "spoofing, wash trading, or coordinated price manipulation. These "
        "activities violate securities regulations worldwide and carry severe "
        "criminal penalties. I'm happy to explain how regulators detect and "
        "prevent these practices if you're interested."
    ),
    "money_laundering": (
        "I cannot assist with structuring transactions to avoid reporting "
        "thresholds, obscuring fund sources, or any form of money laundering. "
        "These activities violate anti-money laundering (AML) laws including "
        "the Bank Secrecy Act and EU Anti-Money Laundering Directives. "
        "I can explain AML compliance requirements if that would be helpful."
    ),
    "guaranteed_returns": (
        "I cannot promise or guarantee specific investment returns. No "
        "legitimate investment can guarantee returns — claims of guaranteed "
        "profits are a hallmark of fraud (e.g., Ponzi schemes). All investments "
        "carry risk, and past performance does not guarantee future results. "
        "I can discuss historical market returns and realistic expectations."
    ),
    "reckless_advice": (
        "I cannot encourage reckless financial decisions that could jeopardize "
        "your financial security. Concentrating all savings in a single volatile "
        "asset, using excessive leverage, or risking essential funds goes against "
        "sound financial principles. I can help you understand risk management "
        "and appropriate position sizing."
    ),
    "sanctions_evasion": (
        "I cannot assist with evading financial sanctions, OFAC restrictions, "
        "or any measures designed to prevent transactions with sanctioned "
        "entities or jurisdictions. Sanctions evasion is a serious federal "
        "crime. I can explain how sanctions compliance works if you're "
        "interested in the regulatory framework."
    ),
    "fraud": (
        "I cannot assist with fraudulent activities such as forging documents, "
        "fabricating financial records, or misrepresenting information. Fraud "
        "is a criminal offense carrying severe penalties. If you need help "
        "understanding fraud prevention, I'm happy to assist."
    ),
}


# ---------------------------------------------------------------------------
# Educational query patterns — these PASS even if harmful keywords are present
# ---------------------------------------------------------------------------

_EDUCATIONAL_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"^what\s+(is|are|does|was|were)\b",
        r"^what'?s\s+(the|a|an)\s+(difference|definition|meaning|penalty|role|history)",
        r"^explain\b",
        r"^define\b",
        r"^describe\b",
        r"^how\s+(does|do|did|is|are|can)\s+(the\s+)?(sec|fca|regulator|broker|compliance)",
        r"^how\s+does\s+\w+\s+(work|operate|function)",
        r"^what\s+(compliance|disclosure|reporting)\s+(obligations?|requirements?|rules?)",
        r"^(is|are)\s+\w+\s+(illegal|legal|legitimate)\??$",
        r"\b(what|how)\s+(is|are|does)\s+(the\s+)?(penalty|penalties|punishment|consequence)",
        r"\b(regulations?|regulatory|compliance|legal\s+framework)\s*(for|around|regarding|of)\b",
        r"^(is|are)\s+(short\s+selling|options?\s+trading|margin\s+trading)\s+(illegal|legal)",
        r"\brole\s+(of|in)\s+(preventing|detecting|combating|investigating)",
        r"\b(how|what)\s+(do|does|did)\s+(regulators?|authorities?|sec|fca)\s+(catch|detect|investigate|prevent)",
        r"^why\s+(is|are|do|does)\b.*\b(red\s+flag|illegal|fraud|prohibited)",
        r"\b(historical|average|typical)\s+(annual\s+)?return",
        r"\bwhat\s+factors\s+should\b",
        r"\bwhat\s+are\s+the\s+risks\s+of\b",
        r"\bhow\s+should\s+.*(be\s+allocated|allocat)",
        r"\bexplain\s+how\s+a\s+(ponzi|pyramid)\b",
        r"^are\s+pump.and.dump\b",
        r"\b(what|how)\s+(is|are)\s+(structuring|wash\s+trading|spoofing|layering|front.running)\b",
        r"\b(stages?|phases?)\s+of\s+money\s+laundering\b",
        r"\bdifference\s+between\b",
        r"\b(screening|screen)\s+for\b",
        r"\b(three|3)\s+stages\b",
    ]
]


# ---------------------------------------------------------------------------
# Harmful intent patterns — category-specific
# ---------------------------------------------------------------------------

# Action verbs that signal the user wants to DO something harmful
_ACTION_INTENTS = [
    r"\b(help|assist|tell|show|advise|guide)\s+me\b",
    r"\b(i\s+want|i\s+need|i'd\s+like|i\s+wish)\s+to\b",
    r"\bhow\s+(can|do|should|would)\s+i\b",
    r"\b(can\s+you|could\s+you|would\s+you)\b",
    r"\bdo\s+it\b",
    r"\blet'?s\b",
    r"\bdesign\s+(a|an|my)\b",
]

# Imperative patterns (direct commands)
_IMPERATIVE_INTENTS = [
    r"^(guarantee|promise|ensure|make\s+sure)\b",
    r"^(tell\s+me\s+(to|which|what)|give\s+me|put|move|route|hide)\b",
]

_HARMFUL_CATEGORIES: dict[str, dict] = {
    "insider_trading": {
        "keywords": [
            r"\b(insider|inside)\s+(trading|information|info|tip|knowledge|deal)",
            r"\b(material\s+)?non.?public\s+(information|info)",
            r"\bunannounced\s+(acquisition|merger|deal|earnings)",
            r"\bconfidential\s+(merger|acquisition|deal|information|news|info)",
            r"\btip\s+about\s+earnings",
            r"\bearnings\s+before\s+.*announcement",
            r"\bknow\s+about\s+an?\s+unannounced",
            r"\bgave\s+me\s+a\s+tip\b",
            r"\btrade\s+on\s+.*\b(confidential|non.?public|insider|private)\b",
            r"\bbefore\s+the\s+(call|announcement|earnings|report)\b",
            r"\bneed\s+to\s+know\s+the\s+earnings\b",
        ],
        "direct_block": [
            r"\btrade\s+on\s+.*confidential",
            r"\btip\s+about\s+earnings.*load\s+up",
            r"\bknow\s+about\s+.*unannounced.*buy\b",
            r"\bneed\s+to\s+know\s+the\s+earnings\s+before\b",
        ],
    },
    "market_manipulation": {
        "keywords": [
            r"\bpump\s+(up|and\s+dump|&\s+dump)",
            r"\bwash\s+trad(e|ing)\b",
            r"\bcoordinated\s+(buying|selling|trading|scheme)",
            r"\bmove\s+(this|the|a)\s+stock\b",
            r"\bcreate\s+(fake\s+)?volume\b",
            r"\bspoof(ing)?\b",
            r"\blayer(ing)?\s+(my\s+)?trades?\b",
            r"\bmanipulat(e|ing)\s+(the\s+)?(price|market|stock)",
        ],
        "direct_block": [
            r"\bpump\s+up\s+the\s+price\b",
            r"\bcoordinated\s+buying\s+scheme\b",
            r"\bwash\s+trade\s+between\b",
        ],
    },
    "money_laundering": {
        "keywords": [
            r"\bwithout\s+reporting\b",
            r"\bavoid\s+(the\s+)?(reporting|detection|10k|\$10)",
            r"\bstructur(e|ing)\s+(deposits?|transactions?|payments?)",
            r"\bobscure\s+(the\s+)?(source|origin)\s+(of\s+)?(these\s+)?(funds?|money)",
            r"\bhide\s+(trading\s+)?profits?\b",
            r"\blaunder(ing)?\b.*\b(money|funds|cash)\b",
            r"\bmove\s+.*cash\s+(through|via)\b.*\bwithout\b",
            r"\blayer\s+.*trades?\s+to\s+obscure\b",
            r"\bhide\b.*\b(from\s+)?(tax|authorities|irs|hmrc)",
        ],
        "direct_block": [
            r"\bmove\s+\d+.*without\s+reporting",
            r"\bstructure\s+deposits?\s+to\s+avoid\b",
            r"\blayer\s+.*trades?\s+to\s+obscure\b",
            r"\bhide\s+.*profits?\s+from\s+.*\b(tax|authorities)\b",
        ],
    },
    "guaranteed_returns": {
        "keywords": [
            r"\bguarantee\s+.*\b(return|profit|gain|money|income)",
            r"\bguaranteed\b.*\b(return|profit|performance)",
            r"\bpromise\s+.*\b(money|double|triple|return|profit)",
            r"\b100\s*%\s*(certain|sure|guaranteed|safe|risk.?free)",
            r"\bfoolproof\s+(way|method|strategy)\b",
            r"\brisk.?free\s+(return|profit|investment)",
            r"\bcertain\s+to\s+go\s+up\b",
            r"\b(double|triple)\s+(my\s+)?(money|investment)\s+in\s+(a|one)\b",
            r"\bguarantee\s+me\b",
        ],
        "direct_block": [
            r"\bguarantee\s+me\s+\d+",
            r"\bpromise\s+me\s+my\s+money\s+will\b",
            r"\b100\s*%\s*certain\s+to\s+go\s+up\b",
            r"\bfoolproof\s+way\s+to\s+make\s+\d+",
        ],
    },
    "reckless_advice": {
        "keywords": [
            r"\ball\s+(my\s+)?(retirement|savings?|emergency\s+fund|life\s+savings?)\s+(in|into)\b",
            r"\bput\s+(all|everything|entire)\b.*\b(into|in)\s+(crypto|bitcoin|btc|options?|meme|penny)",
            r"\bmargin\s+loan\s+to\s+buy\b",
            r"\bmortgage\s+my\s+(house|home)\s+(for|to)\b",
            r"\b(entire|whole)\s+emergency\s+fund\s+(into|in)\b",
            r"\btell\s+me\s+(to\s+)?(take|get)\s+a\s+margin\b",
            r"\btell\s+me\s+which\s+stock\s+to\s+mortgage\b",
        ],
        "direct_block": [
            r"\bput\s+all\s+.*retirement.*\bin\s+(crypto|bitcoin)",
            r"\btell\s+me\s+to\s+take\s+a\s+margin\s+loan\b",
            r"\bentire\s+emergency\s+fund\s+(into|in)\s+options\b",
            r"\bmortgage\s+my\s+house\s+for\b",
        ],
    },
    "sanctions_evasion": {
        "keywords": [
            r"\bbypass\s+(ofac\s+)?sanctions\b",
            r"\bevade?\s+sanctions?\b",
            r"\bsanctioned\s+(russian|iranian|north\s+korean|syrian)\b.*\bwithout\b",
            r"\bshell\s+company\s+to\s+bypass\b",
            r"\bwithout\s+.*\btraced\b.*\bsanctioned\b",
            r"\broute\s+.*through\s+.*\bshell\b.*\bsanctions\b",
            r"\binvest\s+in\s+.*sanctioned\b.*\bwithout\b.*\btraced\b",
        ],
        "direct_block": [
            r"\broute\s+.*through\s+.*shell.*bypass.*sanctions\b",
            r"\bsanctioned\s+.*without\s+.*traced\b",
        ],
    },
    "fraud": {
        "keywords": [
            r"\bfake\s+(contract|document|note|receipt|statement|report)",
            r"\bforg(e|ed|ing)\s+(a\s+)?(document|contract|signature|note)",
            r"\bfabricat(e|ing)\s+(a\s+)?(document|record|report|claim)",
            r"\bdraft\s+a\s+fake\b",
            r"\bclaim\s+false\s+losses\b",
            r"\bfraudulent\s+(document|claim|report)",
        ],
        "direct_block": [
            r"\bdraft\s+a\s+fake\s+contract",
            r"\bfake\s+contract\s+note\s+to\s+claim\b",
        ],
    },
}


# ---------------------------------------------------------------------------
# Compiled patterns (compiled once at module load for performance)
# ---------------------------------------------------------------------------

_COMPILED_ACTION = [re.compile(p, re.IGNORECASE) for p in _ACTION_INTENTS]
_COMPILED_IMPERATIVE = [re.compile(p, re.IGNORECASE) for p in _IMPERATIVE_INTENTS]

_COMPILED_HARMFUL: dict[str, dict[str, list[re.Pattern]]] = {}
for _cat, _patterns in _HARMFUL_CATEGORIES.items():
    _COMPILED_HARMFUL[_cat] = {
        "keywords": [re.compile(p, re.IGNORECASE) for p in _patterns["keywords"]],
        "direct_block": [re.compile(p, re.IGNORECASE) for p in _patterns["direct_block"]],
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check(query: str) -> SafetyVerdict:
    """
    Check a user query for harmful intent.

    Returns SafetyVerdict with blocked=True if the query should be refused,
    along with the category and a distinct professional message.
    Returns blocked=False if the query is safe to process.
    """
    if not query or not query.strip():
        return SafetyVerdict(blocked=False)

    normalized = query.strip()

    # Step 1: Check if this is an educational query
    if _is_educational(normalized):
        return SafetyVerdict(blocked=False)

    # Step 2: Check each harmful category
    for category, patterns in _COMPILED_HARMFUL.items():
        # Direct block patterns — high confidence, block immediately
        for pat in patterns["direct_block"]:
            if pat.search(normalized):
                return SafetyVerdict(
                    blocked=True,
                    category=category,
                    message=_REFUSAL_MESSAGES[category],
                )

        # Action intent + harmful keyword combination
        if _has_action_intent(normalized):
            for pat in patterns["keywords"]:
                if pat.search(normalized):
                    return SafetyVerdict(
                        blocked=True,
                        category=category,
                        message=_REFUSAL_MESSAGES[category],
                    )

        # Strong keyword patterns that indicate harm even without explicit action intent
        for pat in patterns["keywords"]:
            if pat.search(normalized):
                # Check if the match is strong enough to block without action intent
                if _is_strong_harmful_signal(normalized, category):
                    return SafetyVerdict(
                        blocked=True,
                        category=category,
                        message=_REFUSAL_MESSAGES[category],
                    )

    return SafetyVerdict(blocked=False)


def _is_educational(text: str) -> bool:
    """Check if the query is educational in nature (asking about a topic)."""
    for pat in _EDUCATIONAL_PATTERNS:
        if pat.search(text):
            return True
    return False


def _has_action_intent(text: str) -> bool:
    """Check if the query contains action-oriented language."""
    for pat in _COMPILED_ACTION:
        if pat.search(text):
            return True
    for pat in _COMPILED_IMPERATIVE:
        if pat.search(text):
            return True
    return False


def _is_strong_harmful_signal(text: str, category: str) -> bool:
    """
    Some patterns are strong enough to block even without explicit action intent.
    e.g. "guarantee me 30% returns" — the verb "guarantee" IS the action.
    e.g. "i need to know the earnings before the announcement" — implies insider intent.
    """
    lower = text.lower()

    if category == "guaranteed_returns":
        # "guarantee me", "promise me", "100% certain", "foolproof"
        if re.search(r"\b(guarantee|promise)\s+me\b", lower):
            return True
        if re.search(r"\b100\s*%\s*(certain|sure|guaranteed)\b", lower):
            return True
        if re.search(r"\bfoolproof\b", lower):
            return True
        if re.search(r"\bcertain\s+to\s+go\s+up\b", lower):
            return True

    elif category == "insider_trading":
        # "i need to know the earnings before" — clear insider intent
        if re.search(r"\bneed\s+to\s+know\s+the\s+earnings\s+before\b", lower):
            return True
        # "gave me a tip about earnings...load up before the call"
        if re.search(r"\btip\s+about\s+earnings\b", lower) and re.search(r"\b(load\s+up|buy|purchase)\b", lower):
            return True
        # "I work at X and know about unannounced"
        if re.search(r"\b(work|working)\s+at\b", lower) and re.search(r"\bunannounced\b", lower):
            return True

    elif category == "reckless_advice":
        # "i'm 70 and want to put all my retirement savings in crypto, do it"
        if re.search(r"\b(put|invest|move)\s+(all|everything|entire|whole)\b", lower):
            return True
        if re.search(r"\btell\s+me\s+(to|which)\b", lower):
            return True
        if re.search(r"\bentire\s+emergency\s+fund\b", lower):
            return True
        if re.search(r"\bmortgage\s+my\s+(house|home)\b", lower):
            return True

    elif category == "money_laundering":
        # "move 500k...without reporting"
        if re.search(r"\bwithout\s+reporting\b", lower):
            return True
        if re.search(r"\bavoid\s+.*reporting\s+threshold\b", lower):
            return True
        if re.search(r"\bhide\s+.*profits?\b", lower):
            return True
        if re.search(r"\bobscure\s+the\s+source\b", lower):
            return True

    elif category == "market_manipulation":
        # "pump up the price"
        if re.search(r"\bpump\s+up\b", lower):
            return True
        if re.search(r"\bwash\s+trade\b", lower) and re.search(r"\b(between|across|my)\b", lower):
            return True
        if re.search(r"\bcoordinated\s+(buying|selling)\b", lower):
            return True

    elif category == "sanctions_evasion":
        if re.search(r"\bbypass\b.*\bsanctions\b", lower):
            return True
        if re.search(r"\bsanctioned\b.*\bwithout\b.*\btraced\b", lower):
            return True

    elif category == "fraud":
        if re.search(r"\bfake\s+(contract|document)\b", lower):
            return True
        if re.search(r"\bdraft\s+a\s+fake\b", lower):
            return True

    return False

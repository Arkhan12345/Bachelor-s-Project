from __future__ import annotations

import ast
import operator
import re
from collections import OrderedDict
from threading import RLock
from typing import Dict, Iterable, List, Optional


PUBLICATION_TERMS = {
    "publication",
    "publications",
    "pubmed",
    "literature",
}

PUBLICATION_DOCUMENT_TERMS = {
    "paper",
    "papers",
    "article",
    "articles",
}

SAMPLE_TERMS = {
    "sample",
    "samples",
    "annotation",
    "annotations",
    "subtype",
    "recurrence",
    "survival",
    "platinum",
    "taxol",
    "debulking",
}

PATHWAY_TERMS = {
    "pathway",
    "pathways",
    "enrichment",
    "enriched",
    "ic",
    "ica",
}

def normalize_chat_text(text: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        re.sub(r"[^a-z0-9\s]", " ", (text or "").lower()),
    ).strip()


def is_greeting(message: str) -> bool:
    normalized = re.sub(r"[^a-z\s]", "", (message or "").lower()).strip()
    return normalized in {
        "hi",
        "hello",
        "hey",
        "hey there",
        "hi there",
        "hello there",
        "good morning",
        "good afternoon",
        "good evening",
    }


def is_vague_followup(message: str) -> bool:
    normalized = normalize_chat_text(message)
    if not normalized:
        return False
    if normalized in {
        "more",
        "a bit more",
        "tell me more",
        "can you tell me more",
        "something else",
        "anything else",
        "what else",
        "go on",
        "continue",
        "expand",
        "elaborate",
    }:
        return True
    words = normalized.split()
    return len(words) <= 5 and any(
        word in words for word in ("more", "else", "expand", "elaborate")
    )


def _contains_term(words: Iterable[str], terms: set[str]) -> bool:
    return any(word in terms for word in words)


def _last_scope(history: List[Dict[str, str]]) -> Optional[str]:
    for item in reversed(history):
        if item.get("role") == "user" and item.get("scope"):
            return item["scope"]
    return None


def classify_chat_scope(message: str, history: List[Dict[str, str]]) -> str:
    """Classify which page evidence, if any, should accompany a question."""
    normalized = normalize_chat_text(message)
    words = normalized.split()

    if is_vague_followup(message):
        return _last_scope(history) or "general"
    if _contains_term(words, PUBLICATION_TERMS):
        return "publications"
    if _contains_term(words, PUBLICATION_DOCUMENT_TERMS) and any(
        cue in normalized
        for cue in (
            "first ",
            "loaded ",
            "listed ",
            "this paper",
            "the paper",
            "these papers",
            "for this ic",
            "for this gene",
        )
    ):
        return "publications"
    if _contains_term(words, SAMPLE_TERMS):
        return "samples"
    if _contains_term(words, {"age", "grade", "stage"}) and _contains_term(
        words,
        {"median", "mean", "high", "low", "sample", "samples", "cohort", "tumor", "tumour", "ic"},
    ):
        return "samples"
    if _contains_term(words, PATHWAY_TERMS) or re.search(r"\bic\s*\d+\b", normalized):
        return "pathways"
    if "growth promoting" in normalized or "this component" in normalized:
        return "pathways"
    if any(
        phrase in normalized
        for phrase in (
            "this gene",
            "selected gene",
            "related gene",
            "these results",
            "this result",
            "current analysis",
        )
    ):
        return "domain"
    return "general"


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _evaluate_arithmetic(node: ast.AST) -> float | int:
    if isinstance(node, ast.Expression):
        return _evaluate_arithmetic(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        if abs(node.value) > 1_000_000_000_000:
            raise ValueError("number is too large")
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_evaluate_arithmetic(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate_arithmetic(node.left)
        right = _evaluate_arithmetic(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 12:
            raise ValueError("exponent is too large")
        result = _BINARY_OPERATORS[type(node.op)](left, right)
        if abs(result) > 1_000_000_000_000_000:
            raise ValueError("result is too large")
        return result
    raise ValueError("unsupported expression")


def try_answer_calculation(message: str) -> Optional[str]:
    """Answer small arithmetic expressions exactly without asking the LLM."""
    expression = (message or "").strip()
    expression = expression.replace("×", "*").replace("÷", "/").replace("^", "**")
    expression = re.sub(r"(?<=\d)\s*[xX]\s*(?=\d)", "*", expression)

    prompt_match = re.fullmatch(
        r"\s*(?:what(?:'s| is)?|calculate|compute|solve)\s+(.+?)\s*[?.!]?\s*",
        expression,
        flags=re.IGNORECASE,
    )
    if prompt_match:
        expression = prompt_match.group(1)
    else:
        expression = expression.rstrip("?.!").strip()

    if not expression or len(expression) > 120:
        return None
    if not re.fullmatch(r"[0-9\s+\-*/().%]+", expression):
        return None

    try:
        tree = ast.parse(expression, mode="eval")
        if sum(1 for _ in ast.walk(tree)) > 30:
            return None
        result = _evaluate_arithmetic(tree)
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError, OverflowError):
        return None

    if isinstance(result, float):
        if not (float("-inf") < result < float("inf")):
            return None
        if result.is_integer():
            return str(int(result))
        return f"{result:.10g}"
    return str(result)


class ConversationStore:
    """Small thread-safe server-side LRU store for bounded chat histories."""

    def __init__(self, max_messages: int = 12, max_conversations: int = 256):
        self.max_messages = max(2, max_messages)
        self.max_conversations = max(1, max_conversations)
        self._items: OrderedDict[str, List[Dict[str, str]]] = OrderedDict()
        self._lock = RLock()

    def get(self, key: str) -> List[Dict[str, str]]:
        with self._lock:
            history = self._items.get(key, [])
            if key in self._items:
                self._items.move_to_end(key)
            return [dict(item) for item in history]

    def set(self, key: str, history: List[Dict[str, str]]) -> None:
        with self._lock:
            self._items[key] = [
                dict(item) for item in history[-self.max_messages :]
            ]
            self._items.move_to_end(key)
            while len(self._items) > self.max_conversations:
                self._items.popitem(last=False)

    def clear(self, key: str) -> None:
        with self._lock:
            self._items.pop(key, None)

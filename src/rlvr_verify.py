"""Deterministic checkers for allenai/RLVR-IFeval ground_truth (func_name + params),
approximating open-instruct's IFEval verifiers."""
import json, re


def _words(t):
    return re.findall(r"\b\w+\b", t)


def _cmp(n, N, q):
    q = (q or "at least").lower()
    if q == "at least":
        return n >= N
    if q == "at most":
        return n <= N
    if q == "around":
        return abs(n - N) <= max(1, round(N * 0.1))
    return n == N


def _paras(t, splitter=r"\n\s*\*\*\*\s*\n"):
    return [p for p in re.split(splitter, t.strip()) if p.strip()]


def _count_sentences(t):
    """Counts sentences ending in [.!?] in O(N) linear time, avoiding ReDoS backtracking."""
    count = 0
    in_text = False
    for ch in t:
        if ch not in ".!?":
            in_text = True
        elif in_text:
            count += 1
            in_text = False
    return count


def verify(text, gt):
    g = json.loads(gt) if isinstance(gt, str) else gt
    f, t = g["func_name"], text.strip()
    N, q = g.get("N"), g.get("quantifier")
    if f == "validate_lowercase":
        return t == t.lower()
    if f == "validate_uppercase":
        return t == t.upper()
    if f == "validate_no_commas":
        return "," not in t
    if f == "validate_quotation":
        return len(t) > 1 and t[0] == '"' and t[-1] == '"'
    if f == "validate_json_format":
        s = re.sub(r"^```(json)?|```$", "", t).strip()
        try:
            json.loads(s); return True
        except Exception:
            return False
    if f == "validate_title":
        return re.search(r"<<[^\n]+>>", t) is not None
    if f == "validate_end":
        return t.endswith(g["end_phrase"].strip())
    if f == "verify_postscript":
        return re.search(r"\s*" + re.escape(g["postscript_marker"].strip()), t) is not None
    if f == "validate_placeholders":
        return len(re.findall(r"\[[^\]]+\]", t)) >= N
    if f == "validate_highlighted_sections":
        return len(re.findall(r"\*[^\n\*]+\*", t)) >= N
    if f == "validate_sections":
        return len(re.split(r"\s?" + re.escape(g["section_splitter"]) + r"\s?\d+\s?", t)) - 1 >= N
    if f == "verify_bullet_points":
        return len(re.findall(r"^\s*[\*\-]\s", t, flags=re.M)) == N
    if f == "verify_keywords":
        return all(re.search(re.escape(k), t, flags=re.I) for k in g["keyword_list"])
    if f == "verify_keyword_frequency":
        return len(re.findall(r"\b" + re.escape(g["word"]) + r"\b", t, flags=re.I)) >= N
    if f == "validate_forbidden_words":
        return not any(re.search(r"\b" + re.escape(w) + r"\b", t, flags=re.I) for w in g["forbidden_words"])
    if f == "verify_letter_frequency":
        return t.lower().count(g["letter"].lower()) >= N
    if f == "validate_frequency_capital_words":
        return _cmp(len([w for w in _words(t) if w.isupper() and len(w) > 1]), N, q)
    if f == "validate_word_constraint":
        return _cmp(len(_words(t)), N, q)
    if f == "verify_sentence_constraint":
        return _cmp(_count_sentences(t), N, q)
    if f == "verify_paragraph_count":
        return len(_paras(t)) == N
    if f == "validate_paragraphs":
        ps = _paras(t, r"\n\s*\n")
        return len(ps) == N and len(ps) >= g["i"] and ps[g["i"] - 1].strip().split()[0].strip("\"'*,.").lower() == g["first_word"].lower()
    if f == "validate_two_responses":
        parts = [p.strip() for p in t.split("******")]
        return len(parts) == 2 and all(parts) and parts[0] != parts[1]
    if f == "validate_choice":
        opts = [o.strip() for o in re.split(r",\s*", g["options"]) if o.strip()]
        return any(o in t for o in opts)
    if f == "validate_repeat_prompt":
        orig = g["original_prompt"].split("First, repeat the request")[0].strip()
        return t.startswith(orig[:80])
    return False

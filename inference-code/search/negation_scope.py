"""
Splitting free text into its affirmed and negated parts (EN + VI).

Stdlib-only (just `re`), no imports from config/models/kis_c_scoring - it
deliberately returns raw TEXT rather than tokens so that
`kis_c_scoring.tokenize_answer` stays the single tokenizer (its STOPWORDS and
MIN_TOKEN_LEN decisions are not duplicated here) without creating an import
cycle between the two modules. Split out of kis_c_scoring.py to keep both
files under this repo's 200-line convention.

Why this exists: an operator answering a KIS-C clarifying question very often
answers in the negative - "no, not red, the jacket is blue", "không phải màu
đỏ, áo màu xanh". Matching such an answer as a flat bag of words scores the
ruled-out candidate exactly as highly as the correct one, because the negated
attribute ("red"/"đỏ") is still lexically present in its caption. Nor can the
retriever be relied on to sort it out: NevIR (Weller et al., EACL 2024, arXiv:2305.07614) finds
bi-encoder and sparse architectures rank negated pairs WORSE than random, and
cross-encoders only slightly above.
"""
import re

# Words that flip the polarity of whatever follows them in the same clause.
NEGATION_CUES = frozenset({
    "no", "not", "non", "without", "never", "nor",
    "isnt", "arent", "dont", "doesnt", "didnt", "wasnt", "werent",
    "khong", "không", "chua", "chưa", "chang", "chẳng",
})

# Words that carry no meaning of their own when they immediately follow a cue
# ("không phải ...", "không có ..."). Consumed as part of the cue so they never
# land in the negated set, where they would penalise any candidate whose text
# happens to contain them - "phải" in particular is a content word elsewhere
# ("bên phải" = on the right), so it must not be added to STOPWORDS globally.
CUE_CONTINUATIONS = frozenset({"phai", "phải", "co", "có", "la", "là"})

# A negation's scope ends at the clause boundary, not at the end of the string:
# in "không phải màu đỏ, áo màu xanh" only the first clause is negated.
CLAUSE_SPLIT = re.compile(r"[,;.!?]+|\bnhưng\b|\bnhung\b|\bbut\b", re.UNICODE)


def split_negation_scope(text: str) -> tuple:
    """
    Returns `(affirmed_text, negated_text)` - the words the answer asserts and
    the words it rules out, each as a plain space-joined string for the caller
    to tokenize.

    Scope rule (deliberately simple, and simple is load-bearing here: this runs
    on a live 7-minute competition clock and must never raise): within each
    clause, everything from the first negation cue to the end of that clause is
    negated; everything before the cue, and every clause without one, is
    affirmed. So "áo không có màu đỏ" affirms "áo" and negates "màu đỏ".

    Text with no negation cue at all returns `(text_words, "")`, which makes
    the caller's behaviour identical to the previous flat-bag matching - the
    common affirmative case is unaffected.
    """
    if not text:
        return "", ""

    affirmed, negated = [], []
    for clause in CLAUSE_SPLIT.split(text.lower()):
        words = re.findall(r"\w+", clause, re.UNICODE)
        in_negation = False
        skip_next = False
        for index, word in enumerate(words):
            if skip_next:
                skip_next = False
                continue
            if word in NEGATION_CUES:
                in_negation = True
                if index + 1 < len(words) and words[index + 1] in CUE_CONTINUATIONS:
                    skip_next = True
                continue
            (negated if in_negation else affirmed).append(word)

    return " ".join(affirmed), " ".join(negated)

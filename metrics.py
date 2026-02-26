"""Evaluation metrics for OCR benchmarks.

TEDS implementation is from OmniDocBench (IBM's PubTabNet original).
CDM evaluation uses OmniDocBench's bundled CDM module.
"""

from __future__ import annotations

import importlib.util
import random
import re

import evaluate as hf_evaluate
from rapidfuzz.distance import Levenshtein

# ─── Load OmniDocBench modules via importlib to avoid name collision ──
# metrics.py (this file) shadows the OmniDocBench `metrics` package,
# so `from metrics.table_metric import TEDS` resolves to this file.
# Use importlib.util.spec_from_file_location to load directly by path.
_OMNIDOCBENCH_METRICS = "/home/ubuntu/OmniDocBench/metrics"


def _load_omnidoc_module(name: str, filename: str):
    """Load a module from OmniDocBench/metrics/ by absolute path."""
    path = f"{_OMNIDOCBENCH_METRICS}/{filename}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ─── OmniDocBench text preprocessing (from utils/data_preprocess.py) ──

# Matches inline LaTeX: $...$ or \(...\)
_INLINE_REG = re.compile(r'\$([^$]+)\$|\\\(([^)]+)\\\)')


def textblock2unicode(text: str) -> str:
    """Convert inline LaTeX formulas to unicode representation.

    Matches OmniDocBench/utils/data_preprocess.py textblock2unicode().
    Uses pylatexenc LatexNodes2Text to convert inline math to unicode.
    """
    from pylatexenc.latex2text import LatexNodes2Text

    inline_matches = _INLINE_REG.finditer(text)
    removal_positions = []
    for match in inline_matches:
        position = [match.start(), match.end()]
        content = match.group(1) if match.group(1) is not None else match.group(2)
        clean_content = re.sub(r'\\([\\_&%^])', '', content)
        try:
            if any(char in clean_content for char in r'\^_'):
                if clean_content.endswith('\\'):
                    clean_content += ' '
                unicode_content = LatexNodes2Text().latex_to_text(clean_content)
                removal_positions.append((position[0], position[1], unicode_content))
        except Exception:
            continue

    for start, end, unicode_content in sorted(removal_positions, reverse=True):
        text = text[:start] + unicode_content.strip() + text[end:]
    return text


def clean_string(input_string: str) -> str:
    """Remove special characters, keep Chinese/English/digits only.

    Matches OmniDocBench/utils/data_preprocess.py clean_string().
    """
    input_string = input_string.replace('\\t', '').replace('\\n', '').replace('\t', '').replace('\n', '').replace('/t', '').replace('/n', '')
    return re.sub(r'[^\w\u4e00-\u9fff]', '', input_string)


def omnidocbench_preprocess(text: str) -> str:
    """Full OmniDocBench text preprocessing pipeline: textblock2unicode + clean_string."""
    return clean_string(textblock2unicode(text))


# ─── Text-level metrics ───────────────────────────────────────────────

def normalized_edit_distance(pred: str, gt: str) -> float:
    """Normalized Edit Distance (NED). 0 = identical, 1 = completely different.

    Matches OmniDocBench: Levenshtein.distance / max(len(pred), len(gt)).
    """
    if not gt and not pred:
        return 0.0
    max_len = max(len(pred), len(gt))
    if max_len == 0:
        return 0.0
    return Levenshtein.distance(pred, gt) / max_len


def compute_cer(pred: str, gt: str) -> float:
    """Character Error Rate = edit_distance / len(gt).

    Standard IAM/HTR metric. Normalizes by ground truth length (not max).
    Lower = better. Can exceed 1.0 if prediction is much longer than GT.
    """
    if not gt:
        return 0.0 if not pred else 1.0
    return Levenshtein.distance(pred, gt) / len(gt)


def compute_wer(pred: str, gt: str) -> float:
    """Word Error Rate = word-level edit distance / num_gt_words.

    Standard IAM/HTR metric. Uses Levenshtein on word sequences.
    Lower = better.
    """
    gt_words = gt.split()
    pred_words = pred.split()
    if not gt_words:
        return 0.0 if not pred_words else 1.0
    return Levenshtein.distance(pred_words, gt_words) / len(gt_words)


# ─── BLEU (HuggingFace evaluate — matches OmniDocBench/UniMERNet) ─────

def compute_bleu(pred: str, gt: str) -> float:
    """BLEU score using HuggingFace evaluate library.

    Matches OmniDocBench (cal_metric.py call_BLEU) and UniMERNet (test.py score_text).
    Both use evaluate.load("bleu").compute(predictions=..., references=...).
    """
    if not pred.strip() or not gt.strip():
        return 0.0
    bleu = hf_evaluate.load("bleu", keep_in_memory=True, experiment_id=random.randint(1, int(1e8)))
    try:
        result = bleu.compute(predictions=[pred], references=[gt])
        return result["bleu"]
    except (ZeroDivisionError, ValueError):
        return 0.0


# ─── NID (Normalized Indel Distance) ──────────────────────────────────

def compute_nid(pred: str, gt: str) -> float:
    """Normalized Indel Similarity for Upstage DP-Bench.

    Uses only insertion and deletion (no substitution).
    NID = 1 - indel_distance / (len(pred) + len(gt))
    Returns 1 for identical strings, 0 for completely different.
    Higher is better.
    """
    from rapidfuzz.distance import Indel
    if not gt and not pred:
        return 1.0
    total_len = len(pred) + len(gt)
    if total_len == 0:
        return 1.0
    return 1.0 - Indel.distance(pred, gt) / total_len


# ─── TEDS (Official OmniDocBench / PubTabNet implementation) ──────────

def _wrap_table_html(s: str) -> str:
    """Ensure HTML string is wrapped in <html><body> for lxml parsing.

    lxml.html.fromstring() on a bare <table> returns the element directly
    (not wrapped), so TEDS's xpath('body/table') finds nothing and returns 0.
    Wrapping guarantees the expected document structure.
    """
    stripped = s.strip()
    if stripped.lower().startswith("<html"):
        return stripped
    return f"<html><body>{stripped}</body></html>"


def _normalize_table_html(html_str: str) -> str:
    """Minimal table HTML normalization for TEDS comparison.

    Uses lxml (same parser as TEDS internally) to avoid tree structure
    mismatches that occur with BeautifulSoup serialization.

    Collapses whitespace in ALL text/tail nodes within cells (including
    descendants like <b>, <i>, <span>). PubTabNet GT is pretty-printed
    with newlines/spaces inside formatting tags that would otherwise
    inflate char-level edit distance in TEDS.

    th→td and thead unwrap are handled inside TEDS.evaluate() itself.
    """
    from lxml import html as _lxml_html, etree as _etree

    parser = _lxml_html.HTMLParser(remove_comments=True, encoding='utf-8')
    try:
        doc = _lxml_html.fromstring(html_str.encode('utf-8'), parser=parser)
    except Exception:
        return html_str

    # Collapse whitespace in ALL text nodes within cells (td and th)
    for cell in doc.xpath('.//td') + doc.xpath('.//th'):
        # Cell's own text
        if cell.text:
            cell.text = re.sub(r'\s+', ' ', cell.text).strip()
        # All descendant elements' text and tail
        for desc in cell.iter():
            if desc is cell:
                continue
            if desc.text:
                desc.text = re.sub(r'\s+', ' ', desc.text).strip()
            if desc.tail:
                desc.tail = re.sub(r'\s+', ' ', desc.tail).strip()

    return _etree.tostring(doc, encoding='unicode')


def compute_teds(pred_html: str, gt_html: str, structure_only: bool = False) -> float:
    """TEDS score using official OmniDocBench implementation.

    This uses the IBM PubTabNet TEDS code from OmniDocBench which:
    - Tokenizes cell content (tags + characters)
    - Uses Levenshtein distance for cell content comparison in CustomConfig
    - Properly handles colspan/rowspan attributes
    - Returns similarity in [0, 1]. 1 = identical.

    Both pred and GT are normalized (<th>→<td>, whitespace stripped).
    """
    _table_mod = _load_omnidoc_module("table_metric", "table_metric.py")
    TEDS = _table_mod.TEDS
    teds = TEDS(structure_only=structure_only)
    try:
        norm_pred = _normalize_table_html(_wrap_table_html(pred_html))
        norm_gt = _normalize_table_html(_wrap_table_html(gt_html))
        return teds.evaluate(norm_pred, norm_gt)
    except Exception:
        return 0.0


# ─── CDM (Character Detection Matching for Formula) ──────────────────

def _load_cdm_modules():
    """Load CDM-related modules via importlib to avoid 'metrics' name collision.

    Returns (CDM_class, latex2bbox_color_module).
    """
    import sys

    _base = f"{_OMNIDOCBENCH_METRICS}/cdm/modules"

    # 1. Load sub-dependencies first (latex2bbox_color needs these)
    _tokenize_mod = _load_omnidoc_module(
        "metrics.cdm.modules.tokenize_latex.tokenize_latex",
        "cdm/modules/tokenize_latex/tokenize_latex.py",
    )
    sys.modules["metrics.cdm.modules.tokenize_latex.tokenize_latex"] = _tokenize_mod

    _latex_proc = _load_omnidoc_module(
        "metrics.cdm.modules.latex_processor",
        "cdm/modules/latex_processor.py",
    )
    sys.modules["metrics.cdm.modules.latex_processor"] = _latex_proc

    _visual_matcher = _load_omnidoc_module(
        "metrics.cdm.modules.visual_matcher",
        "cdm/modules/visual_matcher.py",
    )
    sys.modules["metrics.cdm.modules.visual_matcher"] = _visual_matcher

    # 2. Load latex2bbox_color
    l2b = _load_omnidoc_module(
        "metrics.cdm.modules.latex2bbox_color",
        "cdm/modules/latex2bbox_color.py",
    )
    sys.modules["metrics.cdm.modules.latex2bbox_color"] = l2b

    # 3. Load CDM class (cdm_metric.py has relative imports to .cdm.modules)
    _cdm_mod = _load_omnidoc_module("metrics.cdm_metric", "cdm_metric.py")
    CDM = _cdm_mod.CDM

    return CDM, l2b


# Non-CJK template for pdflatex (no Source Han Sans SC font needed)
_PDFLATEX_FORMULA_TEMPLATE = r"""
\documentclass[12pt]{article}
\usepackage[landscape]{geometry}
\usepackage{geometry}
\geometry{a<PaperSize>paper,scale=0.98}
\pagestyle{empty}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{amssymb}
\usepackage{upgreek}
\usepackage{amsmath}
\usepackage{xcolor}
\begin{document}
\makeatletter
\renewcommand*{\@textcolor}[3]{%%
  \protect\leavevmode
  \begingroup
    \color#1{#2}#3%%
  \endgroup
}
\makeatother
%% Polyfill \mathcolor for TexLive < 2023 (CDM uses it for per-token coloring)
\ifdefined\mathcolor\else
  \let\mathcolor\textcolor
\fi
\begin{displaymath}
%s
\end{displaymath}
\end{document}
"""


_cdm_cache: tuple | None = None  # (CDM_class, l2b_module)


def _ensure_cdm_patches():
    """Apply monkey-patches for CDM once (pdflatex, ImageMagick v6, RANSAC).

    Caches the patched modules so subsequent calls return the same objects
    (re-loading would lose the monkey-patches).
    """
    global _cdm_cache
    if _cdm_cache is not None:
        return _cdm_cache

    CDM, _l2b = _load_cdm_modules()

    # 1. Use pdflatex with non-CJK template
    _l2b.formular_template = _PDFLATEX_FORMULA_TEMPLATE

    # 2. Replace xelatex → pdflatex, magick → convert (ImageMagick v6)
    _orig_run_cmd = _l2b.run_cmd

    def patched_run_cmd(cmd, timeout_sec=30, temp_dir=None):
        cmd = cmd.replace("xelatex ", "pdflatex ")
        cmd = cmd.replace("magick ", "convert ")
        return _orig_run_cmd(cmd, timeout_sec=timeout_sec, temp_dir=temp_dir)

    _l2b.run_cmd = patched_run_cmd

    # 3. Fix RANSAC: scikit-image >= 0.25 uses 'rng' not 'random_state'
    import skimage.measure as _skm
    _orig_ransac = _skm.ransac

    def _patched_ransac(*args, **kwargs):
        if "random_state" in kwargs:
            kwargs["rng"] = kwargs.pop("random_state")
        return _orig_ransac(*args, **kwargs)

    _skm.ransac = _patched_ransac
    _cdm_cache = (CDM, _l2b)
    return _cdm_cache


def compute_cdm(pred_latex: str, gt_latex: str, img_id: str = "0") -> dict[str, float]:
    """CDM score using official OmniDocBench CDM implementation.

    Requires pdflatex + ImageMagick installed.
    Returns dict with 'recall', 'precision', 'F1_score'.

    Uses pdflatex instead of xelatex to avoid CJK font dependency.
    For math-only formulas (UniMERNet), CJK support is not needed.
    Each call uses a fresh temp directory to avoid stale cached bbox files.
    """
    import shutil as _shutil
    import tempfile as _tempfile

    CDM, _ = _ensure_cdm_patches()

    # Normalize LaTeX (remove $$ delimiters, same as OmniDocBench code)
    gt_clean = gt_latex.lstrip("$$").rstrip("$$").strip()
    gt_clean = gt_clean.lstrip("$").rstrip("$").strip()
    pred_clean = pred_latex.split("```latex")[-1].split("```")[0]
    pred_clean = pred_clean.lstrip("$$").rstrip("$$").strip()
    pred_clean = pred_clean.lstrip("$").rstrip("$").strip()

    # Use a fresh temp directory per call to avoid latex2bbox_color caching
    tmp_dir = _tempfile.mkdtemp(prefix="cdm_")
    try:
        cdm = CDM(output_root=tmp_dir)
        return cdm.evaluate(gt_clean, pred_clean, img_id)
    finally:
        _shutil.rmtree(tmp_dir, ignore_errors=True)


# ─── KIE ANLS (NanoNets official) ─────────────────────────────────────

def compute_kie_anls(pred_fields: dict, gt_fields: dict) -> float:
    """Average Normalized Levenshtein Similarity for KIE.

    Matches NanoNets/docext official evaluation:
    - For each GT field, compute NLS = 1 - edit_distance / max(len(pred), len(gt))
    - Missing predictions treated as empty string (full mismatch)
    - No threshold — continuous score [0, 1]
    - Final score = simple average across all GT fields
    """
    if not gt_fields:
        return 1.0 if not pred_fields else 0.0

    scores = []
    for key, gt_val in gt_fields.items():
        gt_str = str(gt_val) if gt_val is not None else ""
        pred_val = pred_fields.get(key)
        pred_str = str(pred_val) if pred_val is not None else ""

        if not gt_str and not pred_str:
            scores.append(1.0)
        else:
            max_len = max(len(pred_str), len(gt_str))
            if max_len == 0:
                scores.append(1.0)
            else:
                scores.append(1.0 - Levenshtein.distance(pred_str, gt_str) / max_len)

    return sum(scores) / len(scores) if scores else 0.0


# ─── OCRBench VQA scoring (official MultimodalOCR) ────────────────────

def ocrbench_score(pred: str, answers: list[str], dataset_name: str = "") -> float:
    """Official OCRBench v1 scoring from MultimodalOCR repo.

    Pure substring match for all answer lengths (binary 0 or 1).
    Returns max score across all valid answers.

    Matches official example.py: `if answer in predict: score += 1`

    Special case (from example.py/blip2.py): HME100k (math expressions)
    does NOT lowercase, and removes all spaces before matching.
    """
    score = 0.0
    is_hme = dataset_name == "HME100k"

    # Handle numeric predictions (from vqa_metric.py)
    if isinstance(pred, (int, float)):
        pred = str(pred)

    for answer in answers:
        # Handle numeric answers (from vqa_metric.py)
        if isinstance(answer, (int, float)):
            answer = str(answer)

        if is_hme:
            # HME100k: no lowercasing, remove all spaces
            ans_clean = answer.strip().replace("\n", " ").replace(" ", "")
            pred_clean = pred.strip().replace("\n", " ").replace(" ", "")
            if ans_clean in pred_clean:
                score = max(score, 1.0)
        else:
            ans_clean = answer.lower().strip().replace("\n", " ")
            pred_clean = pred.lower().strip().replace("\n", " ")
            if ans_clean in pred_clean:
                score = max(score, 1.0)
    return score


# ─── LaTeX metrics ─────────────────────────────────────────────────────

def _normalize_latex(text: str) -> str:
    """Remove unnecessary whitespace from LaTeX code.

    Exact copy of UniMERNet/test.py normalize_text() (lines 72-87).
    Does NOT remove delimiters ($, $$, \\[...\\]) or style commands —
    those are NOT part of the original UniMERNet evaluation.
    """
    text_reg = r'(\\(operatorname|mathrm|text|mathbf)\s?\*? {.*?})'
    letter = '[a-zA-Z]'
    noletter = '[\\W_^\\d]'
    names = [x[0].replace(' ', '') for x in re.findall(text_reg, text)]
    text = re.sub(text_reg, lambda match: str(names.pop(0)), text)
    news = text
    while True:
        text = news
        news = re.sub(r'(?!\\ )(%s)\s+?(%s)' % (noletter, noletter), r'\1\2', text)
        news = re.sub(r'(?!\\ )(%s)\s+?(%s)' % (noletter, letter), r'\1\2', news)
        news = re.sub(r'(%s)\s+?(%s)' % (letter, noletter), r'\1\2', news)
        if news == text:
            break
    return text


def latex_exact_match(pred: str, gt: str) -> float:
    """Exact match after LaTeX normalization. Returns 0 or 1."""
    return 1.0 if _normalize_latex(pred) == _normalize_latex(gt) else 0.0


def latex_edit_distance(pred: str, gt: str) -> float:
    """Normalized edit distance on normalized LaTeX. Lower = better."""
    return normalized_edit_distance(_normalize_latex(pred), _normalize_latex(gt))

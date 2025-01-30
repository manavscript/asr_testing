from jiwer import wer
from indicnlp.tokenize import indic_tokenize
import unicodedata
from Levenshtein import distance as lev_distance

def normalize_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("।", ".")  # Replace Hindi danda with a period
    return text

def compute_hindi_wer(reference, hypothesis):
    reference = normalize_text(reference)
    hypothesis = normalize_text(hypothesis)
    ref_tokens = indic_tokenize.trivial_tokenize(reference, lang='hi')
    hyp_tokens = indic_tokenize.trivial_tokenize(hypothesis, lang='hi')
    return wer(" ".join(ref_tokens), " ".join(hyp_tokens))

def compute_hindi_cer(reference, hypothesis):
    reference = normalize_text(reference)
    hypothesis = normalize_text(hypothesis)
    return lev_distance(reference, hypothesis) / max(len(reference), 1)
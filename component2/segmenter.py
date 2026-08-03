import re

def split_sentences(text: str) -> list[str]:
    """
    Regex/heuristic sentence splitter (split on ./?/!, guard abbreviations/ellipses).
    """
    # Simple split that guards common abbreviations and ellipses.
    # We add a newline after punctuation that indicates end of sentence.
    text = re.sub(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', '\n', text)
    sentences = [s.strip() for s in text.split('\n') if s.strip()]
    if not sentences and text.strip():
        return [text.strip()]
    return sentences

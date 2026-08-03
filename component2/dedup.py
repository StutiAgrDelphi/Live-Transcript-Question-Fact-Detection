from dataclasses import dataclass
from collections import deque
import difflib
import re
from shared.schema import Flag

@dataclass
class _DedupRecord:
    normalized_text: str
    flag: Flag
    mention_count: int = 1

class DedupEngine:
    """
    Maintains a rolling list of recent _DedupRecords.
    Fuzzy matches new candidates to prevent duplicate flags.
    """
    def __init__(self, threshold: float = 0.82, max_records: int = 30):
        self.threshold = threshold
        self.records = deque(maxlen=max_records)
        
    def _normalize(self, text: str) -> str:
        text = text.lower()
        # strip punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # remove filler words
        fillers = r'\b(um|yeah|so|uh|like|you know)\b'
        text = re.sub(fillers, '', text)
        # normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def is_duplicate(self, flag: Flag) -> bool:
        norm_text = self._normalize(flag.resolved_text or flag.text)
        for record in self.records:
            similarity = difflib.SequenceMatcher(None, norm_text, record.normalized_text).ratio()
            if similarity >= self.threshold:
                record.mention_count += 1
                return True

        self.records.append(_DedupRecord(normalized_text=norm_text, flag=flag))
        return False

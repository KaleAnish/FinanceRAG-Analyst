import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class FakeFastTokenizer:
    """Word/punctuation tokenizer with HF-style call signature and offsets.

    Lets chunking tests run without downloading BGE. Adds 2 special tokens,
    like BERT's [CLS]/[SEP].
    """

    _TOKEN = re.compile(r"\w+|[^\w\s]")

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def __call__(self, text, add_special_tokens=True, truncation=False,
                 return_offsets_mapping=False):
        offsets = [(m.start(), m.end()) for m in self._TOKEN.finditer(text)]
        ids = list(range(len(offsets)))
        if add_special_tokens:
            ids = [0] + ids + [0]
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out

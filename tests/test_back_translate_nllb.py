import json
import tempfile
import unittest
from pathlib import Path

from scripts.back_translate_nllb import make_augmented_records, read_records


class BackTranslationHelpersTest(unittest.TestCase):
    def test_read_jsonl_and_keep_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text(
                '{"premise":"p","hypothesis":"h","label":1}\n\n',
                encoding="utf-8",
            )
            self.assertEqual(list(read_records(path)), [{"premise": "p", "hypothesis": "h", "label": 1}])

    def test_read_json_array(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.json"
            path.write_text(json.dumps([{"text": "hello"}]), encoding="utf-8")
            self.assertEqual(list(read_records(path)), [{"text": "hello"}])

    def test_make_augmented_records_preserves_label(self):
        records = [{"premise": "old p", "hypothesis": "old h", "label": 2}]
        augmented = make_augmented_records(
            records,
            {"premise": ["new p"], "hypothesis": ["new h"]},
            start_index=7,
            pivot_lang="fra_Latn",
            include_metadata=True,
        )
        self.assertEqual(augmented[0]["premise"], "new p")
        self.assertEqual(augmented[0]["hypothesis"], "new h")
        self.assertEqual(augmented[0]["label"], 2)
        self.assertEqual(augmented[0]["source_index"], 7)


if __name__ == "__main__":
    unittest.main()

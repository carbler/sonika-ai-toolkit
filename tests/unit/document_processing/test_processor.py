"""Unit tests for DocumentProcessor (document_processing.processor).

Text extraction for binary formats (PDF/DOCX/XLSX/PPTX) is dispatched to
optional libraries; here we test token counting, the plain-text path (real
tmp file), extension dispatch, and chunking — none of which need those extras.
"""

from unittest.mock import patch

import pytest

from sonika_ai_toolkit.document_processing.processor import DocumentProcessor


class TestCountTokens:
    def test_non_empty_text_has_positive_count(self):
        assert DocumentProcessor.count_tokens("hello world") > 0

    def test_empty_text_is_zero(self):
        assert DocumentProcessor.count_tokens("") == 0

    def test_fallback_approximation_on_encoding_error(self):
        with patch("tiktoken.encoding_for_model", side_effect=Exception("boom")):
            # 8 chars // 4 == 2
            assert DocumentProcessor.count_tokens("abcdefgh") == 2


class TestExtractTextFromTxt:
    def test_reads_utf8_file(self, tmp_path):
        f = tmp_path / "note.txt"
        f.write_text("café y té", encoding="utf-8")
        assert DocumentProcessor.extract_text_from_txt(str(f)) == "café y té"

    def test_strips_surrounding_whitespace(self, tmp_path):
        f = tmp_path / "note.txt"
        f.write_text("  padded  \n", encoding="utf-8")
        assert DocumentProcessor.extract_text_from_txt(str(f)) == "padded"


class TestExtractTextDispatch:
    def test_txt_extension_dispatches_to_txt_extractor(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("content", encoding="utf-8")
        assert DocumentProcessor.extract_text(str(f), "txt") == "content"

    def test_md_and_csv_route_through_txt(self, tmp_path):
        f = tmp_path / "doc.md"
        f.write_text("# title", encoding="utf-8")
        assert DocumentProcessor.extract_text(str(f), "md") == "# title"

    def test_unsupported_extension_raises(self):
        with pytest.raises(ValueError):
            DocumentProcessor.extract_text("x.xyz", "xyz")

    def test_dispatch_calls_matching_extractor(self):
        with patch.object(DocumentProcessor, "extract_text_from_pdf",
                          return_value="pdf-text") as mock_pdf:
            result = DocumentProcessor.extract_text("file.pdf", "pdf")
        mock_pdf.assert_called_once_with("file.pdf")
        assert result == "pdf-text"


class TestCreateChunks:
    def test_short_text_single_chunk(self):
        chunks = DocumentProcessor.create_chunks("Just one short sentence.")
        assert len(chunks) == 1
        assert chunks[0]["chunk_index"] == 0
        assert chunks[0]["content"]
        assert chunks[0]["token_count"] > 0

    def test_long_text_splits_into_multiple_chunks(self):
        text = ". ".join(f"sentence number {i} with some words" for i in range(200))
        chunks = DocumentProcessor.create_chunks(text, chunk_size=50, overlap=5)
        assert len(chunks) > 1
        # chunk_index is sequential starting at 0
        assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))

    def test_empty_text_yields_no_chunks(self):
        assert DocumentProcessor.create_chunks("") == []

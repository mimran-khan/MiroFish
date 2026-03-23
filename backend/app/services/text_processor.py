"""
Text processing helpers.
"""

import re
from typing import List, Optional

from ..utils.file_parser import FileParser, split_text_into_chunks
from ..utils.logger import get_logger

logger = get_logger('mirofish.text_processor')

_NUMERIC_FIELD_RE = re.compile(r'[\d.]+')
_DELIMITER_RE = re.compile(r'[\t|]')

_TABULAR_TO_NARRATIVE_PROMPT = """\
You are a financial data analyst. Convert the following structured/tabular data \
into a narrative analyst report written in natural prose paragraphs.

Rules:
- For each row or entity, describe it by name and list its key attributes in sentence form.
- Group related entities together (e.g. same sector, similar margin levels) and note patterns.
- Mention specific numeric values so that entity extraction can identify them.
- Do NOT use tables, bullet points, or markdown formatting — write full paragraphs only.
- Preserve ALL entity names and numeric values from the source data.
- Keep the report factual; do not add opinions or speculation.

Source data:
"""

_BATCH_SIZE_CHARS = 6000


class TextProcessor:
    """Extract, chunk, and normalize text for downstream pipelines."""
    
    @staticmethod
    def extract_from_files(file_paths: List[str]) -> str:
        """Concatenate text extracted from many files."""
        return FileParser.extract_from_multiple(file_paths)
    
    @staticmethod
    def split_text(
        text: str,
        chunk_size: int = 500,
        overlap: int = 50
    ) -> List[str]:
        """
        Split text into overlapping chunks.

        Args:
            text: Source text
            chunk_size: Chunk length
            overlap: Overlap between chunks

        Returns:
            Chunk strings
        """
        return split_text_into_chunks(text, chunk_size, overlap)
    
    @staticmethod
    def preprocess_text(text: str) -> str:
        """
        Normalize whitespace and newlines.

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        text = re.sub(r'\n{3,}', '\n\n', text)
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        return text.strip()
    
    @staticmethod
    def get_text_stats(text: str) -> dict:
        """Return simple length/word/line counts."""
        return {
            "total_chars": len(text),
            "total_lines": text.count('\n') + 1,
            "total_words": len(text.split()),
        }
    
    @staticmethod
    def is_tabular(text: str) -> bool:
        """
        Detect whether text is predominantly tabular/structured data.
        
        A line is considered tabular if it contains 3+ numeric-looking fields
        or uses common delimiters (tab, pipe, comma-heavy).
        Returns True when >40% of non-empty lines match.
        """
        lines = [ln for ln in text.split('\n') if ln.strip()]
        if len(lines) < 3:
            return False
        
        tabular_count = 0
        for line in lines:
            numeric_fields = _NUMERIC_FIELD_RE.findall(line)
            has_delimiters = bool(_DELIMITER_RE.search(line))
            if len(numeric_fields) >= 3 or (len(numeric_fields) >= 2 and has_delimiters):
                tabular_count += 1
        
        ratio = tabular_count / len(lines)
        logger.debug(f"Tabular detection: {tabular_count}/{len(lines)} lines matched ({ratio:.1%})")
        return ratio > 0.4
    
    @staticmethod
    def convert_tabular_to_narrative(text: str) -> str:
        """
        Use the LLM to convert tabular/structured data into narrative prose
        that Zep can properly extract entities from.
        
        Processes text in batches to handle large datasets.
        """
        from ..utils.llm_client import LLMClient
        
        llm = LLMClient()
        lines = text.split('\n')
        
        header_lines = []
        for line in lines[:5]:
            if line.strip() and not _NUMERIC_FIELD_RE.fullmatch(line.strip()):
                header_lines.append(line)
            else:
                break
        header = '\n'.join(header_lines) + '\n' if header_lines else ''
        
        data_lines = lines[len(header_lines):]
        
        batches = []
        current_batch: List[str] = []
        current_size = 0
        for line in data_lines:
            current_batch.append(line)
            current_size += len(line) + 1
            if current_size >= _BATCH_SIZE_CHARS:
                batches.append('\n'.join(current_batch))
                current_batch = []
                current_size = 0
        if current_batch:
            batches.append('\n'.join(current_batch))
        
        if not batches:
            return text
        
        logger.info(f"Converting tabular data to narrative: {len(batches)} batch(es), "
                     f"{len(lines)} total lines")
        
        narrative_parts = []
        for i, batch in enumerate(batches):
            batch_text = header + batch
            messages = [
                {"role": "system", "content": _TABULAR_TO_NARRATIVE_PROMPT},
                {"role": "user", "content": batch_text},
            ]
            result = llm.chat(messages=messages, temperature=0.3, max_tokens=4096)
            narrative_parts.append(result)
            logger.info(f"Batch {i + 1}/{len(batches)} converted: {len(result)} chars")
        
        return "\n\n".join(narrative_parts)

"""
Sentence Aggregator for LLM Streaming → TTS Pipeline

Buffers LLM text chunks and yields complete sentences for TTS processing.
Handles edge cases like missing punctuation, newlines, and long text.
"""
import re
from loguru import logger


class SentenceAggregator:
    """
    Aggregates streaming text chunks into complete sentences.

    Handles:
    - Sentence-ending punctuation (.!?)
    - Paragraph breaks (double newlines)
    - Line breaks (single newlines)
    - Long text without punctuation (max_chars fallback)
    """

    def __init__(self, min_chars: int = 20, max_chars: int = 200):
        """
        Initialize the aggregator.

        Args:
            min_chars: Minimum characters before checking for sentence boundaries
            max_chars: Force flush if no punctuation found (fallback)
        """
        self.buffer = ""
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.total_chars_processed = 0

    def add_chunk(self, text: str) -> list[str]:
        """
        Add a text chunk and return any complete sentences.

        Args:
            text: New text chunk from LLM stream

        Returns:
            List of complete sentences (may be empty if still buffering)
        """
        if not text:
            return []

        self.buffer += text
        self.total_chars_processed += len(text)
        sentences = []

        while len(self.buffer) >= self.min_chars:
            # Priority 1: Sentence-ending punctuation (.!?)
            # Match punctuation followed by whitespace or newline
            match = re.search(r'[.!?][\s\n]+', self.buffer)
            if match:
                end_idx = match.end()
                sentence = self.buffer[:end_idx].strip()
                if sentence:
                    sentences.append(sentence)
                    logger.debug(f"Sentence (punctuation): '{sentence[:50]}...'")
                self.buffer = self.buffer[end_idx:]
                continue

            # Priority 2: Paragraph breaks (double newlines)
            match = re.search(r'\n\n+', self.buffer)
            if match:
                end_idx = match.start()  # Don't include the newlines
                sentence = self.buffer[:end_idx].strip()
                if sentence:
                    sentences.append(sentence)
                    logger.debug(f"Sentence (paragraph): '{sentence[:50]}...'")
                self.buffer = self.buffer[match.end():]
                continue

            # Priority 3: Single newline (line break)
            if '\n' in self.buffer:
                idx = self.buffer.index('\n')
                if idx > 10:  # Don't break on very short lines
                    sentence = self.buffer[:idx].strip()
                    if sentence:
                        sentences.append(sentence)
                        logger.debug(f"Sentence (newline): '{sentence[:50]}...'")
                    self.buffer = self.buffer[idx + 1:]
                    continue

            # Priority 4: Max chars fallback (no punctuation found)
            if len(self.buffer) >= self.max_chars:
                # Try to break at comma first, then space
                break_idx = self.max_chars
                comma_idx = self.buffer.rfind(',', 0, self.max_chars)
                space_idx = self.buffer.rfind(' ', 0, self.max_chars)

                if comma_idx > self.max_chars // 2:
                    break_idx = comma_idx + 1
                elif space_idx > self.max_chars // 2:
                    break_idx = space_idx + 1

                sentence = self.buffer[:break_idx].strip()
                if sentence:
                    sentences.append(sentence)
                    logger.debug(f"Sentence (max_chars): '{sentence[:50]}...'")
                self.buffer = self.buffer[break_idx:]
                continue

            # Not enough text yet, wait for more
            break

        return sentences

    def flush(self) -> str | None:
        """
        Flush remaining buffer (call at end of stream).

        Returns:
            Remaining text, or None if buffer is empty
        """
        remaining = self.buffer.strip()
        self.buffer = ""
        if remaining:
            logger.debug(f"Flushed remaining: '{remaining[:50]}...'")
            return remaining
        return None

    def reset(self):
        """Reset the aggregator state."""
        self.buffer = ""
        self.total_chars_processed = 0

    @property
    def pending_chars(self) -> int:
        """Number of characters currently buffered."""
        return len(self.buffer)


# Quick test
if __name__ == "__main__":
    import sys
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")

    agg = SentenceAggregator(min_chars=10, max_chars=100)

    # Test 1: Normal sentences
    print("\n--- Test 1: Normal sentences ---")
    chunks = ["Hello, ", "this is ", "a test. ", "And another ", "sentence!"]
    for chunk in chunks:
        sentences = agg.add_chunk(chunk)
        for s in sentences:
            print(f"  -> '{s}'")
    remaining = agg.flush()
    if remaining:
        print(f"  -> (flush) '{remaining}'")

    agg.reset()

    # Test 2: Newlines
    print("\n--- Test 2: Newlines ---")
    chunks = ["First line\n", "Second line\n\n", "Third paragraph"]
    for chunk in chunks:
        sentences = agg.add_chunk(chunk)
        for s in sentences:
            print(f"  -> '{s}'")
    remaining = agg.flush()
    if remaining:
        print(f"  -> (flush) '{remaining}'")

    agg.reset()

    # Test 3: No punctuation (max_chars fallback)
    print("\n--- Test 3: No punctuation ---")
    long_text = "This is a very long piece of text without any punctuation " * 5
    sentences = agg.add_chunk(long_text)
    for s in sentences:
        print(f"  -> '{s[:60]}...'")
    remaining = agg.flush()
    if remaining:
        print(f"  -> (flush) '{remaining[:60]}...'")

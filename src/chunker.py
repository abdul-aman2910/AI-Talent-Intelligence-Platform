class TextChunker:
    """
    Splits text into overlapping chunks without splitting words.
    """

    def __init__(self, chunk_size=500, overlap=100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split_text(self, text):
        words = text.split()

        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:

            # Check whether adding the word exceeds chunk size
            if current_length + len(word) + 1 > self.chunk_size:

                if current_chunk:
                    chunks.append(" ".join(current_chunk))

                # Keep words from the end for overlap
                overlap_words = []
                overlap_length = 0

                for previous_word in reversed(current_chunk):
                    if overlap_length + len(previous_word) + 1 > self.overlap:
                        break

                    overlap_words.insert(0, previous_word)
                    overlap_length += len(previous_word) + 1

                current_chunk = overlap_words + [word]
                current_length = sum(len(w) + 1 for w in current_chunk)

            else:
                current_chunk.append(word)
                current_length += len(word) + 1

        # Add remaining words
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks
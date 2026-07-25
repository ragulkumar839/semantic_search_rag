import os


def chunk_text(pages, source, max_length=600, overlap=100):
    """
    Recursive overlapping semantic chunking to keep structural sections intact while
    providing boundary overlap between adjacent chunks.
    """
    chunks = []
    filename = os.path.basename(source)

    for page in pages:
        page_number = page["page"]
        text = page["text"]

        # Split by double newline first (paragraphs/sections)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        current_chunk = ""

        for para in paragraphs:
            # If paragraph fits in current chunk buffer
            if len(current_chunk) + len(para) + 2 <= max_length:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
            else:
                # Save existing current_chunk if not empty
                if current_chunk and len(current_chunk) >= 40:
                    chunks.append({
                        "chunk": current_chunk.strip(),
                        "page": page_number,
                        "source": filename,
                        "filename": filename
                    })
                    # Overlap: keep tail of current_chunk for next chunk context
                    tail = current_chunk[-overlap:] if len(current_chunk) > overlap else ""
                    current_chunk = tail + "\n\n" + para if tail else para
                else:
                    current_chunk = para

                # If paragraph itself exceeds max_length, split words with overlap
                if len(current_chunk) > max_length:
                    words = current_chunk.split()
                    sub_chunk = ""
                    for word in words:
                        if len(sub_chunk) + len(word) + 1 <= max_length:
                            sub_chunk += (" " if sub_chunk else "") + word
                        else:
                            if len(sub_chunk) >= 40:
                                chunks.append({
                                    "chunk": sub_chunk.strip(),
                                    "page": page_number,
                                    "source": filename,
                                    "filename": filename
                                })
                            # Overlap tail
                            word_tail = sub_chunk[-overlap:] if len(sub_chunk) > overlap else ""
                            sub_chunk = word_tail + " " + word

                    current_chunk = sub_chunk

        if current_chunk and len(current_chunk) >= 40:
            chunks.append({
                "chunk": current_chunk.strip(),
                "page": page_number,
                "source": filename,
                "filename": filename
            })

    return chunks
import json
import uuid
from collections import namedtuple
from rds import db

from tqdm import tqdm
from llama_index.core.node_parser.text.sentence import SentenceSplitter

chunker = SentenceSplitter(chunk_size=512)
sentence_window_map = {}
sentence_window = namedtuple("sentence_window", ["prev", "next"])


if __name__ == "__main__":
    """
    This script is used to chunk pages from the EXTRACTED_PDF_PAGE table,
    then contruct a mapping of chunks to their respective sentence windows
    (prev, next) UUIDs and store them in a JSON file, while the chunks and
    their respective UUIDs are stored in chunked_512_recursive_dhanush.
    """
    with db.get_cursor() as cur:
        # Query the EXTRACTED_PDF_PAGE table to get the pages and their ids
        cur.execute(
            """SELECT id, extracted_page FROM "EXTRACTED_PDF_PAGE" WHERE document_type = 'AR'"""
        )
        # note that we are doing the chunking here by pages and not on the entire document
        pages = cur.fetchall()

        insert_query_512 = """
        INSERT INTO "chunked_512_recursive_dhanush" (
            pdf_page_id,
            chunk_id,
            chunk_text
        ) VALUES (%s, %s, %s)
        """

        for page_id, extracted_text in tqdm(pages):
            page_chunks = chunker.split_text(extracted_text)

            uuids = [uuid.uuid5(uuid.NAMESPACE_DNS, text) for text in page_chunks]

            values_to_insert = [
                (page_id, str(uuids[idx]), chunk)
                for idx, chunk in enumerate(page_chunks)
            ]
            cur.executemany(insert_query_512, values_to_insert)

            for idx, chunk in enumerate(page_chunks):
                curr_uuid = uuids[idx]
                curr_chunk_window = sentence_window(
                    str(uuids[idx - 1]) if idx > 0 else None,
                    str(uuids[idx + 1]) if idx < len(page_chunks) - 1 else None,
                )
                sentence_window_map[str(uuids[idx])] = curr_chunk_window

        print("New rows inserted successfully.")

        with open("src/swr/sentence_window_map.json", "w") as f:
            json.dump(sentence_window_map, f, indent=4)

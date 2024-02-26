import uuid
from rds import db
from tqdm import tqdm
from llama_index.core.node_parser.text.sentence import SentenceSplitter


class Node:
    def __init__(self, pdf_document_name, text, chunk_size, parent=None):
        self.id = uuid.uuid4()
        self.pdf_document_name = pdf_document_name
        self.text = text
        self.chunk_size = chunk_size
        self.parent = parent
        self.children = []

    def add_child(self, child_node):
        self.children.append(child_node)
        child_node.parent = self


def chunk_text(pdf_document_name, text, chunk_sizes):
    def _chunk_text_recursive(text, chunk_sizes, parent_node=None):
        if not chunk_sizes:
            return None
        chunk_size = chunk_sizes[0]
        chunker = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=20)
        chunks = chunker.split_text(text)
        for chunk in chunks:
            node = Node(pdf_document_name, chunk, chunk_size, parent=parent_node)
            if parent_node:
                parent_node.add_child(node)
            _chunk_text_recursive(chunk, chunk_sizes[1:], parent_node=node)

    dummy_root = Node(pdf_document_name, text, -1)
    _chunk_text_recursive(text, chunk_sizes, parent_node=dummy_root)
    return dummy_root


def get_all_nodes(root):
    nodes = []
    nodes_to_visit = [root]
    while nodes_to_visit:
        current_node = nodes_to_visit.pop()
        nodes.append(current_node)
        nodes_to_visit.extend(current_node.children)
    return nodes


def visit_and_insert_rds(root, cursor):
    nodes = get_all_nodes(root)

    insert_query_amr = """
    INSERT INTO "amr_nodes" (
        chunk_id,
        parent_chunk_id,
        pdf_document_name,
        chunk_text,
        chunk_size
    ) VALUES (%s,%s,%s,%s,%s)
    """
    # Filter out the dummy root node before preparing the data to be inserted
    filtered_nodes = [node for node in nodes if node.chunk_size != -1]

    # prepare the data to be inserted
    nodes_to_insert = [
        (
            str(node.id),
            str(node.parent.id) if node.parent else None,
            node.pdf_document_name,
            node.text,
            node.chunk_size,
        )
        for node in filtered_nodes
    ]

    cursor.executemany(insert_query_amr, nodes_to_insert)


if __name__ == "__main__":
    """
    This script is used to chunk pages from the EXTRACTED_PDF_PAGE table,
    into 2048, 512, and 128 chunks recursively. While chunking, we also generate
    UUIDs and store them in nodes. We store the chunks in RDS.
    """
    with db.get_cursor() as cur:
        cur.execute(
            """SELECT pdf_document_name, concatenated_pages FROM "EXTRACTED_PDF" WHERE document_type = 'AR'"""
        )
        pages = cur.fetchall()

        # need to temporarily remove the foreign key constraint to the amr_nodes table
        cur.execute(
            """
            ALTER TABLE "amr_nodes"
            DROP CONSTRAINT amr_nodes_parent_chunk_id_fkey;
            """
        )

        for pdf_document_name, concatenated_pages in tqdm(pages):
            root_node = chunk_text(
                pdf_document_name, concatenated_pages, [2048, 512, 128]
            )
            # ignore the root node because it contains the entire document
            visit_and_insert_rds(
                root_node,
                cur,
            )

        # need to add the foreign key constraint to the amr_nodes table
        cur.execute(
            """
            ALTER TABLE "amr_nodes"
            ADD CONSTRAINT amr_nodes_parent_chunk_id_fkey FOREIGN KEY (parent_chunk_id)
            REFERENCES amr_nodes(chunk_id);
            """
        )

        print("New rows inserted successfully.")

import unittest
from typing import Optional
from DuRAG.rds import db


class TestChunkingHierarchy(unittest.TestCase):
    def setUp(self):
        # Connect to your database
        self.cursor = db.get_cursor()

    def test_chunking_hierarchy(self):
        query = """
        SELECT *
        FROM amr_nodes
        ORDER BY chunk_size DESC;
        """
        with self.cursor as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
        print(f"fetched rows {len(rows)=}")

        def get_children(parent_id: Optional[str]) -> list:
            """
            Get the child nodes of a parent node based on the parent_chunk_id.

            Args:
                parent_id: The chunk_id of the parent node.

            Returns:
                A list of child nodes.
            """
            return [row for row in rows if row[1] == parent_id]

        def concat_child_texts(children: list) -> str:
            """
            Concatenate the chunk_text of all child nodes.

            Args:
                children: A list of child nodes.

            Returns:
                The concatenated chunk_text of all child nodes.
            """
            return "".join(child[3] for child in children)

        def verify_hierarchy(parent_id: Optional[str]) -> bool:
            """
            Recursively verify the chunking hierarchy for a parent node.

            Args:
                parent_id: The chunk_id of the parent node.

            Returns:
                True if the chunking hierarchy is correct, False otherwise.
            """
            children = get_children(parent_id)
            print(f"{len(children)=}")
            if not children:
                return True

            parent_text = next((row[3] for row in rows if row[0] == parent_id), "")
            child_texts = concat_child_texts(children)

            if parent_text != child_texts:
                self.fail(
                    f"Chunking hierarchy error for parent node {parent_id}:\n"
                    f"Parent text: {parent_text}\n"
                    f"Child texts concatenated: {child_texts}"
                )

            return all(verify_hierarchy(child[0]) for child in children)

        # Start the recursive verification from the root node (parent_chunk_id is None)
        root_nodes = [row[0] for row in rows if row[4] == 2048]
        for root_node in root_nodes:
            self.assertTrue(verify_hierarchy(root_node))


if __name__ == "__main__":
    unittest.main()

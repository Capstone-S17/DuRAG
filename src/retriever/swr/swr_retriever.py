import json
from DuRAG.retriever.retriever import Retriever
import pkg_resources


class SentenceWindow:
    def __init__(
        self,
        left_uuid,
        left_content,
        center_uuid,
        center_content,
        right_uuid,
        right_content,
        pdf_name,
    ):
        self.left_uuid = left_uuid
        self.left_content = left_content
        self.center_uuid = center_uuid
        self.center_content = center_content
        self.right_uuid = right_uuid
        self.right_content = right_content
        self.pdf_name = pdf_name

    def __str__(self):
        return f"SentenceWindow(\n  PDF: {self.pdf_name}\n  Left: [{self.left_uuid}] {self.left_content}\n  Center: [{self.center_uuid}] {self.center_content}\n  Right: [{self.right_uuid}] {self.right_content}\n)"

    def __repr__(self):
        return self.__str__()

    def joined_text(self) -> str:
        # Join the content from left, center, and right while handling None values
        contents = [self.left_content, self.center_content, self.right_content]
        return " ".join(content for content in contents if content)


class SentenceWindowRetriever(Retriever):
    """
    Class to retrieve 128 token chunks from Weaviate then adds the before and after chunks.
    This class does not handle the reranking of the chunks but can help to format for the reranker.
    Since the base chunk size is 128 tokens, the maximum window size is 384 tokens which is within
    the 512 token limit of the Reranker.

    Example usage:
        response = self.semantic_search(query, filters) # or full_text  or hybrid, comes from Retriever Base Class
        sentence_windows = self.get_sentence_windows(response.objects) # to find the sentence windows
    """

    def __init__(
        self,
        weaviate_client,
    ):
        super().__init__(weaviate_client, "SWR_chunks")
        json_file = pkg_resources.resource_filename(
            "DuRAG", "retriever/swr/sentence_window_map.json"
        )
        self.sentence_window_map = self._load_sentence_window_map(json_file)

    def _load_sentence_window_map(self, path) -> dict:
        with open(path, "r") as f:
            return json.load(f)

    def retrieve_by_uuid(self, uuid):
        return self.collection.query.fetch_object_by_id(uuid)

    def get_sentence_windows(self, response_objects) -> list[SentenceWindow]:
        sentence_windows = []
        for o in response_objects:
            left_uuid, left_content, right_uuid, right_content = (
                None,
                None,
                None,
                None,
            )
            retrieved_uuid = o.uuid
            fetch_window = self.sentence_window_map.get(str(retrieved_uuid))
            if fetch_window:
                left, right = fetch_window
                if left is not None:
                    left_data = self.retrieve_by_uuid(left)
                    left_uuid = left_data.uuid
                    left_content = left_data.properties["content"]
                center_uuid = o.uuid
                center_content = o.properties["content"]
                if right is not None:
                    right_data = self.retrieve_by_uuid(right)
                    right_uuid = right_data.uuid
                    right_content = right_data.properties["content"]
            else:
                # If there's no mapping found, just use the original chunk
                center_uuid = o.uuid
                center_content = o.properties["content"]
                # this should never happen lol
            pdf_name = o.properties["pdf_name"]
            window = SentenceWindow(
                left_uuid,
                left_content,
                center_uuid,
                center_content,
                right_uuid,
                right_content,
                pdf_name,
            )
            sentence_windows.append(window)
        return sentence_windows

    @classmethod
    def get_rerank_format(cls, query: str, sentence_windows: list[SentenceWindow]):
        rerank_data = []
        for window in sentence_windows:
            # Create a tuple for reranking format using the UUID of the center chunk
            rerank_tuple = (
                window.center_uuid,
                query,
                window.joined_text(),
                window.pdf_name,
            )
            rerank_data.append(rerank_tuple)
        return rerank_data

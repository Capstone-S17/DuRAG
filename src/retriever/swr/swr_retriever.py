import json
from retriever.retriever import Retriever


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
        sentence_window_map_path="src/swr/sentence_window_map.json",
    ):
        super().__init__(weaviate_client, "SWR_chunks")
        self.sentence_window_map = self._load_sentence_window_map(
            sentence_window_map_path
        )

    def _load_sentence_window_map(self, path) -> dict:
        with open(path, "r") as f:
            return json.load(f)

    def retrieve_by_uuid(self, uuid):
        return self.collection.query.fetch_object_by_id(uuid)

    def get_sentence_windows(self, response_objects) -> list[list[tuple[str, str]]]:
        sentence_window_included = []
        for o in response_objects:
            window_data = []
            retrieved_uuid = o.uuid
            fetch_window = self.sentence_window_map.get(str(retrieved_uuid))
            if fetch_window:
                left, right = fetch_window
                # Fetch and append the left chunk if it exists
                if left is not None:
                    left_data = self.retrieve_by_uuid(left)
                    window_data.append(
                        (left_data.uuid, left_data.properties["content"])
                    )
                # Append the original chunk
                window_data.append((o.uuid, o.properties["content"]))
                # Fetch and append the right chunk if it exists
                if right is not None:
                    right_data = self.retrieve_by_uuid(right)
                    window_data.append(
                        (right_data.uuid, right_data.properties["content"])
                    )
            else:
                # If there's no mapping found, just append the original chunk
                window_data.append((o.uuid, o.properties["content"]))
                # this should never happen lol
            sentence_window_included.append(window_data)
        return sentence_window_included

    @classmethod
    def window_text_joiner(cls, window: list[tuple[str, str]]) -> str:
        return " ".join(i[1] for i in window)

    @classmethod
    def get_rerank_format(
        cls, query: str, sentence_windows: list[list[tuple[str, str]]]
    ):
        return [[query, cls.window_text_joiner(window)] for window in sentence_windows]

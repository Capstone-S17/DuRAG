import weaviate.classes as wvc
import json
from typing import Optional


class SentenceWindowRetriever:
    def __init__(
        self,
        weaviate_client,
        sentence_window_map_path="src/swr/sentence_window_map.json",
    ):
        self.client = weaviate_client
        self.sentence_window_map = self._load_sentence_window_map(
            sentence_window_map_path
        )
        self.collection = self.client.collections.get("SWR_chunks")

    def _load_sentence_window_map(self, path):
        with open(path, "r") as f:
            return json.load(f)

    def query_collection(self, query: str, filters: Optional[str] = None, limit=10):
        filter_param = (
            wvc.query.Filter.by_property("content").like(filters) if filters else None
        )
        return self.collection.query.near_text(
            query=query,
            # include_vector=True,
            filters=filter_param,
            limit=limit,
        )

    def retrieve_by_uuid(self, uuid):
        return self.collection.query.fetch_object_by_id(uuid)

    def get_sentence_windows(self, response_objects):
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

    def window_text_joiner(self, window):
        return " ".join(i[1] for i in window)

    def get_swr_outputs(self, query, filters=None):
        response = self.query_collection(query, filters)
        sentence_windows = self.get_sentence_windows(response.objects)
        return sentence_windows

    def get_rerank_format(self, query, filters=None):
        self.sentence_windows = self.get_swr_outputs(query, filters)
        pairs = [
            [query, self.window_text_joiner(window)] for window in self.sentence_windows
        ]
        return pairs

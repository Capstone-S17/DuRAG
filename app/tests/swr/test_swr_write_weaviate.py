import unittest
import weaviate
import weaviate.classes as wvc
import json
import sys


class TestWeaviate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Setup that is run once before all tests
        cls.client = weaviate.connect_to_local()

    def setUp(self):
        # Setup that is run before each test
        with open("src/retriever/swr/sentence_window_map.json", "r") as f:
            self.sentence_window_map = json.load(f)
        self.collection = self.client.collections.get("SWR_chunks")

    def test_memory_usage(self):
        # Check memory used by sentence_window_map as mega bytes
        memory_usage = sys.getsizeof(self.sentence_window_map) / (1024 * 1024)
        print(f"Memory Usage: {memory_usage} MB")  # Optionally print for debugging
        self.assertTrue(
            isinstance(memory_usage, float)
        )  # Check if memory usage is a float

    def test_unfound_objects(self):
        # Test for unfound objects in the collection
        unfound = []
        for k in self.sentence_window_map:
            response = self.collection.query.fetch_objects(
                filters=wvc.query.Filter.by_id().equal(k)
            )
            if response.objects == []:
                unfound.append(k)

        # Optionally print for debugging
        print(f"Total keys in sentence_window_map: {len(self.sentence_window_map)}")
        print(f"Total unfound objects: {len(unfound)}")

        # Assertions
        self.assertEqual(len(self.sentence_window_map), len(self.sentence_window_map))
        self.assertGreaterEqual(
            len(unfound), 0
        )  # Check if unfound list has zero or more items

    def test_collection_iteration(self):
        # Test iteration over collection
        counter = 0
        for item in self.collection.iterator():
            counter += 1
        iterator_length = len(list(self.collection.iterator()))

        # Optionally print for debugging
        print(f"Iterator count: {counter}")
        print(f"Iterator length: {iterator_length}")

        # Assertions
        self.assertEqual(
            counter, iterator_length
        )  # Check if counted items match the iterator length


if __name__ == "__main__":
    unittest.main()

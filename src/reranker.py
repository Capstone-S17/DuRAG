from FlagEmbedding import FlagReranker


class Reranker:
    def __init__(
        self, model_name: str = "BAAI/bge-reranker-large", use_fp16: bool = True
    ) -> None:
        """
        Initializes the reranker with the specified model.

        Args:
            model_name: The name of the model to be used for reranking.
            use_fp16: Flag indicating whether to use 16-bit floating-point precision (FP16).
                      This can enable faster computation at the cost of some precision.

        The Reranker class utilizes a pre-trained model to rerank a list of input pairs (query, chunk).
        """
        # Initialize the FlagReranker with the given model and precision setting.
        self.reranker = FlagReranker(model_name, use_fp16=use_fp16)

    def rerank_top_k(
        self, pairs: list[tuple[str, str]], k: int = 5
    ) -> list[tuple[tuple[str, str], float]]:
        """
        Reranks a list of pairs and returns the top `k` pairs based on the scores from the reranker model.

        Args:
            pairs: A list of pairs (query, chunk) to be reranked.
            k: The number of top pairs to return after reranking.

        Returns:
            A list of the top `k` reranked pairs, each with associated scores.

        The method computes scores for each pair using the reranker model and sorts the list
        in descending order based on these scores. It then returns the top `k` pairs from this sorted list.
        """
        scores = self.reranker.compute_score(pairs)

        scored_pairs = list(zip(pairs, scores))

        scored_pairs.sort(key=lambda x: x[1], reverse=True)

        top_k_pairs = scored_pairs[:k]

        return top_k_pairs

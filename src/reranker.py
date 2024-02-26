from FlagEmbedding import FlagReranker


class Reranker:
    def __init__(self, model_name="BAAI/bge-reranker-large", use_fp16=True):
        """
        Initializes the reranker with the given model.

        Args:
            model_name (str): The name of the model to be used for reranking.
            use_fp16 (bool): Whether to use 16-bit floating-point arithmetic
                             for faster computation with a slight degradation in performance.
        """
        self.reranker = FlagReranker(model_name, use_fp16=use_fp16)

    def rerank_top_k(self, pairs, k=5):
        """
        Reranks a list of pairs and returns the top k pairs based on the scores.

        Args:
            pairs (list): A list of pairs to be reranked.
            k (int): The number of top pairs to be returned.

        Returns:
            list: The top k reranked pairs based on their scores.
        """
        scores = self.reranker.compute_score(pairs)
        # sort the pairs based on the scores in descendin order
        scored_pairs = list(zip(pairs, scores))
        scored_pairs = sorted(scored_pairs, key=lambda x: x[1], reverse=True)
        # return the chunks only
        scores = [s[0][1] for s in scored_pairs]
        return scores[:k]

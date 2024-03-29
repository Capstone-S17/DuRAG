from DuRAG.retriever.RetrievalObject import RetrievalObject
from FlagEmbedding import FlagReranker
from DuRAG.logger import logger


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
        logger.info(f"Initializing reranker with model: {model_name}")
        self.reranker = FlagReranker(model_name, use_fp16=use_fp16)
        logger.info("Reranker initialized successfully")

    def rerank_top_k(
        self, retrieval_objects: list[RetrievalObject], k: int = 5
    ) -> list[RetrievalObject]:
        """
        Reranks the top k retrieval_objects based on the computed scores.

        Args:
            retrieval_objects: A list of RetrievalObject instances to be reranked.
            k. The number of top objects to be retures. Defaults to 5.

        Returns:
            list[RetrievalObject]: The top k reranked RetrievalObject instances.
        """
        if not retrieval_objects:
            logger.critical("No retrieval objects provided, returning an empty list.")
            return []

        if len(retrieval_objects) == 1:
            logger.warning("Only one retrieval object provided, skipping reranking.")
            return retrieval_objects

        logger.debug(f"Reranker received {len(retrieval_objects)=}")
        pairs = [(obj.query, obj.chunk) for obj in retrieval_objects]

        try:
            scores = self.reranker.compute_score(pairs)
            logger.debug(f"{scores=}")
        except ValueError as e:
            logger.error(f"Error during reranking: {e}")
            logger.error("Returning the original retrieval objects.")
            return retrieval_objects

        for obj, score in zip(retrieval_objects, scores):
            obj.score = score

        retrieval_objects.sort(key=lambda x: x.score, reverse=True)
        return retrieval_objects[:k]

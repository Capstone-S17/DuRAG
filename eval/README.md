## Evaluation result

## Eval set medium
| hybrid search alpha | Retrieval method | Answer relevance | Context relevance | Groundness | Answer correctness    |
| ------------------- | ------- | ---------------- | ----------------- | ------------ | --------------------- |
| 0.25                | SWR     | 0.852   |   0.836    |        0.662      |0.74
| 0.25                | AMR     |      0.86   |    0.85      | 0.672            | 0.76
| 0.25                 | SWR+AMR     |   0.85    |     **0.898**             | **0.717**          | **0.80**
| 0.5                 | SWR     |      0.866      |    0.884     | 0.716      |0.78
| 0.5                 | AMR     |   0.864      |  0.872               | 0.638             |**0.80**
| 0.5                 | SWR+AMR     |   **0.87**    |     0.894             | 0.714          | **0.80**
| 0.75                | SWR     |     0.826      |    0.814     | 0.712      |0.77
| 0.75                | AMR     |      0.826      |  0.839               | 0.638        |0.77
| 0.75                 | SWR+AMR     |   **0.87**    |     0.894             | 0.714          | **0.80**


## Eval set hard
| hybrid search alpha | Retrieval method | Answer relevance | Context relevance | Groundness | Answer correctness    |
| ------------------- | ------- | ---------------- | ----------------- | ------------ | --------------------- |
| 0.25                | SWR     |   **0.930**   |  0.842   |   0.670       |**0.660**
| 0.25                | AMR     |   0.860   | 0.824      |  0.632        | 0.620
| 0.25                 | SWR+AMR |   0.854  |     0.868      |  0.722       |0.620
| 0.5                 | SWR     |   0.862  | 0.83     |    0.754          |0.60
| 0.5                 | AMR     |   0.798   |  0.868      |0.594         |0.580
| 0.5                 | SWR+AMR |   0.824  |     **0.870**      |  **0.758**       |0.620
| 0.75                | SWR     |  0.901   |  0.811   |   0.690       |  0.590
| 0.75                | AMR     |    0.820   | 0.814      |  0.633        | 0.620
| 0.75                 | SWR+AMR |   0.834  |     0.86      |  0.750       |0.610

- hybrid search fusion algorithm [ranked, relative] (try only on the best model)

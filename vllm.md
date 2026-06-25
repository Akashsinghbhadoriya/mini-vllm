# VLLM

1. During llm inference each tokens are generated one by one which is also known as Autoregressive Generation and during each token generation all the previous tokens are utilized for generating the next token.
2. Attention is the bottleneck for generation of each tokens so we use KV cache for storing K and V values of the previous tokens in a cache and reuse them later for further generation. So if we have multiple users the KV cache explodes exponentially so vllm manages this efficiently.
3. Prefill Phase -> Suppose the user sends the prompt so we do prompt processing the tokenization and the key value cache creation is handled over here this phase is also known as prefill phase.
4. Decode Phase -> Now when the KV Cache exists we do the generation process of the prompts by generating one token at a time this phase is called decoding phase.

### Prefill vs Decode

| Property        | Prefill       | Decode       |
| --------------- | ------------- | ------------ |
| Input size      | Large         | Small        |
| Parallelism     | High          | Low          |
| GPU utilization | High          | Lower        |
| KV cache        | Created       | Reused       |
| Cost            | Compute-heavy | Memory-heavy |

Almost all vllm optimizations revolves around balancing these two phases.
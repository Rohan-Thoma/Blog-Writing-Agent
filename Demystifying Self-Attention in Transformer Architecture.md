# Demystifying Self-Attention in Transformer Architecture

## Introduction to Transformer Architecture and Role of Self-Attention

The Transformer architecture revolutionized natural language processing by introducing a fundamentally different approach to sequence modeling. At its core, the Transformer consists of two primary components: an encoder and a decoder. The encoder is composed of a stack of identical layers, each containing a multi-head self-attention mechanism followed by position-wise feed-forward networks. Similarly, the decoder also comprises stacked layers that include masked self-attention, encoder-decoder attention, and feed-forward networks. This modular design allows the model to progressively encode contextual information and generate outputs in an autoregressive manner.

Self-attention, the key innovation in Transformers, enables each token within a sequence to attend to every other token, regardless of their positional distance. Unlike traditional sequence models such as Recurrent Neural Networks (RNNs) or Convolutional Neural Networks (CNNs), which process tokens sequentially or within limited context windows, self-attention provides a direct path for interaction between any two positions in the sequence. RNNs inherently impose temporal dependencies causing slower training and issues with long-range context due to vanishing gradients, while CNNs rely on stacking layers and fixed receptive fields, limiting their effective context range.

The advantage of self-attention lies in its ability to capture long-range dependencies efficiently. By computing attention scores that weigh the relevance of all tokens to a given token, self-attention builds representations that integrate contextual cues across the entire input sequence in each layer. This global view is essential for modeling complex linguistic phenomena like co-reference or syntax, where dependencies can span arbitrary distances.

Moreover, self-attention facilitates significant computational improvements. Since attention operations do not require sequential token processing, Transformers can fully leverage parallel hardware architectures such as GPUs or TPUs during training and inference. This parallelization reduces training time drastically compared to RNNs. Additionally, the direct connections formed by self-attention improve gradient flow, mitigating vanishing and exploding gradients that traditionally hinder deep sequence models.

The impact of introducing self-attention through the Transformer architecture has been profound. It paved the way for major advances such as BERT, GPT, and numerous state-of-the-art NLP models that excel in tasks involving language understanding, generation, and translation. Self-attention’s flexibility and scalability continue to drive innovation across diverse applications in NLP and beyond, establishing it as a foundational technique for modern deep learning sequence modeling.

> **[IMAGE GENERATION FAILED]** High-level architecture of the Transformer illustrating encoder layers with multi-head self-attention and feed-forward networks.
>
> **Alt:** Diagram of Transformer architecture showing encoder and decoder stacks and the role of self-attention in each layer
>
> **Prompt:** A clear, technical diagram showing the Transformer architecture comprising stacked encoder and decoder layers. Each encoder layer contains multi-head self-attention and feed-forward neural networks. Arrows show token flows and positional embeddings, highlighting how each token attends to all others via self-attention. Labels identify encoder, decoder, self-attention, and feed-forward components.
>
> **Error:** 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. \n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_input_token_count, limit: 0, model: gemini-2.5-flash-preview-image\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-2.5-flash-preview-image\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-2.5-flash-preview-image\nPlease retry in 31.918534171s.', 'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': 'type.googleapis.com/google.rpc.Help', 'links': [{'description': 'Learn more about Gemini API quotas', 'url': 'https://ai.google.dev/gemini-api/docs/rate-limits'}]}, {'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_input_token_count', 'quotaId': 'GenerateContentInputTokensPerModelPerMinute-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-2.5-flash-preview-image'}}, {'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerMinutePerProjectPerModel-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-2.5-flash-preview-image'}}, {'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-2.5-flash-preview-image'}}]}, {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '31s'}]}}


## Mechanics of Self-Attention: Query, Key, and Value Vectors

In the Transformer architecture, the self-attention mechanism revolves around three key components: the Query (Q), Key (K), and Value (V) vectors. Each input token embedding is linearly projected into these three vectors, which enable the model to compute attention weights and generate context-aware representations.

**Query, Key, and Value Vectors and Their Dimensions**

Given an input sequence represented as a tensor of shape `(batch_size, seq_length, embedding_dim)`, we apply three learned linear transformations to produce Q, K, and V:

- **Query (Q):** captures the current token’s representation for which we want to compute attention.
- **Key (K):** represents the content of each token to compare against the Query.
- **Value (V):** contains the information to aggregate based on attention weights.

If `d_k` is the dimensionality of the Query and Key vectors, and `d_v` is that of the Value vectors, the transformations result in tensors:
- `Q`: `(batch_size, seq_length, d_k)`
- `K`: `(batch_size, seq_length, d_k)`
- `V`: `(batch_size, seq_length, d_v)`

Typically, `d_k` and `d_v` are equal and set to `embedding_dim / num_heads` in multi-head attention to keep computations efficient.

**Scaled Dot-Product Attention Scores**

To measure how much focus each token should give to others, compute raw attention scores by the dot product of Q with K:

```python
scores = torch.matmul(Q, K.transpose(-2, -1))  # shape: (batch_size, seq_length, seq_length)
```

Each element `scores[i, j]` indicates the similarity between the query vector of the ith token and the key vector of the jth token.

To avoid large magnitude dot products which can lead to small gradient signals, these scores are scaled by the square root of `d_k`:

```python
scaled_scores = scores / math.sqrt(d_k)
```

**Softmax for Normalizing Attention Weights**

The raw scores are converted into a probability distribution using the softmax function along the last dimension:

```python
attention_weights = torch.softmax(scaled_scores, dim=-1)
```

This normalization ensures that attention weights sum to 1 for each query token, making it easier to interpret them as distributional weights over all tokens.

**Weighted Sum of Value Vectors**

The attention weights then serve to weight the Value vectors, generating a context-aware representation for each token:

```python
context = torch.matmul(attention_weights, V)  # shape: (batch_size, seq_length, d_v)
```

This weighted sum aggregates information from relevant tokens, modulated by their computed importance.

**Role of Scaling for Gradient Stability**

Scaling by `1 / sqrt(d_k)` prevents the dot products from growing too large in magnitude. Large values can push the softmax into regions with extremely small gradients, which slows down or destabilizes training. The scaling keeps values in a range conducive to effective gradient flow and model convergence.

**Tensor Shapes and Batch Processing**

In practice, self-attention is computed over batches and multiple attention heads:

- Input tensor shape: `(batch_size, seq_length, embedding_dim)`
- After linear projections: Q, K, V have shapes `(batch_size, num_heads, seq_length, head_dim)`
- Attention scores shape: `(batch_size, num_heads, seq_length, seq_length)`
- Attention weights: same shape as scores
- Output context: `(batch_size, num_heads, seq_length, head_dim)`

Finally, multi-head outputs are concatenated and linearly transformed back to `(batch_size, seq_length, embedding_dim)` for subsequent layers.

### Minimal PyTorch Example

```python
import torch
import math

batch_size, seq_length, embedding_dim = 2, 5, 64
num_heads = 8
head_dim = embedding_dim // num_heads

# Random input embeddings
x = torch.rand(batch_size, seq_length, embedding_dim)

# Linear projections for Q, K, V (weights simulated as random for example)
W_q = torch.rand(embedding_dim, embedding_dim)
W_k = torch.rand(embedding_dim, embedding_dim)
W_v = torch.rand(embedding_dim, embedding_dim)

Q = torch.matmul(x, W_q).view(batch_size, seq_length, num_heads, head_dim).transpose(1, 2)
K = torch.matmul(x, W_k).view(batch_size, seq_length, num_heads, head_dim).transpose(1, 2)
V = torch.matmul(x, W_v).view(batch_size, seq_length, num_heads, head_dim).transpose(1, 2)

# Compute scaled dot-product attention scores
scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)

# Normalize scores to probabilities
attention_weights = torch.softmax(scores, dim=-1)

# Weighted sum to get context vectors
context = torch.matmul(attention_weights, V)

# Concatenate heads and reshape
context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, embedding_dim)
```

This computational flow encapsulates the core operations of self-attention, enabling the Transformer to learn rich, context-sensitive token representations.

> **[IMAGE GENERATION FAILED]** The flow of computations in scaled dot-product self-attention: input embeddings projected into query, key, and value vectors; dot-product and scaling; softmax for attention weights; weighted sum to produce context vectors.
>
> **Alt:** Flow diagram of scaled dot-product self-attention showing query, key, value projections and attention weight computation
>
> **Prompt:** Technical flowchart of self-attention mechanism showing input embeddings projected into query, key, and value vectors. Include matrix multiplication for dot product between query and key, scaling by sqrt of dimension, softmax normalization to get attention weights, and weighted sum with value vectors to produce output. Include tensor shapes annotations and arrows to indicate data flow.
>
> **Error:** 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. \n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_input_token_count, limit: 0, model: gemini-2.5-flash-preview-image\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-2.5-flash-preview-image\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-2.5-flash-preview-image\nPlease retry in 31.64411203s.', 'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': 'type.googleapis.com/google.rpc.Help', 'links': [{'description': 'Learn more about Gemini API quotas', 'url': 'https://ai.google.dev/gemini-api/docs/rate-limits'}]}, {'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_input_token_count', 'quotaId': 'GenerateContentInputTokensPerModelPerMinute-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-2.5-flash-preview-image'}}, {'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerMinutePerProjectPerModel-FreeTier', 'quotaDimensions': {'model': 'gemini-2.5-flash-preview-image', 'location': 'global'}}, {'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-2.5-flash-preview-image'}}]}, {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '31s'}]}}


## Implementing a Minimal Self-Attention Module in Code

Below is a concise PyTorch-based example illustrating the core components of a self-attention module, designed for clarity and hands-on experimentation.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MinimalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear projections for Q, K, V - map embed_dim to embed_dim for all heads combined
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection to combine heads back to embed_dim
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x: (batch_size, seq_len, embed_dim)
        batch_size, seq_len, _ = x.size()

        # Linear projections and reshape to (batch_size, num_heads, seq_len, head_dim)
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute scaled dot-product attention scores
        # scores: (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Softmax over the last dimension (keys) to get attention weights
        attn_weights = F.softmax(scores, dim=-1)

        # Weighted sum of values
        attn_output = torch.matmul(attn_weights, V)  # (batch_size, num_heads, seq_len, head_dim)

        # Concatenate heads and apply final linear projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(attn_output)

        return output, attn_weights
```

### Key Implementation Details

- **Query, Key, Value Projections:** We use three independent `nn.Linear` layers to transform input embeddings into Q, K, and V tensors, each with shape adapted for multi-head processing.
  
- **Attention Score Computation:** Attention scores are dot-products of queries and keys, scaled by the square root of head dimensionality (`head_dim`) to stabilize gradients.
  
- **Softmax Application:** Softmax normalizes scores along the keys dimension, ensuring attention weights sum to one for each query.
  
- **Multi-Head Handling:** We reshape and transpose tensors to split the embedding dimension into multiple heads, enabling the model to attend to information from different representation subspaces in parallel.
  
- **Dimensionality Choices:** Typical values are `embed_dim` = 512 or 768 and `num_heads` = 8 or 12, with `head_dim = embed_dim / num_heads`. Ensuring `embed_dim` divides evenly by `num_heads` is critical to avoid tensor shape errors.

### Common Pitfalls

- **Dimensionality Mismatches:** Incorrect tensor reshaping or inconsistent embedding sizes across Q, K, V projections cause runtime errors. Always verify shapes after each transformation with debugging prints or assertions: e.g., `print(Q.shape)`.

- **Softmax Temperature:** The scaling factor (`head_dim ** 0.5`) is essential. Omitting it can lead to extremely peaked or flat attention distributions, hurting training stability and convergence.

### Testing Strategies

- **Shape Checks:** Validate shapes after projections, before and after multi-head splits, and after recombination.

- **Attention Output Inspection:** Inspect attention weights visually or by statistics to confirm diverse, meaningful attention patterns rather than uniform or degenerate outputs.

- **Unit Tests:** Write tests to pass inputs of various batch sizes and sequence lengths, verifying output shapes and no exceptions raised.

This minimal implementation serves as a foundation for experimenting with variants like masking, different softmax temperatures, and more sophisticated attention mechanisms.

## Multi-Head Self-Attention and Its Benefits

Multi-head self-attention is a key extension of the standard self-attention mechanism in Transformer architectures. It improves the model's ability to capture complex relationships by splitting and processing input embeddings in parallel.

The input embeddings are divided into multiple smaller heads, each with a reduced dimensionality. For example, given an embedding dimension \(d_{model}\) and number of heads \(h\), each head operates on vectors of size \(d_k = d_{model} / h\). This division allows the model to compute separate attention distributions concurrently, enabling parallel computation and more fine-grained focus on different parts of the input sequence.

After computing attention outputs independently in each head, their respective context vectors are concatenated and linearly transformed back to the original embedding dimension. This concatenation aggregates information from diverse representation subspaces, letting the network attend to multiple aspects of the input simultaneously thus such as syntax, semantics, or positional relationships. By combining these varied perspectives, multi-head attention improves the model's capability to represent complex dependencies and nuanced features within the sequence.

This architecture enhances both expressiveness and learning dynamics. Modeling multiple types of relationships at once makes the network more robust in capturing long-range dependencies and various linguistic phenomena. Additionally, having multiple heads tends to stabilize training by distributing representational responsibilities and reducing the risk of overfitting a single attention pattern.

However, the computational cost of multi-head attention grows linearly with the number of heads, increasing memory and processing time. Optimizing this cost involves strategies like efficient batching and parallelization on hardware accelerators. Frameworks often implement multi-head attention in a fused manner to leverage matrix multiplication efficiencies and reduce overhead.

Choosing the right number of heads depends on practical limits set by embedding size and available memory. Because each head operates on \(d_{model}/h\)-dimensional vectors, the total embedding dimension must be divisible by the number of heads. Common choices include 8 or 16 heads for embeddings of size 512 or 1024. Increasing heads too much can reduce per-head dimension excessively, harming representational power, while too few heads limit diversity. Balancing these factors is crucial for maximizing performance without incurring prohibitive computational costs.

## Edge Cases and Practical Debugging Tips for Self-Attention Models

When implementing self-attention in transformer architectures, developers often encounter a set of recurring pitfalls. Here are actionable insights to address them effectively:

- **Shape Mismatches**: Self-attention computations involve multiple tensor dimensionsbatch size, number of heads, sequence length, and feature size. Common errors arise when these dimensions are misaligned, especially during reshaping or permutation steps (e.g., splitting hidden states into multiple heads). Always verify shapes after each transformation using assertions or debugging prints. For example, queries, keys, and values should share compatible shapes before the scaled dot-product operation:

  ```python
  assert queries.shape == (batch_size, num_heads, seq_len, head_dim)
  assert keys.shape == queries.shape
  assert values.shape == queries.shape
  ```

- **Incorrect Scaling Factor**: Scaling the dot product of queries and keys by the square root of the head dimension (d_k) stabilizes gradient flow. Using an incorrect factor can lead to unstable attention scores, resulting in gradient explosion or vanishing attention weights. Ensure this factor matches the dimensionality of your heads precisely:

  ```python
  scaling_factor = torch.sqrt(torch.tensor(head_dim, dtype=torch.float32))
  attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / scaling_factor
  ```

- **Inspecting Attention Weights**: Degenerate attention, such as uniform or nearly one-hot distributions, often indicates model issues. Visualize attention weight distributions or print statistics (mean, variance) per head to spot these anomalies. This helps identify if the model is collapsing focus or ignoring parts of the input.

- **Logging Intermediate Tensors and Gradient Norms**: Debugging training instabilities benefits from logging intermediate outputs and gradient norms. Track tensors like attention scores, weights, and output values during forward and backward passes. Gradient norm clipping can also be employed if exploding gradients are detected:

  ```python
  for name, param in model.named_parameters():
      if param.grad is not None:
          print(f"{name} grad norm: {param.grad.norm().item():.4f}")
  ```

- **Numeric Stability Fixes**: To prevent overflow during the softmax calculation, add a small epsilon or, better yet, apply the standard practice of subtracting the maximum attention score per query before softmax:

  ```python
  max_scores, _ = attention_scores.max(dim=-1, keepdim=True)
  stable_scores = attention_scores - max_scores
  attention_weights = torch.softmax(stable_scores, dim=-1)
  ```

- **Unit Testing Attention Outputs**: Implement unit tests that verify shapes, valid probability distributions (attention weights should sum to 1 along the last dimension), and expected output ranges. Example test could check if softmax outputs sum to 1 for a mock input tensor.

By integrating these debugging strategies, you improve the robustness and transparency of your self-attention implementations, saving valuable development time and increasing model reliability.

## Performance and Computational Cost Considerations in Self-Attention

Self-attention mechanisms lie at the core of Transformer architectures but come with significant computational costs, particularly as input sequence lengths grow. The fundamental challenge stems from the quadratic complexity: for an input sequence of length *n*, self-attention computes pairwise interactions between all pairs of tokens, resulting in O(n  ) time and memory complexity. This cost quickly escalates with longer sequences, limiting scalability in practical applications.

When comparing single-head to multi-head attention, multi-head setups multiply these costs by the number of heads *h*, as separate attention computations run in parallel. While multi-head attention captures diverse representations and generally improves model expressiveness, it also increases resource consumption in terms of compute cycles and memory usage.

To mitigate these challenges, several optimization strategies have emerged:

- **Sparse Attention:** Limiting attention computations to a subset of token pairs (e.g., local neighborhoods or fixed patterns) reduces complexity to near-linear, trading off some global context for efficiency.
- **Approximate Methods:** Techniques such as low-rank factorization or kernel-based approximations reduce the dot-product calculations required, enabling faster inference.
- **Pruning:** Removing less important attention heads or parameters dynamically or during training minimizes overhead without drastic accuracy loss.

These optimizations inherently involve trade-offs between model accuracy and runtime cost. Sparse or approximate methods may dampen model performance on tasks needing full context, so tuning these mechanisms requires domain-specific evaluation.

Hardware acceleration plays a critical role in alleviating self-attention bottlenecks. GPUs and TPUs leverage parallelism for matrix multiplications fundamental to attention computation. Effective batching strategies also boost throughput by exploiting hardware parallelism further, although care is needed to accommodate variable sequence lengths through padding or masking.

Memory management techniques like activation checkpointing can substantially reduce GPU memory consumption by trading compute overhead for storage savings. Instead of storing all intermediate activations, checkpointing recomputes them during backpropagation, enabling larger models or longer sequences to fit within memory constraints.

Developers building Transformer-based applications should balance precision and efficiency by considering these computational characteristics and leveraging suitable optimizations, hardware features, and memory-saving tactics to meet their performance goals.

> **[IMAGE GENERATION FAILED]** Multi-head self-attention architecture showing splitting of embeddings into multiple heads, parallel attention computations, concatenation, and implications on computational cost and performance.
>
> **Alt:** Illustration of multi-head self-attention mechanism and computational cost scaling
>
> **Prompt:** Illustration showing multi-head self-attention mechanism: splitting input embedding into multiple heads, parallel scaled dot-product attention in each head, concatenation of output heads, and final linear transformation. Include annotations highlighting benefits for capturing diverse relationships and notes on scaling computational cost with number of heads. Visualize complexity trade-offs and hardware acceleration aspects.
>
> **Error:** 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. \n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-2.5-flash-preview-image\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-2.5-flash-preview-image\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_input_token_count, limit: 0, model: gemini-2.5-flash-preview-image\nPlease retry in 31.386077192s.', 'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': 'type.googleapis.com/google.rpc.Help', 'links': [{'description': 'Learn more about Gemini API quotas', 'url': 'https://ai.google.dev/gemini-api/docs/rate-limits'}]}, {'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-2.5-flash-preview-image'}}, {'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerMinutePerProjectPerModel-FreeTier', 'quotaDimensions': {'model': 'gemini-2.5-flash-preview-image', 'location': 'global'}}, {'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_input_token_count', 'quotaId': 'GenerateContentInputTokensPerModelPerMinute-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-2.5-flash-preview-image'}}]}, {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '31s'}]}}

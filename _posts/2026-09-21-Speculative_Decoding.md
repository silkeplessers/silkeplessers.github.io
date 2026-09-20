---
layout: post
title: "Guess First, Verify Fast: The Evolution of Speculative Decoding"
date: 2026-09-21
categories: [Transformers, Inference Optimization]
tags: [Transformer, Speculative Decoding, EAGLE, EAGLE-3, MTP, DFlash, DFlash 2, DSpark]
excerpt: "Inference speed has become one of the biggest challenges in modern AI. While model capabilities continue to improve, autoregressive generation remains inherently sequential. Speculative decoding offers a way around this limitation by allowing models to predict and verify multiple tokens at once. In this article, we explore the techniques behind speculative decoding and how approaches such as EAGLE, MTP, DFlash, and DSpark are pushing LLM inference toward greater efficiency and parallelism."
---

Over the past few years, the AI industry has been obsessed with model quality. Every new model release promises better reasoning, stronger coding capabilities, larger context windows, or higher benchmark scores. Yet behind the scenes, a different challenge has quietly become one of the biggest concerns for organizations deploying large language models at scale: inference efficiency.

For many production systems, the question is no longer whether a model can answer a question correctly. Instead, the challenge is whether it can do so quickly and cheaply enough to serve millions of requests. As models grow larger and application adoption increases, latency and infrastructure costs can quickly become more important than marginal improvements in benchmark performance. A model that delivers the same answer twice as fast can often create more business value than a model that is only slightly more capable.

This focus on efficiency has led researchers to reexamine the biggest remaining bottleneck in LLM inference: autoregressive generation itself. Rather than merely accelerating individual decoding steps, the goal has increasingly become finding ways to generate more tokens with fewer sequential operations.

This shift in priorities helps explain why speculative decoding has moved from a niche research topic to one of the hottest areas in LLM inference. Over the last two years, a growing number of model families have started shipping with built-in support for speculative decoding techniques, particularly through Multi-Token Prediction (MTP) heads. At the same time, inference frameworks such as vLLM, SGLang, and TensorRT-LLM have rapidly expanded their support for speculative decoding algorithms, making techniques like EAGLE-3, DFlash, and DSpark practical deployment options rather than purely academic research projects.

The reason for this interest is simple. Despite all of the progress in model architecture, modern language generation remains constrained by a limitation that has existed since the earliest GPT-style models: text generation is still fundamentally autoregressive. Every generated token depends on the token before it. No matter how many GPUs are available, the next token cannot be produced until the current token exists. This seemingly simple constraint has become one of the most significant barriers to achieving lower latency and higher throughput. Speculative decoding is, at its heart, an attempt to work around that limitation without changing the final answer produced by the model.

## The Autoregressive Wall

To understand why speculative decoding has become so important, it is worth taking a closer look at how LLM inference actually works.

When a user submits a prompt, the model first processes the entire input sequence. This stage, commonly referred to as the prefill phase, is relatively efficient. During prefill, the model can process many tokens simultaneously, allowing modern GPUs to exploit their highly parallel architecture. Large batches of computation can be executed at once, keeping hardware utilization high.

The situation changes dramatically when generation begins.

Once the model starts producing an answer, it enters the decode phase. Instead of processing entire sequences, the model now generates one token at a time. Each generated token is appended to the sequence and becomes part of the context for generating the next token.

This means that generating a response containing 200 tokens requires roughly 200 decoding iterations. Every iteration involves loading model weights, performing transformer computations, updating attention caches, and producing the next token prediction. Even though the operation is conceptually simple, it becomes remarkably expensive when repeated hundreds or thousands of times.

<div class="img-medium">

![Infographic comparing LLM prefill and decode phases, highlighting parallel prompt processing versus sequential token generation and explaining why decode dominates inference latency..](/assets/images/Speculative_Decoding/prefill_decode.png)
*LLM inference workflow showing why prefill is highly parallel while decode becomes the primary performance bottleneck during autoregressive generation.*

</div>

The problem is not merely computational. For modern frontier models, decoding is often memory-bandwidth bound rather than compute-bound. In other words, the GPU spends an enormous amount of time moving model parameters and KV-cache data through memory systems instead of performing arithmetic operations. As model sizes continue growing, this memory movement becomes increasingly expensive.

This is why decoding latency improves more slowly than raw hardware numbers would suggest. Even the most powerful GPU cannot escape the sequential nature of autoregressive generation: for a single request, the next token cannot be produced until the previous one has been decoded, so per-sequence speed is bounded by the number of sequential decoding steps required. Batching can raise aggregate throughput and improve hardware utilization, but it does not remove the sequential token dependency within an individual sequence.

The challenge was no longer improving model quality, but finding ways to generate the same output with fewer expensive decoding steps. Speculative decoding grew out of this need and has since become one of the most important inference optimizations in modern LLM serving.

## The Core Idea Behind Speculative Decoding

The intuition behind speculative decoding is surprisingly straightforward.

Language models are often quite predictable over short horizons. When a sentence begins with "The capital of France is", there are not many likely continuations. Humans can often predict the next few words with high confidence, and language models can do the same.

Speculative decoding exploits this predictability. Rather than forcing the large model to generate every token individually, a faster drafting mechanism is allowed to guess several tokens ahead. Those drafted tokens are appended to the current context and passed through the larger target model in a single forward pass. Because a transformer computes logits for every position in the sequence simultaneously, the target model can score all drafted tokens at once, producing the probabilities it would have assigned to each position under normal decoding. The acceptance test is then applied from left to right, keeping the longest valid prefix of drafted tokens and discarding the rest. If the guesses are correct, multiple tokens can be accepted simultaneously, allowing the target model to skip several otherwise expensive decoding iterations. Verification isn't only a yes/no check. Because the target scores every drafted position in the same pass, it also produces its own token at the first rejected position, or one extra token if everything was accepted. Each verification pass therefore yields at least one and at most k+1 tokens, and a rejection costs no extra pass.

The important detail is that the target model always remains the final authority. Incorrect guesses are rejected and generation continues normally. The speculative component is therefore not replacing the target model. It is merely helping the target model move faster when future tokens are sufficiently predictable.

<div class="img-medium">

![The autoregressive bottleneck in LLM inference. Traditional decoding generates one token at a time, whereas speculative decoding can accept multiple correctly predicted tokens at once, improving throughput and reducing latency without changing the model's output.](/assets/images/Speculative_Decoding/autoregressive_vs_specdec.png)
*The autoregressive bottleneck in LLM inference. Traditional decoding generates one token at a time, whereas speculative decoding can accept multiple correctly predicted tokens at once, improving throughput and reducing latency without changing the model's output distribution.*

</div>

This distinction is crucial because it explains why speculative decoding can often provide significant speedups without degrading output quality. The final output still comes from the same target model. Speculation reduces the number of sequential target-model decoding rounds. The target still evaluates the drafted positions, but it can score them together in one verification pass, trading additional parallel work for fewer latency-dominating sequential passes.

As simple as this idea sounds, the challenge has always been determining how to generate useful draft tokens efficiently. The history of speculative decoding can largely be viewed as a series of increasingly sophisticated answers to that question.

### Moving beyond simple draft models

The earliest implementations of speculative decoding relied on a separate draft model. The concept was intuitive: a small language model could generate candidate continuations much faster than a large frontier model. The larger model would then verify those candidates and accept as many as possible.

This approach demonstrated that substantial inference acceleration was possible and quickly established speculative decoding as a practical technique for serving large language models. However, maintaining a separate draft model introduced its own complexity. The draft model had to be trained, deployed, and kept aligned with the target model, creating additional infrastructure and operational overhead.

An obvious question followed: could the drafter make better use of information already computed by the target model, rather than behaving as an entirely independent smaller language model?

EAGLE addresses this by training a lightweight draft model around the target model’s internal representations. The drafter remains a separate learned component, but it leverages the internal representations already produced by the target model itself.

The key insight is that hidden states contain far more information than the next token prediction alone. During generation, every transformer layer builds a rich representation of the current context. Those representations already encode strong signals about likely future continuations. Instead of training a separate draft model from scratch, EAGLE learns to exploit these hidden states directly.

Concretely, EAGLE predicts future tokens from the target model's hidden-state space rather than from previously generated token IDs alone. This allows a lightweight drafting module to anticipate upcoming generations using information the model has already computed, avoiding much of the redundancy present in traditional two-model systems.

The result is a fundamentally different approach to drafting. Instead of pairing a large model with a separately trained assistant model, EAGLE turns information already available inside the target model into a source of speculative proposals. 

## EAGLE-3 and the Importance of Accepted Tokens

EAGLE demonstrated that the drafter did not have to operate as an independent miniature language model. By conditioning a lightweight draft model on target-derived features, it improved alignment between drafting and verification.

However, EAGLE still faced the same challenge that affects every speculative decoding technique: proposal quality tends to degrade as speculation extends further into the future. Hidden states contain information about likely continuations, but uncertainty naturally increases the farther ahead the model tries to predict.

This creates a practical bottleneck. A speculative window of eight tokens may look impressive on paper, but its value depends entirely on how many of those tokens survive verification. If a mismatch occurs early in the window, the remaining drafted tokens are discarded regardless of how they were generated.

EAGLE-3 changes two important parts of the original design. First, it abandons explicit future-feature prediction and trains the drafter to predict tokens directly. Second, while the original EAGLE primarily relies on the final hidden representation used for next-token prediction, EAGLE-3 broadens the information available to the drafting mechanism by incorporating representations from multiple layers throughout the transformer.

The motivation is that no single layer carries everything a drafter needs. As the EAGLE-3 authors put it, top-layer features are effectively specialised for predicting the very next token, which makes them a weak basis for predicting tokens further ahead, while intermediate layers hold information that is useful for those later positions. By fusing features from the lower, middle and upper layers into a single vector, EAGLE-3 gives the drafting mechanism a richer view of the generation process than any single hidden state can provide.

EAGLE-3 also changes how the drafter is trained, using a technique called training-time test. The problem it solves is a mismatch between training and inference. At inference, the drafter proposes several tokens in a row, and from the second token onward its input is its own earlier guess, which may be slightly off. Errors can therefore compound with every step. In conventional training, though, the drafter only ever sees clean inputs computed by the target model, so it never practises working from its own imperfect predictions. It is trained under easy conditions and then deployed under harder ones, and its accuracy drops at deeper draft positions. Training-time test closes that gap by running the multi-step drafting loop during training. The drafter feeds its own outputs back in as inputs, and the loss is computed at every step, not just the first.

The goal is not necessarily to speculate further ahead. The goal is to make each speculative prediction more accurate. In practice, this translates into longer accepted runs during verification. More drafted tokens match what the target model would have generated anyway, reducing wasted computation and improving end-to-end throughput.

<div class="img-medium">

![Different transformer layers capture different types of information. EAGLE-3 leverages these richer representations to generate proposals that more closely match the target model's final output.](/assets/images/Speculative_Decoding/eagle_eagle3.png)
*Different transformer layers capture different types of information. EAGLE-3 leverages these richer representations to generate proposals that more closely match the target model's final output.*

</div>

## Multi-Token Prediction: Native Speculation

Multi-Token Prediction (MTP) extends the traditional next-token prediction objective used to train autoregressive language models. Rather than learning only to predict the immediate next token, the model is trained to anticipate several future positions at once.

The exact implementation varies across architectures. Some models attach multiple prediction heads to a shared transformer representation, while others use lightweight sequential modules that gradually forecast further into the future. In both cases, the objective is the same: teaching the model to look beyond the next token and develop stronger representations of future context.

These additional predictions can improve training efficiency, but they also create an interesting opportunity during inference. These future-token forecasts can also be reused as speculative proposals during inference, turning the model itself into a built-in drafting mechanism. The target model then verifies those proposals in the same way as other speculative decoding systems.

<div class="img-medium">

![Infographic titled “MTP Speculative Decoding Flow.” A prompt and context input feeds into a target model augmented with MTP heads. The model generates a sequential drafting chain of tokens labeled T+1, T+2, T+3, and T+4, with T+1 shown as the first generated token and later tokens shown as speculative draft tokens. The drafted sequence is sent to a target verification stage. Verification branches into two outcomes: a green “Accept Tokens” path that appends accepted tokens to the output, and an orange “Reject → Use Target Token” path indicating that the target model’s own token replaces the first rejected draft token. A note explains that the correction token comes directly from the verification pass and does not require an additional verification pass. Another note states that MTP architectures may perform drafting sequentially or in parallel depending on the implementation.](/assets/images/Speculative_Decoding/MTP.png)
*MTP speculative decoding extends a target model with auxiliary prediction heads that draft multiple future tokens. The target model then verifies the drafted sequence in a single pass, accepting correct tokens and replacing the first rejected token with its own prediction.*

</div>

MTP can therefore be viewed as a form of native speculation. Instead of relying on an entirely separate drafting model, the ability to propose future continuations is learned directly during training. That said, MTP should not be confused with fully parallel drafting approaches such as DFlash. Depending on the architecture, generating those future predictions may still involve sequential computation. The distinction between parallel and autoregressive drafting therefore depends on the specific MTP design rather than the training objective itself.

## The Shift Toward Parallel Drafting

EAGLE and MTP improve speculative decoding, but both remain fundamentally autoregressive. Draft generation still depends on sequential hidden-state updates.

Newer approaches focus on increasing the amount of drafting work that can be performed in parallel, reducing reliance on sequential computation and improving hardware utilization during inference.

## DFlash

DFlash drafts using a block-diffusion approach rather than an autoregressive one. A normal autoregressive draft model still has to run one forward pass per guessed token. DFlash instead uses a small draft model that fills in an entire block of future tokens in a single forward pass, conditioned on hidden states pulled straight from the target model.

DFlash uses non-causal attention over a block of masked future positions, enabling all positions in the block to be decoded in parallel. Unlike standard autoregressive drafting, where token t+1 must be generated before token t+2, DFlash removes these intra-block dependencies and predicts every position simultaneously. The resulting draft cost scales weakly with block size, allowing many future tokens to be proposed for nearly the cost of a single decoding step.

The parallelism that makes DFlash fast is also its main limitation. Because the entire block is drafted in a single pass, there is no autoregressive mechanism that lets later positions react to a revised prediction at an earlier position within the same block. All tokens are proposed simultaneously from a shared representation rather than being generated one after another. Language, however, is highly sequential: the best choice for a token often depends on the exact tokens that came before it. As the draft length grows, this lack of intra-block autoregressive conditioning can reduce prediction quality.

In practice, you can see this in the model's acceptance rates. The first few tokens in a drafted block are often quite accurate, but confidence tends to fall as you move deeper into the sequence. That's not especially surprising: the model is making several predictions at once, without knowing how its earlier guesses will ultimately resolve. The further ahead it looks, the more uncertainty accumulates, and the more likely it is that later predictions will need to be corrected.

That becomes important during verification. The target model only accepts a continuous run of correct tokens from the beginning of the block. If a token fails halfway through, everything after it is effectively lost, regardless of whether some of those later predictions might have been correct. As a result, drafting further ahead does not automatically translate into larger speedups. The challenge is keeping the entire block coherent enough that a meaningful portion survives verification.

## How DSpark fixes it

DSpark, from DeepSeek, doesn't throw out DFlash's approach: it preserves the parallel drafting backbone but augments it with a lightweight sequential component that models dependencies within the proposed block.

A low-rank Markov head biases each draft position's prediction using the token that comes immediately before it, reintroducing a cheap form of the token-to-token dependency that pure parallel drafting throws away, without paying for a fully sequential pass through the whole backbone. Alongside it, a confidence head estimates how likely each drafted token is to survive verification, and a scheduler in the serving engine uses those estimates to decide how far into the block to verify. Under light load, checking extra tokens is nearly free. Under heavy load, it wastes capacity that other requests could use, so the block is cut short.

The appeal is that DSpark gets most of the benefit of sequential awareness back cheaply: it's still one forward pass per block, just with a light correction layered on top. Reported results show this closing a meaningful chunk of the gap in accepted draft length compared with both plain DFlash and autoregressive-style drafters, and translating into substantially faster per-user generation in production serving.

<div class="img-medium">

![DFlash maximizes parallelism by predicting every draft position independently. DSpark preserves the same single-pass drafting approach while reintroducing lightweight token dependencies, improving acceptance rates and extending accepted token runs during verification.](/assets/images/Speculative_Decoding/dflash_dspark.png)
*DFlash maximizes parallelism by predicting every draft position independently. DSpark preserves the same single-pass drafting approach while reintroducing lightweight token dependencies, improving acceptance rates and extending accepted token runs during verification.*

</div>

### Not the end of the story

The DFlash team has since followed up with DFlash 2, which attacks the same tail-decay problem from two directions. The first is the missing intra-block ordering: a small convolution lets each position in the block see the one before it, restoring a local sense of order without giving up the single-pass design. The second is a candidate selector. Instead of picking each position's best token independently, it scores combinations of adjacent candidates and walks a single coherent path through the block, which targets the same collision problem DSpark's Markov head addresses. The result is a stronger draft that remains more coherent across longer blocks, improving acceptance rates and, according to the authors, edging out DSpark on several benchmarks.

## Memory: The Often Overlooked Constraint

Inference speed is only part of the deployment equation. Memory requirements can be equally important, particularly for organizations operating large models near GPU capacity limits.

One of the attractive aspects of MTP is that it requires no separate checkpoint or draft model: the draft capability rides inside the model's own weights. It is not free however, however: the additional prediction modules still introduce parameters, but it avoids the deployment complexity, extra process and KV overhead of a standalone drafter.

EAGLE, DFlash and DSpark generally require an additional drafting component. While these draft models are typically much smaller than the primary model, they still consume memory resources and may introduce additional KV-cache overhead.

The choice between approaches therefore involves more than raw throughput. Deployment teams must balance speedup gains against memory consumption, operational complexity and hardware constraints. In many production environments, the best solution is not necessarily the fastest solution but the one that delivers the best overall efficiency profile.

## Measuring Success

One of the most common mistakes when evaluating speculative decoding systems is focusing exclusively on tokens per second.

Throughput certainly matters, but it rarely tells the complete story.

Acceptance rate remains one of the most important metrics because it determines how much speculative work survives verification. Looking at acceptance rate per speculative position can provide even deeper insight by revealing where prediction quality begins to deteriorate.

Many teams also track average accepted token length, as this often correlates more directly with real-world speedups than overall acceptance percentages. End-to-end latency, time-to-first-token and cost per generated token are equally important, especially for production-facing applications where user experience and infrastructure costs matter more than synthetic benchmark performance.

Ultimately, speculative decoding is a systems optimization problem rather than a purely algorithmic one. The best configuration depends on workload characteristics, model size, hardware architecture and operational constraints.

## Looking Ahead

The evolution of speculative decoding can be viewed as a gradual attempt to loosen the grip of autoregressive generation. The journey began with simple draft models, progressed through hidden-state approaches such as EAGLE, extended into native forecasting mechanisms such as MTP, and is now moving toward increasingly parallel drafting architectures such as DFlash and DSpark.

What is particularly interesting is that none of these approaches completely eliminate autoregression. Instead, they reduce the amount of sequential work required to generate high-quality text. Each new generation pushes a little more computation toward parallel execution while preserving the quality guarantees provided by the target model.

Whether future language models eventually move beyond autoregressive generation entirely remains an open research question. What is clear, however, is that inference efficiency has become one of the most active areas of innovation in modern AI. As model quality improvements become increasingly incremental, the ability to generate the same answer faster and more cheaply may become one of the most valuable capabilities of all.

Speculative decoding is no longer a niche optimization. It is rapidly becoming a fundamental component of the modern LLM inference stack, and its evolution offers a fascinating glimpse into how the next generation of language model serving systems may be built.
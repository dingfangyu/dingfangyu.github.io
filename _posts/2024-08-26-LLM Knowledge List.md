---
title: 'LLM Knowledge List - Basics'
tags:
  - LLM
  - training
  - inference
  - fine-tuning
  - kernel
  - model architecture
  - hardware
---

This post summarizes the basic knowledge about LLM (~50 papers) 

* TOC 
{:toc}


<!-- git add . && git commit -m 'llm' && git push  -->

## Training

- parallelisms, distributed training
  - data parallelism
    - allreduce grad
  - pipeline parallelism
  - tensor parallelism
    - matrix computations in self-attention (SA) & feed-forward network (FFN) are composed of neuron (i.e. a row/column vector) computations
    - we can gather the neurons into (disjoint) clusters, and dispatch different clusters to different computational units (e.g. GPUs) to implement tensor parallelism
  - sequence parallelism
    - ring attention, computation-communication overlap
  - expert parallelism
    - deepspeed-MoE
  - zero
    - zero-infinity, offloading system making use of CPU/NVMe memories
  
- training task property

  - compute bound

    - when the training scale is not extremely large, the computation cost is dominant, rather than the GPU memory accessing cost and network communication cost

    - the computation cost can be measured in terms of Floating-point operations (FLOPs or Flops), which is easy to estimate by the rule [[backward flops details]](../Transformer-Gradients/):

      $$
      forward\_FLOPs = 2 \times \#tokens \times \#activated\_paramters 
      $$

      $$
      backward\_FLOPs = 2 \times forward\_FLOPs 
      $$

    - where "activated" is specially referred for the case of Mixture-of-Expert (MoE) models, if the model is not MoE (i.e. is dense model), the activated paramters is the total parameters
  
  - memory consuming 
  
    - during the training process, other than the parameters, we also need to save the gradients and optimizer states proportionally to the original amount of parameters 
    - 224444: p(16)/g(16)/p(32)/g(32)/os1(32)/os2(32)
    - we need (20 * parameter_size) GB GPU memory to store the whole model states (p, g, os1, os2) for fp16-fp32 mixed precision training

## Inference

- algorithm
  - speculative inference
    - sequoia
  - sparsity
    - dejavu
    - powerinfer
    - moefication

- prompting
  - RAG

- serving system
  - vllm
  - tensorRT
  - disaggregation (on heterogeneous hardware settings)
    - DistServe
      - prefill-decoding disaggregation
    - Attention Offloading
      - SA-FFN disaggregation

  - FlexGen
    - offloading

- inference task property
  - prefill-decoding 2 stages
    - KV cache
  - memory bound
    - when the batch size is not extremely large (~200), the bottleneck of inference speed is the GPU memory accessing of model weights & KV cache, rather than the computation cost and communication cost
    - it is meaningless to transform a memory bound task to a GPU-CPU/NVMe IO bound task, e.g. designing an offloading system for inference



## Fine-Tuning

- supervised fine-tuning (SFT)
- efficiency
  - parameter-efficient fine-tuning (PEFT)
    - LoRA (low-rank)
      - DoRA
      - LoRA+
      - AdaLoRA
  - memory-efficient (training)
    - GaLore (low-rank, SVD)
  - add noise to reduce overfitting
    - NEFT
- FT task property
  - compute bound (like training)
  - not memory consuming (with parameter/memory-efficient methods)



## Kernel

- flash attention
  - 1,2,
  - 3
    - hopper gpu (overlap cuda core, tensor core, TMA)
  - decoding
- kernel fusion
  - triton
    - unsloth (lora)



## Model Architecture

- transformer
  - SA
    - GQA, MQA, MLA (deepseek)
    - window slice (mixtral)
  - FFN
    - MoE
      - MoE scaling law, #experts
    - SwiGLU (llama)
  - positional encoding
    - RoPE, to rotate a d-dim vector 2-dim by 2-dim
  - normalization
    - RMSnorm
  - residual link
    - early exiting
- efficient transformer (attention)
  - linformer
    - J-L lemma
  - performer
    - matmul associativity
    - gaussian integral (gaussian distribution pdf integral)
    - random fourier/positive feature maps
    - variance analysis
  - linear attention
  - hyperattention
- other structures (RNNs)
  - mamba
  - TTT



## Compression

- kv cache, long context
  - CaM
  - StreamingLLM
  - memory augmented



## Quantization

- AWQ



## Hardware

- gpu

  - specifications
    - tflops
    - membdw
    - nvlink
    - cap
    - pcie
    - ib
  - hierarchy
    - mem - grain
      - hbm - grid
      - l2 cache - block cluster
      - shared mem - block, warp
      - register mem - thread
  - architecture
    - tensor core
    - cuda core
    - TMA
    - SM
    - warp
    - register

  - cpu
    - thread
  - disk
    - IO bdw
      - page cache




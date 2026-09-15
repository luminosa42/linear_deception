# Detecting Strategic Deception: Small-Scale Replication + Causal-Use Extension

This is a research project for the Bluedot technical AI safety project sprint. 
The main aim is to replicate the core methodology of Goldowsky-Dill et al., 2025 (Apollo Research), "Detecting Strategic Deception Using Linear Probes," at small scale, 
and extending it with a causal-use test the original paper leaves open.

## Motivation

The original paper trains linear probes on a model's internal activations to detect whether it is being deceptive, and reports strong separation (AUROC 0.96–0.999) on realistic held-out scenarios. 
This is a correlational result: it shows deception-related information is linearly decodable from activations, not that the model's behavior is causally produced by that direction. This project:

- Replicates the core probing methodology at a laptop-friendly scale (Qwen2.5-1.5B-Instruct, CPU).
- Extends it with an activation patching experiment to test whether the probe direction is causally load-bearing, or merely a decodable correlate — the central open question the original paper does not test.

## Project structure
(to be updated)

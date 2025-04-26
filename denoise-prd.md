Product Requirements Document

Project Overview

Our hackathon project is an end-to-end automated theorem proving system composed of four collaborating microservices:
	•	Team Member A: Tokenizer Service
	•	Implements a Byte-Pair Encoding (BPE) tokenizer with math-symbol extensions.
	•	Exposes POST /tokenize to convert raw theorem text into token IDs.
	•	Provides unit tests for edge-case math expressions and Dockerization for local orchestration.
	•	Team Member B: Forward Noising Service
	•	Integrates SEDD’s entropy-based noise schedule to corrupt proof tokens per timestep αₜ.
	•	Wraps S4 backbone modules for efficient noise propagation.
	•	Exposes POST /noise accepting token IDs and timestep, returning noisy tokens, with performance benchmarks.
	•	Team Member C (You): Reverse Denoiser & Prompt Orchestrator
	•	Receives noisy tokens and αₜ, retrieves relevant lemmas, and constructs few-shot prompts.
	•	Calls Claude API (with RAG context) to denoise and re-logicalize proofs step-by-step.
	•	Handles rate limiting, caching, parallel execution, and exposes POST /denoise for cleaned token outputs.
	•	Team Member D: Retrieval Module & Frontend UI
	•	Manages a vector database of lemmas/theorems and provides POST /retrieve to fetch context.
	•	Builds a Next.js + ShadCN/ui interface for theorem submission, proof visualization, and error handling.
	•	Develops end-to-end tests and deployment configuration for a seamless user experience.

⸻

Service: Reverse Denoiser & Prompt Orchestrator

Owner: Team Member C
Date: April 26, 2025

⸻

1. Purpose & Scope

The Reverse Denoiser & Prompt Orchestrator service turns noisy, SEDD-generated token sequences back into coherent, logically valid proof steps by orchestrating calls to Claude (with Retrieval-Augmented context) at each denoising timestep. It provides a single REST endpoint (POST /denoise) that downstream services (UI or verifier) call to obtain cleaned token streams ready for decoding.

2. Objectives
	•	Accuracy: Correct SEDD noise artifacts and logical gaps using Claude’s reasoning.
	•	Performance: Maintain end-to-end denoising latency under 200 ms per timestep.
	•	Reliability: Gracefully handle Claude API rate limits, transient failures, and malformed inputs.
	•	Scalability: Support parallel denoising of multiple proof subgoals.
	•	Testability: Deliver comprehensive mocks and integration tests for CI/CD.

3. Functional Requirements

ID	Requirement
FR1	Expose POST /denoise accepting { noisy_tokens: number[], alpha_t: float }.
FR2	For each timestep:
	• Fetch top-k relevant lemmas via internal GET /retrieve?query=<subgoal>.
	• Construct a few-shot prompt template including noisy tokens, αₜ, and retrieved context.
	• Call Claude’s completion API with low temperature (≤ 0.2).
	• Parse and return cleaned token IDs or structured proof steps.
FR3	Implement rate-limit back-off (exponential) and retry logic (max 3 attempts).
FR4	Cache identical prompts and their cleaned outputs in-memory or Redis to reduce API calls.
FR5	Support parallel execution for batches of timesteps.
FR6	Validate input schema; return HTTP 400 for malformed requests.
FR7	Expose metrics (Prometheus): denoise_requests_total, denoise_errors_total, avg_latency_ms.

4. Non-Functional Requirements
	•	Latency: ≤ 200 ms/timestep (95th percentile)
	•	Throughput: 50 concurrent denoise calls
	•	Availability: 99.9% uptime
	•	Security: All endpoints require an API key; TLS enforced
	•	Documentation: Auto-generated OpenAPI spec + example curl snippets

5. API Specification

POST /denoise
	•	Request

{
  "noisy_tokens": [12, 45, 78, …],
  "alpha_t": 0.3
}


	•	Response (200 OK)

{
  "clean_tokens": [12, 37, 89, …],
  "debug": {
    "prompt_id": "abc123",
    "claude_latency_ms": 123
  }
}


	•	Error Codes
	•	400: invalid/missing fields
	•	429: rate limit exceeded, with Retry-After header
	•	500: internal error, with human-readable message

6. Data Flow & Dependencies
	1.	Input: receives noisy token sequence + αₜ
	2.	Retrieve Context: calls Retrieval service (/retrieve) to get top-k lemmas
	3.	Prompt Build: fills few-shot template with:
	•	Noisy proof fragment
	•	αₜ value
	•	Retrieved lemmas/theorems
	4.	Claude API: sends prompt → receives denoised proof text
	5.	Parsing: maps text back to token IDs or structured proof-step JSON
	6.	Output: returns cleaned tokens + metadata

7. Success Metrics
	•	Denoise Accuracy: % of timesteps passing formal checker on first try ≥ 85%
	•	Latency: P95 ≤ 200 ms
	•	Cache Hit Rate: ≥ 60% for repeated subgoals
	•	Error Rate: < 0.5% of requests result in 5xx

8. Milestones & Timeline

Milestone	Date
Define OpenAPI spec & data schema	Day 1
Implement core /denoise logic	Day 2–3
Integrate rate-limit & caching layers	Day 4
Mock Claude & write integration tests	Day 5
End-to-end integration with Retrieval UI	Day 6
Performance tuning & load testing	Day 7

9. Risks & Mitigations
	•	API Flakiness: use retries + exponential back-off.
	•	Prompt Drift: maintain versioned prompt templates; add monitoring on output quality.
	•	Cost Overrun: enforce daily token-usage quotas; leverage cache aggressively.

10. Appendix
	•	Prompt Templates: stored under templates/denoise/*.tpl
	•	Environment Variables:
	•	CLAUDE_API_KEY
	•	RETRIEVAL_SERVICE_URL
	•	CACHE_TTL_SEC
	•	MAX_PARALLEL_REQUESTS
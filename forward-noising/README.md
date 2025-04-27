# Discrete Diffusion for Mathematical Proof Generation

This project implements a novel architecture for human-like mathematical proof generation using discrete diffusion models. By leveraging the Score Entropy Discrete Diffusion (SEDD) framework and Claude API for denoising, this system generates coherent, logically sound proofs.

## Architecture

Our system is divided into four microservices:

```
┌────────┐     ┌───────────┐     ┌────────────┐     ┌───────────┐
│Proof   │     │Forward    │     │Claude API  │     │Retrieval  │
│Dataset ├─►───│Noising    │◄───►│Denoising   ├─►───►│Module     │
└────────┘     │(SEDD+SSM) │     │(Reverse    │     └───────────┘
               └───────────┘     │ Diffusion) │
                                  └───────────┘
```

1. **Tokenizer Service**: Implements Byte-Pair Encoding tokenizer with math symbol extensions
2. **Noising Service**: Applies SEDD-based noise to token sequences with support for hyperschedules
3. **Denoiser Service**: Constructs prompts for Claude API to perform denoising and reverse diffusion
4. **Retrieval & UI Service**: Fetches relevant theorems/lemmas and provides a user interface

## Getting Started

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- Anthropic API Key (for Claude)

### Running the Services

1. Clone the repository
2. Set up environment variables:
   ```
   cp .env.example .env
   # Edit .env with your API keys
   ```
3. Start the services:
   ```
   docker-compose up
   ```
4. Open http://localhost:3000 in your browser

## Development

Each service has its own README with specific development instructions:

- [Tokenizer Service](./tokenizer-service/README.md)
- [Noising Service](./noising-service/README.md)
- [Denoiser Service](./denoiser-service/README.md)
- [Retrieval UI Service](./retrieval-ui-service/README.md)

## Testing

Each service contains its own test suite:

```bash
# Run tests for noising service
cd noising-service
pytest

# Run tests for other services similarly
```

## License

MIT License

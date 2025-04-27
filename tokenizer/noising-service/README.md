# Noising Service

This microservice is responsible for implementing the forward noising process of the SEDD (Score Entropy Discrete Diffusion) framework for the proof generation system. The service provides an API to apply controlled noise to token sequences according to specific schedules and parameters.

## Key Features

- Implementation of SEDD's entropy-based noise schedule
- Support for hyperschedules (position-dependent noise levels)
- Simple REST API for integration with other services
- Built-in benchmarking against uniform noise schedules

## Getting Started

### Prerequisites

- Python 3.9+
- Pipenv (for dependency management)

### Installation

1. Clone the repository
2. Install dependencies:
   ```
   pipenv install
   ```
   
   For development:
   ```
   pipenv install --dev
   ```

### Running the service

#### Locally

```bash
pipenv run python -m src.main
```

#### With Docker

```bash
docker build -t noising-service .
docker run -p 8000:8000 noising-service
```

## API Documentation

The service exposes the following endpoints:

### GET /

Returns service status and version information.

### GET /info

Returns configuration details about the noising service.

**Response:**
```json
{
  "vocab_size": 50257,
  "num_timesteps": 1000,
  "schedule_type": "cosine",
  "beta_min": 0.0001,
  "beta_max": 0.02
}
```

### GET /alpha/{timestep}

Get the alpha value (noise level) for a specific timestep.

**Parameters:**
- `timestep`: An integer between 0 and num_timesteps-1

**Response:**
```json
{
  "timestep": 500,
  "alpha": 0.25
}
```

### POST /noise

Apply noise to a token sequence.

**Request Body:**
```json
{
  "tokens": [101, 2054, 2003, 2048, 2010, 102],
  "timestep": 500,
  "use_hyperschedule": false,
  "token_weights": null
}
```

**Response:**
```json
{
  "noised_tokens": [101, 2054, 3045, 2048, 8965, 102],
  "alpha": 0.25,
  "metrics": {
    "unchanged_tokens": 4,
    "percent_unchanged": 66.67,
    "total_tokens": 6
  }
}
```

## Configuration

The service can be configured using environment variables:

- `VOCAB_SIZE`: Size of the token vocabulary (default: 50257)
- `NUM_TIMESTEPS`: Number of diffusion steps (default: 1000)
- `SCHEDULE_TYPE`: Type of noise schedule ('linear', 'cosine', 'quadratic', or 'sedd_entropy', default: 'cosine')
- `PORT`: Service port (default: 8000)
- `HOST`: Service host (default: '0.0.0.0')

## Testing

Run tests with pytest:

```bash
pipenv run pytest
```

## Benchmarking

Compare SEDD vs uniform noise schedule:

```bash
pipenv run python -m benchmarks.run_benchmarks
```

Results will be saved as PNG plots and a JSON file with benchmark data.

## Integration with Other Services

### Integration with Tokenizer Service

The noising service expects token IDs from the tokenizer service in the following format:

```json
{
  "tokens": [101, 2054, 2003, 2048, 2010, 102],
  "token_types": {
    "critical": [101, 102, 2003],
    "variable": [2054, 2048],
    "other": [2010]
  }
}
```

When integrating with the tokenizer service:
1. The tokenizer service should expose a `/tokenize` endpoint that returns tokens
2. Your application should fetch these tokens and forward them to the noising service's `/noise` endpoint

### Integration with Denoiser Service

The noising service provides data to the denoiser service in the following format:

```json
{
  "noised_tokens": [101, 2054, 3045, 2048, 8965, 102],
  "alpha": 0.25,
  "metrics": {
    "unchanged_tokens": 4,
    "percent_unchanged": 66.67,
    "total_tokens": 6
  }
}
```

When integrating with the denoiser service:
1. The denoiser service expects a POST request to its `/denoise` endpoint
2. Forward the `noised_tokens` and `alpha` values from the noising service response
3. The denoiser service will return the denoised tokens

## License

[MIT License](LICENSE)

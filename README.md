# 🧠 MathProof AI: Diffusion-Powered Mathematical Proof Generation

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python 3.11+"/>
  <img src="https://img.shields.io/badge/Framework-FastAPI-009688.svg" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/Frontend-Next.js-black.svg" alt="Next.js"/>
  <img src="https://img.shields.io/badge/LLM-Claude-blueviolet.svg" alt="Claude"/>
  <img src="https://img.shields.io/badge/Hackathon-Pear_VC_×_Anthropic-orange.svg" alt="Pear VC × Anthropic Hackathon"/>
</div>

## 🔎 Overview

This Platform is a cutting-edge system that generates rigorous mathematical proofs using a diffusion-like process with Claude's reasoning capabilities. This project was built for the Pear VC × Anthropic Hackathon and demonstrates an innovative approach to mathematical reasoning.

The system works by:
1. Taking a theorem statement as input
2. Applying controlled noise to create a corrupted version
3. Using Claude to denoise and reconstruct the original statement
4. Retrieving relevant mathematical lemmas using Pinecone and OpenAI embeddings
5. Generating a structured proof with Claude
6. Validating proofs with Z3 theorem prover (test integration)

## 🌟 Key Features

- **Text-Based Diffusion Process**: Applies controlled noise to mathematical statements
- **Claude API Integration**: Leverages Claude for both denoising and proof generation
- **Vector Search**: Uses OpenAI embeddings and Pinecone for retrieving relevant mathematical lemmas
- **Z3 Theorem Prover**: Validates generated proofs for mathematical correctness
- **Modern React Frontend**: Clean interface built with Next.js and Tailwind CSS
- **Fast API Backend**: High-performance Python backend with FastAPI

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- Anthropic API Key
- OpenAI API Key (for embeddings)
- Pinecone API Key (for vector search)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Pear-VC-x-Anthropic-Hackathon
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file with the following variables:
   ```
   CLAUDE_API_KEY=your_claude_api_key
   OPENAI_API_KEY=your_openai_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_ENVIRONMENT=your_pinecone_region
   PINECONE_INDEX=metamath-theorems
   ```

4. **Install frontend dependencies**
   ```bash
   cd frontend
   npm install
   ```

### Running the Application

1. **Start the backend**
   ```bash
   cd Pear-VC-x-Anthropic-Hackathon
   python main.py
   ```

2. **Start the frontend**
   ```bash
   cd frontend
   npm run dev
   ```

3. **Access the application**
   Open your browser and navigate to http://localhost:3000

## 🧩 System Architecture

The system consists of two main components:

1. **FastAPI Backend**
   - Provides a unified REST API for proof generation
   - Integrates with Claude API for denoising and proof generation
   - Connects to Pinecone for vector search using OpenAI embeddings
   - Implements a text-based diffusion model with controllable noise levels

2. **Next.js Frontend**
   - Provides a modern, responsive UI with React
   - Communicates with the backend API
   - Displays theorem inputs, proof outputs, and verification status

## 📝 API Endpoints

### Main Endpoint

```
POST /noise_and_denoise
```

**Request Body:**
```json
{
  "clean_text": "Prove that the sum of two even numbers is even",
  "noise_level": 0.1,
  "top_k": 5
}
```

**Response:**
```json
{
  "formatted_proof": "Theorem: The sum of two even numbers is even.\n\nProof:\n1. Let x and y be even numbers.\n2. By definition, x = 2a and y = 2b for some integers a and b.\n3. Then x + y = 2a + 2b = 2(a + b).\n4. Since a + b is an integer, x + y is divisible by 2.\n5. Therefore, x + y is even."
}
```

## 🧪 Testing

Run the integration tests to verify components are working correctly:

```bash
python test_translation_pipeline.py
```

## 🔧 Technical Details

- **Text Diffusion**: Uses character-level noise with configurable noise levels
- **Vector Retrieval**: OpenAI text-embedding-3-small model with Pinecone vector database
- **Claude Integration**: Uses Claude API for both denoising and proof generation
- **Validation**: Z3 theorem prover with Python bindings for proof verification
- **Frontend**: React with Next.js, styled using Tailwind CSS and shadcn/ui components

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details. 

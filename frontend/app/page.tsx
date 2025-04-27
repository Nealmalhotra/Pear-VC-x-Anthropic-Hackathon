import { Suspense } from "react"
import ProofGenerator from "@/components/proof-generator"
import { LoadingProofGenerator } from "@/components/loading-proof-generator"

export default function Home() {
  return (
    <main className="min-h-screen bg-gradient-to-b from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-950">
      <div className="container mx-auto px-4 py-8">
        <header className="mb-8 text-center">
          <h1 className="text-4xl font-bold tracking-tight text-gray-900 dark:text-gray-50 sm:text-5xl">
            Discrete Diffusion Proof Generator
          </h1>
          <p className="mt-3 text-lg text-gray-600 dark:text-gray-400">
            Generate human-like mathematical proofs using discrete diffusion and Claude API
          </p>
        </header>

        <Suspense fallback={<LoadingProofGenerator />}>
          <ProofGenerator />
        </Suspense>
      </div>
    </main>
  )
}

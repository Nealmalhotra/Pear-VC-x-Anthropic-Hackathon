import { Skeleton } from "@/components/ui/skeleton"

interface ProofStepsProps {
  steps: {
    id: number
    content: string
    isComplete: boolean
  }[]
  isLoading: boolean
}

export function ProofSteps({ steps, isLoading }: ProofStepsProps) {
  if (isLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-6 w-full" />
        <Skeleton className="h-6 w-5/6" />
        <Skeleton className="h-6 w-4/6" />
      </div>
    )
  }

  if (steps.length === 0) {
    return (
      <div className="py-8 text-center text-gray-500">
        <p>No proof steps generated yet</p>
      </div>
    )
  }

  return (
    <div className="space-y-3">
      {steps.map((step) => (
        <div
          key={step.id}
          className="rounded-md border border-gray-200 bg-white p-3 shadow-sm dark:border-gray-800 dark:bg-gray-950"
        >
          <p className="text-gray-800 dark:text-gray-200">{step.content}</p>
        </div>
      ))}
    </div>
  )
}

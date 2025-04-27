import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { RefreshCwIcon } from "lucide-react"

interface DiffusionVisualizerProps {
  currentStep: number
  totalSteps: number
  isGenerating: boolean
}

export function DiffusionVisualizer({ currentStep, totalSteps, isGenerating }: DiffusionVisualizerProps) {
  const progress = Math.min(100, (currentStep / totalSteps) * 100)

  const getStepName = (step: number) => {
    if (step === 0) return "Initializing"
    if (step === 1) return "Context Retrieval"
    return `Denoising Step ${step - 1}`
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {isGenerating && <RefreshCwIcon className="h-4 w-4 animate-spin" />}
          <span className="font-medium">{getStepName(currentStep)}</span>
        </div>
        <Badge variant="outline">
          Step {currentStep}/{totalSteps}
        </Badge>
      </div>

      <Progress value={progress} className="h-2 w-full" />

      <div className="grid grid-cols-3 gap-2 text-center text-xs">
        <div
          className={`rounded-md p-1 ${currentStep >= 1 ? "bg-gray-200 dark:bg-gray-800" : "bg-gray-100 dark:bg-gray-900"}`}
        >
          Retrieval
        </div>
        <div
          className={`rounded-md p-1 ${currentStep >= 2 ? "bg-gray-200 dark:bg-gray-800" : "bg-gray-100 dark:bg-gray-900"}`}
        >
          Diffusion
        </div>
        <div
          className={`rounded-md p-1 ${currentStep >= totalSteps ? "bg-gray-200 dark:bg-gray-800" : "bg-gray-100 dark:bg-gray-900"}`}
        >
          Verification
        </div>
      </div>

      <div className="mt-4 rounded-md bg-gray-50 p-3 text-sm dark:bg-gray-900">
        <p className="font-medium">Diffusion Parameters:</p>
        <div className="mt-1 grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
          <div>Noise Schedule: SEDD</div>
          <div>Timesteps: {totalSteps - 2}</div>
          <div>Backbone: S4</div>
          <div>Denoiser: Claude API</div>
        </div>
      </div>
    </div>
  )
}

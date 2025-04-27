import { CheckCircle2Icon, XCircleIcon, AlertCircleIcon } from "lucide-react"
import { Badge } from "@/components/ui/badge"

interface VerificationStatusProps {
  isVerified: boolean | null
}

export function VerificationStatus({ isVerified }: VerificationStatusProps) {
  if (isVerified === null) {
    return (
      <Badge variant="outline" className="flex items-center gap-1">
        <AlertCircleIcon className="h-4 w-4 text-gray-500" />
        <span>Pending Verification</span>
      </Badge>
    )
  }

  if (isVerified) {
    return (
      <Badge
        variant="outline"
        className="flex items-center gap-1 bg-green-50 text-green-700 dark:bg-green-950 dark:text-green-300"
      >
        <CheckCircle2Icon className="h-4 w-4" />
        <span>Verified</span>
      </Badge>
    )
  }

  return (
    <Badge
      variant="outline"
      className="flex items-center gap-1 bg-red-50 text-red-700 dark:bg-red-950 dark:text-red-300"
    >
      <XCircleIcon className="h-4 w-4" />
      <span>Verification Failed</span>
    </Badge>
  )
}

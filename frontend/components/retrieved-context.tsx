import { Skeleton } from "@/components/ui/skeleton"
import { ScrollArea } from "@/components/ui/scroll-area"

interface RetrievedContextProps {
  items: {
    id: number
    title: string
    content: string
    source: string
  }[]
  isLoading: boolean
}

export function RetrievedContext({ items, isLoading }: RetrievedContextProps) {
  if (isLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-20 w-full" />
        <Skeleton className="h-20 w-full" />
      </div>
    )
  }

  if (items.length === 0) {
    return (
      <div className="py-8 text-center text-gray-500">
        <p>No context retrieved yet</p>
      </div>
    )
  }

  return (
    <ScrollArea className="h-[220px]">
      <div className="space-y-4">
        {items.map((item) => (
          <div
            key={item.id}
            className="rounded-md border border-gray-200 bg-white p-3 shadow-sm dark:border-gray-800 dark:bg-gray-950"
          >
            <h4 className="font-medium text-gray-900 dark:text-gray-100">{item.title}</h4>
            <p className="mt-1 text-sm text-gray-700 dark:text-gray-300">{item.content}</p>
            <p className="mt-2 text-xs text-gray-500">Source: {item.source}</p>
          </div>
        ))}
      </div>
    </ScrollArea>
  )
}

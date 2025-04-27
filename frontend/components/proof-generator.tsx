"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Separator } from "@/components/ui/separator"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { InfoIcon, RefreshCwIcon, ArrowRightIcon, AlertCircleIcon } from "lucide-react"
import { VerificationStatus } from "@/components/verification-status"

export default function ProofGenerator() {
  const [theorem, setTheorem] = useState("")
  const [isGenerating, setIsGenerating] = useState(false)
  const [formattedProof, setFormattedProof] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState("input")

  const handleGenerate = async () => {
    if (!theorem.trim()) return

    setIsGenerating(true)
    setFormattedProof(null)
    setError(null)
    setActiveTab("result")

    const noiseLevel = 0.1
    const topK = 5

    try {
      const response = await fetch("http://localhost:8000/noise_and_denoise", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          clean_text: theorem,
          noise_level: noiseLevel,
          top_k: topK,
        }),
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      setFormattedProof(data.formatted_proof)

    } catch (err: any) {
      console.error("API Call Failed:", err)
      setError(err.message || "Failed to generate proof. Please check the backend service.")
    } finally {
      setIsGenerating(false)
    }
  }

  const handleReset = () => {
    setTheorem("")
    setIsGenerating(false)
    setFormattedProof(null)
    setError(null)
    setActiveTab("input")
  }

  return (
    <div className="space-y-6">
      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="input">Input</TabsTrigger>
          <TabsTrigger value="result" disabled={!formattedProof && !isGenerating && !error}>
            Result
          </TabsTrigger>
        </TabsList>

        <TabsContent value="input" className="space-y-4 pt-4">
          <Card>
            <CardHeader>
              <CardTitle>Enter Your Theorem</CardTitle>
              <CardDescription>
                State the theorem you want to prove in natural language or simple mathematical notation.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Textarea
                placeholder="e.g., Prove that the sum of two even numbers is even"
                value={theorem}
                onChange={(e) => setTheorem(e.target.value)}
                className="min-h-[120px]"
                disabled={isGenerating}
              />
            </CardContent>
            <CardFooter className="flex justify-between">
              <Button variant="outline" onClick={handleReset} disabled={isGenerating}>
                Clear
              </Button>
              <Button onClick={handleGenerate} disabled={!theorem.trim() || isGenerating} className="gap-2">
                {isGenerating ? (
                  <>
                    <RefreshCwIcon className="h-4 w-4 animate-spin" />
                    Generating...
                  </>
                ) : (
                  <>
                    <ArrowRightIcon className="h-4 w-4" />
                    Generate Proof
                  </>
                )}
              </Button>
            </CardFooter>
          </Card>

          <Alert>
            <InfoIcon className="h-4 w-4" />
            <AlertTitle>Proof Generation</AlertTitle>
            <AlertDescription>
              Enter a theorem statement. The system will attempt to retrieve relevant context, denoise the statement if needed (using a fixed noise level for now), and generate a proof using the Claude API.
            </AlertDescription>
          </Alert>
        </TabsContent>

        <TabsContent value="result" className="space-y-6 pt-4">
          {isGenerating && (
            <Card>
              <CardHeader>
                <CardTitle>Generating Proof...</CardTitle>
              </CardHeader>
              <CardContent className="flex items-center justify-center py-12">
                <RefreshCwIcon className="h-8 w-8 animate-spin text-gray-500" />
              </CardContent>
            </Card>
          )}

          {error && !isGenerating && (
            <Alert variant="destructive">
              <AlertCircleIcon className="h-4 w-4" />
              <AlertTitle>Error Generating Proof</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {!isGenerating && formattedProof && (
             <Card>
               <CardHeader className="flex flex-row items-center justify-between">
                 <div>
                   <CardTitle>Generated Proof</CardTitle>
                   <CardDescription>Review the generated mathematical proof below.</CardDescription>
                 </div>
               </CardHeader>
               <CardContent>
                 <div className="rounded-md bg-gray-50 p-4 dark:bg-gray-900 whitespace-pre-wrap">
                   {formattedProof}
                 </div>
               </CardContent>
               <CardFooter className="flex justify-end">
                 <Button variant="outline" onClick={handleReset}>
                   New Proof
                 </Button>
               </CardFooter>
             </Card>
           )}

          {!isGenerating && !formattedProof && !error && activeTab === 'result' && (
            <Card>
              <CardContent className="pt-6 text-center text-gray-500">
                Enter a theorem and click "Generate Proof" to see the result here.
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>
    </div>
  )
}

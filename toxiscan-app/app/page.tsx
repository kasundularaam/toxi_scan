"use client";

import type React from "react";
import { useEffect, useMemo, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Copy, X, Loader2, Upload, ImageIcon, Gauge } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

const API_BASE =
  process.env.NEXT_PUBLIC_TOXISCAN_API ?? "http://localhost:8000";

interface MatchItem {
  label: string;
  match: string;
  start: number;
  end: number;
  // AI mode extras:
  score?: number; // 0..1
  normalized?: string;
}

interface AnalyzeResponseWire {
  source: "text" | "image";
  raw_text: string;
  tagged_text: string;
  matches: MatchItem[];
}

interface AnalysisResult {
  text: string; // tagged_text from backend (contains <cuss>...</cuss>)
  detectedTerms: number;
  matches: MatchItem[];
}

export default function ToxiScanPage() {
  const [activeTab, setActiveTab] = useState<"text" | "image">("text");
  const [textInput, setTextInput] = useState("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [health, setHealth] = useState<{
    detector?: string;
    ai_threshold?: number;
    ai_loaded?: boolean;
  } | null>(null);
  const [threshold, setThreshold] = useState<number | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const { toast } = useToast();

  // fetch health on mount (to get default threshold)
  useEffect(() => {
    (async () => {
      try {
        const r = await fetch(`${API_BASE}/health`);
        const h = await r.json().catch(() => null);
        setHealth(h || null);
        if (h?.ai_threshold != null) {
          setThreshold(Number(h.ai_threshold));
        } else {
          setThreshold(0.75);
        }
      } catch {
        // fallback
        setThreshold(0.75);
      }
    })();
  }, []);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && file.type.match(/^image\/(png|jpe?g|webp|heic|heif)$/i)) {
      setSelectedFile(file);
      const reader = new FileReader();
      reader.onload = (e) => setImagePreview(e.target?.result as string);
      reader.readAsDataURL(file);
    }
  };

  const parseResultText = (text: string) => {
    const parts = text.split(/(<cuss>.*?<\/cuss>)/g);
    return parts
      .map((part, index) => {
        if (part.startsWith("<cuss>") && part.endsWith("</cuss>")) {
          const cleanText = part.replace(/<\/?cuss>/g, "");
          return (
            <span key={index} className="bg-yellow-200 font-bold px-1 rounded">
              {cleanText}
            </span>
          );
        }
        return <span key={index}>{part}</span>;
      })
      .filter((node) => node !== "");
  };

  const overallConfidence = useMemo(() => {
    if (!result || !result.matches?.length) return null;
    const withScores = result.matches.filter(
      (m) => typeof m.score === "number"
    );
    if (!withScores.length) return null;
    const avg =
      withScores.reduce((acc, m) => acc + (m.score || 0), 0) /
      withScores.length;
    return avg; // 0..1
  }, [result]);

  const handleAnalyze = async () => {
    setIsAnalyzing(true);
    setError(null);
    setResult(null);

    try {
      let url = "";
      let fetchOptions: RequestInit = { method: "POST" };

      if (activeTab === "text") {
        if (!textInput.trim()) throw new Error("Please enter some text.");
        url = `${API_BASE}/analyze/text`;
        fetchOptions.headers = { "Content-Type": "application/json" };
        const body: any = { text: textInput };
        if (threshold != null) body.threshold = threshold;
        fetchOptions.body = JSON.stringify(body);
      } else {
        if (!selectedFile) throw new Error("Please choose an image.");
        url = `${API_BASE}/analyze/image`;
        const form = new FormData();
        form.append("image", selectedFile);
        fetchOptions.body = form; // let browser set Content-Type with boundary
      }

      const response = await fetch(url, fetchOptions);

      if (!response.ok) {
        const maybeJson = await response.json().catch(() => null);
        const msg =
          (maybeJson && (maybeJson.detail || maybeJson.message)) ||
          `Analysis failed (${response.status})`;
        throw new Error(msg);
      }

      const data: AnalyzeResponseWire = await response.json();

      const detectedTerms =
        Array.isArray(data.matches) && data.matches.length > 0
          ? data.matches.length
          : (data.tagged_text.match(/<cuss>/g) || []).length;

      setResult({
        text: data.tagged_text,
        detectedTerms,
        matches: data.matches ?? [],
      });
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Analysis failed. Please try again."
      );
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleCopy = () => {
    if (result) {
      const cleanText = result.text.replace(/<\/?cuss>/g, "");
      navigator.clipboard.writeText(cleanText);
      toast({ description: "Text copied to clipboard" });
    }
  };

  const handleClear = () => {
    setResult(null);
    setError(null);
    setTextInput("");
    setSelectedFile(null);
    setImagePreview(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const canAnalyze =
    activeTab === "text" ? textInput.trim().length > 0 : !!selectedFile;

  return (
    <div className="min-h-screen bg-background p-4">
      <div className="max-w-2xl mx-auto space-y-6">
        {/* Header */}
        <header className="text-center">
          <h1 className="text-3xl font-bold text-foreground">ToxiScan</h1>
          <p className="text-muted-foreground mt-2">
            Hate speech detection for Sinhala, English, and Singlish
          </p>
          {health && (
            <div className="mt-2 text-xs text-muted-foreground">
              Mode:{" "}
              <span className="font-medium">{health.detector ?? "ai"}</span> •
              Default threshold:{" "}
              <span className="font-medium">
                {health.ai_threshold != null
                  ? Number(health.ai_threshold).toFixed(3)
                  : "—"}
              </span>{" "}
              • Model: {health.ai_loaded ? "loaded ✅" : "not loaded ❌"}
            </div>
          )}
        </header>

        {/* Error Banner */}
        {error && (
          <Alert variant="destructive" className="relative">
            <AlertDescription className="pr-8">{error}</AlertDescription>
            <Button
              variant="ghost"
              size="sm"
              className="absolute right-2 top-2 h-6 w-6 p-0"
              onClick={() => setError(null)}
            >
              <X className="h-4 w-4" />
            </Button>
          </Alert>
        )}

        {/* Threshold Control */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Model Threshold</CardTitle>
            <p className="text-xs text-muted-foreground">
              Higher = stricter (more precision, less recall)
            </p>
          </CardHeader>
          <CardContent className="pt-0">
            <div className="flex items-center gap-3">
              <input
                type="range"
                min={0.3}
                max={0.95}
                step={0.005}
                value={threshold ?? 0.75}
                onChange={(e) => setThreshold(parseFloat(e.target.value))}
                className="w-full"
                aria-label="Threshold slider"
              />
              <Badge variant="secondary">
                {threshold != null ? threshold.toFixed(3) : "0.750"}
              </Badge>
            </div>
          </CardContent>
        </Card>

        {/* Main Interface */}
        <Card>
          <CardContent className="p-6">
            <Tabs
              value={activeTab}
              onValueChange={(v) => setActiveTab(v as any)}
            >
              <TabsList className="grid w-full grid-cols-2">
                <TabsTrigger value="text">Text</TabsTrigger>
                <TabsTrigger value="image">Image</TabsTrigger>
              </TabsList>

              <TabsContent value="text" className="space-y-4">
                <Textarea
                  placeholder="Type or paste text…"
                  value={textInput}
                  onChange={(e) => setTextInput(e.target.value)}
                  className="min-h-32 resize-none"
                  style={{
                    fontFamily:
                      "var(--font-sinhala), var(--font-sans), system-ui, sans-serif",
                  }}
                />
              </TabsContent>

              <TabsContent value="image" className="space-y-4">
                <div className="space-y-4">
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".png,.jpg,.jpeg,.webp,.heic,.heif"
                    onChange={handleFileSelect}
                    className="hidden"
                  />
                  <Button
                    variant="outline"
                    onClick={() => fileInputRef.current?.click()}
                    className="w-full h-32 border-dashed"
                  >
                    <div className="flex flex-col items-center gap-2">
                      <Upload className="h-8 w-8 text-muted-foreground" />
                      <span>Choose image file</span>
                      <span className="text-sm text-muted-foreground">
                        PNG, JPG, JPEG, WEBP, HEIC
                      </span>
                    </div>
                  </Button>

                  {imagePreview && (
                    <div className="flex items-center gap-3 p-3 border rounded-lg">
                      <ImageIcon className="h-5 w-5 text-muted-foreground" />
                      <img
                        src={imagePreview || "/placeholder.svg"}
                        alt="Selected image"
                        className="h-12 w-12 object-cover rounded"
                      />
                      <span className="text-sm text-muted-foreground flex-1">
                        {selectedFile?.name}
                      </span>
                    </div>
                  )}
                </div>
              </TabsContent>
            </Tabs>

            <Button
              onClick={handleAnalyze}
              disabled={!canAnalyze || isAnalyzing}
              className="w-full mt-4"
            >
              {isAnalyzing && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              Analyze
            </Button>
          </CardContent>
        </Card>

        {/* Results */}
        {result && (
          <Card>
            <CardHeader>
              <CardTitle>Results</CardTitle>
              <p className="text-sm text-muted-foreground">
                {activeTab === "text"
                  ? "Input Text (tagged)"
                  : "Extracted Text from Image (tagged)"}
              </p>
            </CardHeader>

            <CardContent className="space-y-5">
              {/* Tagged text */}
              <div
                className="p-4 bg-muted rounded-lg text-sm leading-relaxed"
                style={{
                  fontFamily:
                    "var(--font-sinhala), var(--font-sans), system-ui, sans-serif",
                }}
                aria-live="polite"
              >
                {parseResultText(result.text)}
              </div>

              {/* Summary row */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <span className="text-sm text-muted-foreground">
                    Detected terms: {result.detectedTerms}
                  </span>
                  <Badge
                    variant={
                      result.detectedTerms === 0 ? "secondary" : "destructive"
                    }
                  >
                    {result.detectedTerms === 0
                      ? "No hate speech found"
                      : "Hate terms found"}
                  </Badge>
                </div>

                <div className="flex gap-2">
                  <Button variant="outline" size="sm" onClick={handleCopy}>
                    <Copy className="h-4 w-4 mr-1" />
                    Copy
                  </Button>
                  <Button variant="outline" size="sm" onClick={handleClear}>
                    Clear
                  </Button>
                </div>
              </div>

              {/* Confidence Overview */}
              {overallConfidence != null && (
                <div className="border rounded-lg p-3">
                  <div className="flex items-center gap-2 text-sm mb-2">
                    <Gauge className="h-4 w-4" />
                    <span className="font-medium">Overall confidence</span>
                    <span className="text-muted-foreground">
                      (avg of matched tokens)
                    </span>
                  </div>
                  <div className="h-2 w-full bg-secondary rounded">
                    <div
                      className="h-2 rounded"
                      style={{
                        width: `${Math.round(overallConfidence * 100)}%`,
                        background:
                          "linear-gradient(90deg, var(--primary), var(--destructive))",
                      }}
                    />
                  </div>
                  <div className="mt-1 text-xs text-muted-foreground">
                    {Math.round(overallConfidence * 100)}%
                  </div>
                </div>
              )}

              {/* Per-token table */}
              {result.matches?.length > 0 && (
                <div className="text-sm">
                  <div className="mb-2 font-medium">Matched tokens</div>
                  <div className="rounded-md border overflow-hidden">
                    <table className="w-full text-sm">
                      <thead className="bg-muted/50">
                        <tr>
                          <th className="text-left p-2">Token</th>
                          <th className="text-left p-2">Normalized</th>
                          <th className="text-left p-2">Confidence</th>
                          <th className="text-left p-2">Range</th>
                          <th className="text-left p-2">Label</th>
                        </tr>
                      </thead>
                      <tbody>
                        {result.matches
                          .slice()
                          .sort((a, b) => (b.score ?? 0) - (a.score ?? 0))
                          .map((m, i) => (
                            <tr key={i} className="border-t">
                              <td className="p-2 font-medium">{m.match}</td>
                              <td className="p-2 text-muted-foreground">
                                {m.normalized ?? "—"}
                              </td>
                              <td className="p-2">
                                {m.score != null
                                  ? `${Math.round(m.score * 100)}%`
                                  : "—"}
                              </td>
                              <td className="p-2 text-muted-foreground">
                                {m.start}–{m.end}
                              </td>
                              <td className="p-2">
                                <Badge variant="destructive">{m.label}</Badge>
                              </td>
                            </tr>
                          ))}
                      </tbody>
                    </table>
                  </div>
                  <p className="mt-2 text-xs text-muted-foreground">
                    Confidence is the model’s probability for each token (after
                    threshold {threshold?.toFixed(3)}).
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}

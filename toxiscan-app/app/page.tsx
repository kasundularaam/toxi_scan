"use client";

import type React from "react";
import { useState, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Copy, X, Loader2, Upload, ImageIcon } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

const API_BASE =
  process.env.NEXT_PUBLIC_TOXISCAN_API ?? "http://localhost:8000";

interface MatchItem {
  label: string;
  match: string;
  start: number;
  end: number;
}

interface AnalysisResult {
  text: string; // tagged_text from backend (contains <cuss>...</cuss>)
  detectedTerms: number; // matches.length (fallback to counting <cuss> tags)
}

export default function ToxiScanPage() {
  const [activeTab, setActiveTab] = useState("text");
  const [textInput, setTextInput] = useState("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { toast } = useToast();

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
        fetchOptions.body = JSON.stringify({ text: textInput });
      } else {
        if (!selectedFile) throw new Error("Please choose an image.");
        url = `${API_BASE}/analyze/image`;
        const form = new FormData();
        form.append("image", selectedFile);
        fetchOptions.body = form; // let the browser set Content-Type with boundary
      }

      const response = await fetch(url, fetchOptions);

      // Try to parse error shape from FastAPI
      if (!response.ok) {
        const maybeJson = await response.json().catch(() => null);
        const msg =
          (maybeJson && (maybeJson.detail || maybeJson.message)) ||
          `Analysis failed (${response.status})`;
        throw new Error(msg);
      }

      const data: {
        source: "text" | "image";
        raw_text: string;
        tagged_text: string;
        matches: MatchItem[];
      } = await response.json();

      const detectedTerms =
        Array.isArray(data.matches) && data.matches.length > 0
          ? data.matches.length
          : (data.tagged_text.match(/<cuss>/g) || []).length;

      setResult({
        text: data.tagged_text, // show tagged version with <cuss> markers
        detectedTerms,
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
      // Copy without tags
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

        {/* Main Interface */}
        <Card>
          <CardContent className="p-6">
            <Tabs value={activeTab} onValueChange={setActiveTab}>
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
            <CardContent className="space-y-4">
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
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}

import { type NextRequest, NextResponse } from "next/server"

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { mode, payload } = body

    if (!mode || !payload) {
      return NextResponse.json({ message: "Missing required fields: mode and payload" }, { status: 400 })
    }

    if (mode !== "text" && mode !== "image") {
      return NextResponse.json({ message: 'Invalid mode. Must be "text" or "image"' }, { status: 400 })
    }

    // Mock response for demonstration
    // In a real implementation, this would call your hate speech detection service
    const mockResponses = [
      "This is a normal message with no issues.",
      "Ane <cuss>Huththo</cuss> Mama Nadun.",
      "මේක <cuss>නරක</cuss> වචනයක්.",
      "This text contains <cuss>bad</cuss> words and <cuss>hate</cuss> speech.",
      "Clean text with no problems detected.",
    ]

    const randomResponse = mockResponses[Math.floor(Math.random() * mockResponses.length)]

    // Simulate processing delay
    await new Promise((resolve) => setTimeout(resolve, 1000 + Math.random() * 2000))

    return NextResponse.json({
      text: randomResponse,
      mode,
    })
  } catch (error) {
    console.error("Analysis error:", error)
    return NextResponse.json({ message: "Internal server error during analysis" }, { status: 500 })
  }
}

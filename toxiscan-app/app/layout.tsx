import type React from "react";
import type { Metadata } from "next";
import { GeistSans } from "geist/font/sans";
import { GeistMono } from "geist/font/mono";
import { Noto_Sans_Sinhala } from "next/font/google";
import "./globals.css";

const notoSansSinhala = Noto_Sans_Sinhala({
  subsets: ["latin", "sinhala"],
  display: "swap",
  variable: "--font-sinhala",
});

export const metadata: Metadata = {
  title: "ToxiScan - Hate Speech Detection",
  description:
    "Detect hate speech in Sinhala, English, and Singlish text and images",
  generator: "v0.app",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <style>{`
html {
  font-family: ${GeistSans.style.fontFamily}, ${notoSansSinhala.style.fontFamily};
  --font-sans: ${GeistSans.variable};
  --font-mono: ${GeistMono.variable};
  --font-sinhala: ${notoSansSinhala.variable};
}
        `}</style>
      </head>
      <body>{children}</body>
    </html>
  );
}

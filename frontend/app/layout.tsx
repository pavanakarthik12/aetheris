import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "Aetheris Chat",
  description: "Phase 1 chat interface for the Aetheris Qwen communication layer.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
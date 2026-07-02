import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "Aetheris",
  description: "Project foundation for the Aetheris cognitive system.",
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
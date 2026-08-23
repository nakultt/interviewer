import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "KEC Local Interviewer",
  description: "A privacy-first, local mock interview console",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

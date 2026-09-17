import type { Metadata } from "next";
import {cookies} from "next/headers";
import "./globals.css";
import {AccessGate} from "@/components/access-gate";
import {ACCESS_COOKIE,accessIsConfigured,validAccessSession} from "@/lib/access";

export const dynamic = "force-dynamic";

export const metadata: Metadata = {
  title: "Cross-species Transport Atlas",
  description: "Explore mouse–human transport plans, coclusters, and filtered STRING physical networks across rho values.",
  other: {
    "codex-preview": "development",
  },
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
};

export default async function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const configured=accessIsConfigured(),cookieStore=await cookies(),authorized=configured&&await validAccessSession(cookieStore.get(ACCESS_COOKIE)?.value);
  return (
    <html lang="en">
      <body className="antialiased">{authorized?children:<AccessGate configurationError={!configured}/>}</body>
    </html>
  );
}

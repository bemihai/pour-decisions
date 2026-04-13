import type { Metadata } from "next";
import { Poppins, Geist_Mono } from "next/font/google";
import { Toaster } from "sonner";
import "./globals.css";
import Navigation from "@/components/Navigation";
import QueryProvider from "@/components/QueryProvider";

const poppins = Poppins({
  variable: "--font-poppins",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  display: "swap",
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: {
    default: "Pour Decisions",
    template: "%s | Pour Decisions",
  },
  description: "RAG-powered wine assistant and cellar manager",
  metadataBase: new URL(
    process.env.NEXT_PUBLIC_BASE_URL ?? "http://localhost:3000",
  ),
  openGraph: {
    siteName: "Pour Decisions",
    type: "website",
    locale: "en_US",
  },
  twitter: {
    card: "summary_large_image",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${poppins.variable} ${geistMono.variable} h-full antialiased`}
      suppressHydrationWarning
    >
      <body className="min-h-full flex flex-col">
        {/* Skip-to-content link — first focusable element, visible only on focus.
            Satisfies WCAG 2.1 SC 2.4.1 (Bypass Blocks). */}
        <a
          href="#main-content"
          className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-[100] focus:rounded-md focus:bg-brand-burgundy focus:px-4 focus:py-2 focus:text-sm focus:font-medium focus:text-white focus:shadow-lg focus:outline-none"
        >
          Skip to content
        </a>

        <QueryProvider>
          <Navigation />
          <main id="main-content" className="flex-1">
            {children}
          </main>
        </QueryProvider>

        <Toaster
          position="bottom-right"
          toastOptions={{
            classNames: {
              toast: "font-sans",
            },
          }}
        />
      </body>
    </html>
  );
}



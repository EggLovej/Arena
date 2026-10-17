import type { Metadata } from "next";
import Link from "next/link";
import type { ReactNode } from "react";

import { ThemeToggle } from "@/components/ThemeToggle.tsx";

import "./globals.css";

export const metadata: Metadata = {
  title: { default: "Arena", template: "%s · Arena" },
  description: "Games for classic AI bots: watch them play each other and see who leads the ratings.",
};

// Applies a saved light or dark choice before the first paint.
const THEME_SCRIPT = `try{var t=localStorage.getItem("arena-theme");if(t==="light"||t==="dark")document.documentElement.setAttribute("data-theme",t)}catch(e){}`;

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_SCRIPT }} />
      </head>
      <body>
        <header className="site-header">
          <div className="wrap site-header-inner">
            <Link href="/" className="wordmark">
              <svg width="22" height="22" viewBox="0 0 22 22" aria-hidden="true">
                <circle cx="5" cy="17" r="3.5" />
                <circle cx="17" cy="17" r="3.5" />
                <circle cx="11" cy="5" r="3.5" />
                <path d="M5 17 L11 5 L17 17 Z" fill="none" strokeWidth="1.6" />
              </svg>
              Arena
            </Link>
            <nav className="site-nav">
              <Link href="/">Games</Link>
              <Link href="/about">About</Link>
              <ThemeToggle />
            </nav>
          </div>
        </header>
        <main className="wrap site-main">{children}</main>
        <footer className="site-footer">
          <div className="wrap">
            Bots play each other on GitHub Actions; the site shows the replays and ratings. Source:{" "}
            <a href="https://github.com/EggLovej/Arena">github.com/EggLovej/Arena</a>
          </div>
        </footer>
      </body>
    </html>
  );
}

"use client";

// Dark or light site colours. The site is dark until someone picks light, and
// the choice is remembered on this device.

import { useEffect, useState } from "react";

type Choice = "dark" | "light";

export function ThemeToggle() {
  const [choice, setChoice] = useState<Choice>("dark");

  useEffect(() => {
    if (document.documentElement.getAttribute("data-theme") === "light") setChoice("light");
  }, []);

  const change = () => {
    const next = choice === "dark" ? "light" : "dark";
    setChoice(next);
    if (next === "light") document.documentElement.setAttribute("data-theme", "light");
    else document.documentElement.removeAttribute("data-theme");
    try {
      if (next === "light") localStorage.setItem("arena-theme", "light");
      else localStorage.removeItem("arena-theme");
    } catch {
      // Not remembered, but still applied for this visit.
    }
  };

  const label = choice === "dark" ? "Switch to light colours" : "Switch to dark colours";
  return (
    <button type="button" className="theme-toggle" onClick={change} aria-label={label} title={label}>
      <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true">
        {choice === "dark" ? (
          <g fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round">
            <circle cx="9" cy="9" r="3.4" />
            <path d="M9 1.5v2M9 14.5v2M1.5 9h2M14.5 9h2M3.7 3.7l1.4 1.4M12.9 12.9l1.4 1.4M3.7 14.3l1.4-1.4M12.9 5.1l1.4-1.4" />
          </g>
        ) : (
          <path d="M14.5 11.2A6.5 6.5 0 0 1 6.8 3.5a6.5 6.5 0 1 0 7.7 7.7Z" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" />
        )}
      </svg>
    </button>
  );
}

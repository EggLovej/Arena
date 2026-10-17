"use client";

// Light or dark site colours. The choice is remembered on this device; until
// one is made, the site follows the system setting.

import { useEffect, useState } from "react";

type Choice = "light" | "dark" | "system";
const NEXT: Record<Choice, Choice> = { system: "light", light: "dark", dark: "system" };
const LABEL: Record<Choice, string> = { system: "Auto", light: "Light", dark: "Dark" };

export function ThemeToggle() {
  const [choice, setChoice] = useState<Choice>("system");

  useEffect(() => {
    try {
      const saved = localStorage.getItem("arena-theme");
      if (saved === "light" || saved === "dark") setChoice(saved);
    } catch {
      // Storage can be blocked; the system setting then decides.
    }
  }, []);

  const change = () => {
    const next = NEXT[choice];
    setChoice(next);
    if (next === "system") document.documentElement.removeAttribute("data-theme");
    else document.documentElement.setAttribute("data-theme", next);
    try {
      if (next === "system") localStorage.removeItem("arena-theme");
      else localStorage.setItem("arena-theme", next);
    } catch {
      // Not remembered, but still applied for this visit.
    }
  };

  return (
    <button type="button" className="theme-toggle" onClick={change} title="Switch between light, dark and the system setting">
      {LABEL[choice]}
    </button>
  );
}

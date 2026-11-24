import type React from "react";

/**
 * Smoothly scrolls to an in-page section anchor (`#section-id`), respecting
 * `prefers-reduced-motion` and allowing an optional callback (such as closing
 * a mobile menu drawer) before scrolling.
 */
export function scrollToSection(
  e: React.MouseEvent<HTMLAnchorElement>,
  href: string,
  beforeScroll?: () => void
) {
  if (!href.startsWith("#")) return;
  const targetId = href.slice(1);
  const target = document.getElementById(targetId);
  if (!target) return;

  e.preventDefault();
  beforeScroll?.();

  const prefersReduced =
    typeof window !== "undefined" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  window.requestAnimationFrame(() => {
    target.scrollIntoView({
      behavior: prefersReduced ? "auto" : "smooth",
      block: "start",
    });
    window.history.replaceState(null, "", href);
  });
}

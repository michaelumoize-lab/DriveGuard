import Image from "next/image";
import * as React from "react";
import { LANDING_MEDIA, type LandingMediaSlotKey } from "@/lib/landing-media";
import { cn } from "@/lib/utils";

interface LandingMediaSlotProps {
  slot: LandingMediaSlotKey;
  className?: string;
  imageClassName?: string;
  overlayClassName?: string;
  sizes?: string;
  /**
   * Elevate loading priority ONLY for the above-the-fold Hero LCP image.
   * Below-the-fold images use lazy loading without fetchPriority overrides.
   */
  isLcp?: boolean;
  /**
   * For images initially hidden inside inactive tabs/panels above or near the fold.
   */
  isHiddenPanel?: boolean;
  showSlotHint?: boolean;
  children?: React.ReactNode;
}

/**
 * Drop-in photographic / screenshot frame for the DriveGuard landing page.
 * Uses a dark optical camera canvas so computer-vision HUD overlays stay legible
 * in both Light Mode and Dark Mode, and switches seamlessly to Next.js `<Image>`
 * once `LANDING_MEDIA[slot].src` is populated.
 */
export function LandingMediaSlot({
  slot,
  className,
  imageClassName,
  overlayClassName,
  sizes = "(max-width: 768px) 100vw, 50vw",
  isLcp = false,
  isHiddenPanel = false,
  showSlotHint = true,
  children,
}: LandingMediaSlotProps) {
  const asset = LANDING_MEDIA[slot];

  return (
    <div
      className={cn(
        "relative overflow-hidden bg-[#121513] text-[#F5F3EC]",
        className
      )}
    >
      {/* Subtle optical camera sensor depth surface */}
      <div
        aria-hidden="true"
        className="absolute inset-0 bg-[radial-gradient(ellipse_75%_65%_at_50%_40%,rgba(245,243,236,0.06),rgba(16,18,17,0.92)_75%)]"
      />
      <div
        aria-hidden="true"
        className="absolute inset-0 opacity-[0.16] bg-[linear-gradient(to_right,rgba(245,243,236,0.07)_1px,transparent_1px),linear-gradient(to_bottom,rgba(245,243,236,0.07)_1px,transparent_1px)] bg-[size:32px_32px]"
      />

      {asset.src ? (
        <Image
          src={asset.src}
          alt={asset.alt}
          fill
          sizes={sizes}
          priority={isLcp}
          fetchPriority={isLcp ? "high" : isHiddenPanel ? "low" : undefined}
          loading={isLcp ? undefined : "lazy"}
          className={cn("object-cover", imageClassName)}
        />
      ) : (
        showSlotHint && (
          <div
            aria-hidden="true"
            className="pointer-events-none absolute bottom-3 right-3 z-10 inline-flex items-center gap-1.5 rounded-md border border-white/10 bg-[#101211]/80 px-2.5 py-1 text-[10px] font-mono tracking-wider text-[#9EA39B] backdrop-blur-sm"
          >
            <span className="size-1.5 rounded-full bg-[#E58B43]/80" />
            <span>media/{asset.placeholderFilename}</span>
          </div>
        )
      )}

      {/* Optional gradient scrim for contrast over photography */}
      {overlayClassName && (
        <div aria-hidden="true" className={cn("absolute inset-0", overlayClassName)} />
      )}

      {/* Foreground computer-vision overlays, HUD cards, or copy */}
      {children}
    </div>
  );
}

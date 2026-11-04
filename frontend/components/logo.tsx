import Image from "next/image";
import * as React from "react";
import { cn } from "@/lib/utils";

export const LOGO_BLACK_BG = "/logo-black-bg.png";
export const LOGO_WHITE_BG = "/logo-white-bg.png";
export const LOGO_WITHOUT_BG = "/logo-without-bg.png";

export interface LogoProps extends React.HTMLAttributes<HTMLDivElement> {
  size?: number;
  width?: number;
  height?: number;
  variant?: "theme" | "transparent" | "black" | "white";
  className?: string;
  imageClassName?: string;
  alt?: string;
  priority?: boolean;
}

/**
 * DriveGuard Brand Logo Component
 *
 * Automatically renders brand logo assets based on the active theme:
 * - Light Mode: /logo-white-bg.png
 * - Dark Mode:  /logo-black-bg.png
 * - Transparent: /logo-without-bg.png
 *
 * Built with Next.js Image and zero-flicker CSS theme switching.
 */
export function Logo({
  size = 32,
  width,
  height,
  variant = "theme",
  className = "size-8",
  imageClassName,
  alt = "DriveGuard Logo",
  priority = true,
  ...props
}: LogoProps) {
  const finalWidth = width || size;
  const finalHeight = height || size;

  if (variant === "transparent") {
    return (
      <div
        className={cn(
          "relative flex aspect-square items-center justify-center rounded-lg overflow-hidden shrink-0 shadow-xs",
          className
        )}
        {...props}
      >
        <Image
          src={LOGO_WITHOUT_BG}
          alt={alt}
          width={finalWidth}
          height={finalHeight}
          className={cn("w-full h-full object-contain", imageClassName)}
          priority={priority}
        />
      </div>
    );
  }

  if (variant === "black") {
    return (
      <div
        className={cn(
          "relative flex aspect-square items-center justify-center rounded-lg overflow-hidden shrink-0 shadow-xs",
          className
        )}
        {...props}
      >
        <Image
          src={LOGO_BLACK_BG}
          alt={alt}
          width={finalWidth}
          height={finalHeight}
          className={cn("w-full h-full object-contain", imageClassName)}
          priority={priority}
        />
      </div>
    );
  }

  if (variant === "white") {
    return (
      <div
        className={cn(
          "relative flex aspect-square items-center justify-center rounded-lg overflow-hidden shrink-0 shadow-xs",
          className
        )}
        {...props}
      >
        <Image
          src={LOGO_WHITE_BG}
          alt={alt}
          width={finalWidth}
          height={finalHeight}
          className={cn("w-full h-full object-contain", imageClassName)}
          priority={priority}
        />
      </div>
    );
  }

  // Default "theme" variant:
  // - Light theme (default): shows white-bg logo (hidden in dark mode)
  // - Dark theme: shows black-bg logo (hidden in light mode)
  return (
    <div
      className={cn(
        "relative flex aspect-square items-center justify-center rounded-lg overflow-hidden shrink-0 shadow-xs",
        className
      )}
      {...props}
    >
      {/* Light Theme Logo */}
      <Image
        src={LOGO_WHITE_BG}
        alt={alt}
        width={finalWidth}
        height={finalHeight}
        className={cn("w-full h-full object-contain dark:hidden", imageClassName)}
        priority={priority}
      />
      {/* Dark Theme Logo */}
      <Image
        src={LOGO_BLACK_BG}
        alt={alt}
        width={finalWidth}
        height={finalHeight}
        className={cn("w-full h-full object-contain hidden dark:block", imageClassName)}
        priority={priority}
      />
    </div>
  );
}

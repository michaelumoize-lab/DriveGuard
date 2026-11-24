"use client";

import React from "react";
import { MotionConfig } from "motion/react";

export function LandingMotionProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  return <MotionConfig reducedMotion="user">{children}</MotionConfig>;
}

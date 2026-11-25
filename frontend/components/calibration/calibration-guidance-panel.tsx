"use client";

import React from "react";
import Link from "next/link";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import {
  Activity,
  AlertTriangle,
  ArrowDown,
  ArrowLeft,
  ArrowRight,
  Camera,
  CheckCircle2,
  Eye,
  FastForward,
  Loader2,
  Navigation,
  Play,
  RotateCcw,
  ScanFace,
  Square,
  Sun,
} from "lucide-react";
import type { CameraResolution, CameraState, OpticalFrameQuality } from "@/types/camera";
import type { BackendProcessingStatus, CalibrationPhase } from "@/types/telemetry";
import type { InitialDriverProfileData } from "@/components/monitor/unified-live-monitor";
import {
  formatCalibrationRejection,
  formatQualityWarning,
  getZoneMeta,
} from "./calibration-zones";

interface CalibrationGuidancePanelProps {
  cameraState: CameraState;
  resolution: CameraResolution | null;
  opticalQuality: OpticalFrameQuality | null;
  calibrationEngine?: "browser" | "python";
  profileSyncStatus?: string;
  isLivekitConnected: boolean;
  connectionStatus: "disconnected" | "connecting" | "connected" | "reconnecting" | "error";
  backendProcessingStatus: BackendProcessingStatus;
  backendType: "cloud" | "local";
  workerError: string | null;
  saveError: string | null;
  hasPendingProfileRetry: boolean;
  isCalibrating: boolean;
  isSavingProfile: boolean;
  isSuccess: boolean;
  calState: CalibrationPhase | null;
  savedProfile: InitialDriverProfileData | null;
  isFaceVisible: boolean | null;
  embeddedInOnboarding: boolean;
  onStartCameraCheck: () => void;
  onBeginCalibration: () => void;
  onCancelCalibration: () => void;
  onSkipZone: () => void;
  onRetrySave: () => void;
  onSwitchToCloud: () => void;
}

function ZoneDirectionIcon({ zone }: { zone?: string | null }) {
  switch (zone) {
    case "LEFT_MIRROR":
      return <ArrowLeft className="size-6 text-sky-500 motion-safe:animate-pulse" aria-hidden="true" />;
    case "RIGHT_MIRROR":
      return <ArrowRight className="size-6 text-amber-500 motion-safe:animate-pulse" aria-hidden="true" />;
    case "DASHBOARD":
      return <ArrowDown className="size-6 text-purple-500 motion-safe:animate-pulse" aria-hidden="true" />;
    case "CENTER":
    default:
      return <Navigation className="size-6 text-emerald-500" aria-hidden="true" />;
  }
}

export function CalibrationGuidancePanel({
  cameraState,
  resolution,
  opticalQuality,
  calibrationEngine = "browser",
  profileSyncStatus,
  isLivekitConnected,
  connectionStatus,
  backendProcessingStatus,
  backendType,
  workerError,
  saveError,
  hasPendingProfileRetry,
  isCalibrating,
  isSavingProfile,
  isSuccess,
  calState,
  savedProfile,
  isFaceVisible,
  embeddedInOnboarding,
  onStartCameraCheck,
  onBeginCalibration,
  onCancelCalibration,
  onSkipZone,
  onRetrySave,
  onSwitchToCloud,
}: CalibrationGuidancePanelProps) {
  const isCameraActive = cameraState === "active";
  const isCameraLoading =
    cameraState === "requesting-permission" || cameraState === "permission-granted";
  const isEngineConnecting =
    isCameraActive &&
    (connectionStatus === "connecting" ||
      connectionStatus === "reconnecting" ||
      backendProcessingStatus === "connecting" ||
      backendProcessingStatus === "awaiting_telemetry");

  const activeZoneMeta = getZoneMeta(calState?.current_zone);
  const qualityWarnings = calState?.quality_warnings ?? [];
  const rejectionInfo = calState?.rejection_reason
    ? formatCalibrationRejection(calState.rejection_reason)
    : null;

  const isLightingOptimal =
    opticalQuality?.lightingCondition === "OPTIMAL" &&
    opticalQuality?.blurCondition === "SHARP";

  const lightingLabel = !opticalQuality
    ? "Checking image…"
    : opticalQuality.lightingCondition === "UNDEREXPOSED"
      ? "Too dark — add light"
      : opticalQuality.lightingCondition === "OVEREXPOSED"
        ? "Too bright — reduce glare"
        : opticalQuality.blurCondition === "BLURRY"
          ? "Blurry — wipe lens or hold steady"
          : "Good lighting & focus";

  return (
    <Card className="border-border/80 lg:col-span-5 flex flex-col justify-between">
      <CardHeader className="pb-3 border-b border-border/60">
        <CardTitle className="text-base font-semibold flex items-center gap-2">
          <Eye className="size-4 text-primary shrink-0" aria-hidden="true" />
          <span>
            {isCalibrating
              ? "Follow Gaze Prompt"
              : isSuccess
                ? "Calibration Saved"
                : isCameraActive
                  ? "Camera & Positioning Check"
                  : "Before You Begin"}
          </span>
        </CardTitle>
        <CardDescription className="text-xs">
          {isCalibrating
            ? "Hold each head position briefly until the progress bar reaches 100%."
            : isSuccess
              ? "Your personal 4-zone thresholds are stored and ready for live monitoring."
              : isCameraActive
                ? "Confirm your lighting, camera angle, and face visibility before starting."
                : "Takes about 15–20 seconds across 4 normal driving head positions."}
        </CardDescription>
      </CardHeader>

      <CardContent className="pt-4 space-y-4 text-xs flex-1">
        {/* Connection / Worker Error Banner (Remote Python mode only) */}
        {calibrationEngine === "python" && (workerError || connectionStatus === "error") && (
          <div
            role="alert"
            className="p-3.5 rounded-xl bg-destructive/10 border border-destructive/30 text-destructive space-y-2"
          >
            <div className="flex items-start gap-2">
              <AlertTriangle className="size-4 shrink-0 mt-0.5" aria-hidden="true" />
              <div className="space-y-1">
                <p className="font-semibold">
                  {backendType === "local"
                    ? "Local AI Engine is not reachable"
                    : "AI Worker connection error"}
                </p>
                <p className="text-foreground/80 leading-relaxed">
                  {backendType === "local"
                    ? "Make sure the local Python backend is running on http://127.0.0.1:8000 (and allow Local Network Access if your browser prompts), or switch to LiveKit Cloud."
                    : workerError || "Could not connect to the remote calibration worker."}
                </p>
              </div>
            </div>
            <div className="flex flex-wrap items-center gap-2 pt-1">
              {backendType === "local" && (
                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  onClick={onSwitchToCloud}
                  className="h-8 text-xs border-destructive/40 hover:bg-destructive/10"
                >
                  Switch to LiveKit Cloud
                </Button>
              )}
              <Button
                type="button"
                size="sm"
                variant="outline"
                onClick={isCameraActive ? onBeginCalibration : onStartCameraCheck}
                className="h-8 text-xs border-destructive/40 hover:bg-destructive/10"
              >
                <RotateCcw className="size-3.5 mr-1" aria-hidden="true" />
                Retry
              </Button>
            </div>
          </div>
        )}

        {/* Save Error Banner (preserves captured profile for 1-click retry) */}
        {saveError && (
          <div
            role="alert"
            className="p-3.5 rounded-xl bg-destructive/10 border border-destructive/30 text-destructive space-y-2"
          >
            <div className="flex items-start gap-2">
              <AlertTriangle className="size-4 shrink-0 mt-0.5" aria-hidden="true" />
              <div className="space-y-1">
                <p className="font-semibold">Could not save your profile</p>
                <p className="text-foreground/80">{saveError}</p>
              </div>
            </div>
            <div className="flex flex-wrap gap-2 pt-1">
              {hasPendingProfileRetry && (
                <Button
                  type="button"
                  size="sm"
                  onClick={onRetrySave}
                  disabled={isSavingProfile}
                  className="h-8 text-xs"
                >
                  {isSavingProfile ? "Saving…" : "Retry Saving Profile"}
                </Button>
              )}
              <Button
                type="button"
                size="sm"
                variant="outline"
                onClick={onBeginCalibration}
                disabled={isSavingProfile}
                className="h-8 text-xs border-destructive/40"
              >
                <RotateCcw className="size-3.5 mr-1" aria-hidden="true" />
                Recalibrate
              </Button>
            </div>
          </div>
        )}

        {/* Backend Rejection Banner */}
        {rejectionInfo && (
          <div
            role="alert"
            className="p-3.5 rounded-xl bg-destructive/10 border border-destructive/30 text-destructive space-y-2"
          >
            <div className="flex items-start gap-2">
              <AlertTriangle className="size-4 shrink-0 mt-0.5" aria-hidden="true" />
              <div className="space-y-1">
                <p className="font-semibold">{rejectionInfo.title}</p>
                <p className="text-foreground/85 leading-relaxed">
                  {rejectionInfo.description}
                </p>
              </div>
            </div>
            <Button
              type="button"
              size="sm"
              variant="outline"
              onClick={onBeginCalibration}
              className="h-8 text-xs border-destructive/40 text-destructive hover:bg-destructive/10"
            >
              <RotateCcw className="size-3.5 mr-1.5" aria-hidden="true" />
              Try Again
            </Button>
          </div>
        )}

        {/* STATE A: Pre-Camera Overview (Step 1 Not Yet Started) */}
        {!isCameraActive && !isSuccess && (
          <div className="space-y-4">
            <div className="rounded-xl border border-border bg-muted/30 p-3.5 space-y-2.5">
              <p className="font-semibold text-foreground text-xs">
                How the 2-step calibration works
              </p>
              <ol className="space-y-2 text-muted-foreground list-decimal pl-4">
                <li>
                  <strong className="text-foreground">Positioning check:</strong> Turn on
                  your camera, adjust lighting, and center your face at normal driving distance.
                </li>
                <li>
                  <strong className="text-foreground">4-zone capture:</strong> Follow brief
                  prompts for <strong className="text-foreground">Road Ahead</strong>,{" "}
                  <strong className="text-foreground">Left Mirror</strong>,{" "}
                  <strong className="text-foreground">Right Mirror</strong>, and{" "}
                  <strong className="text-foreground">Dashboard</strong>.
                </li>
              </ol>
            </div>

            <div className="rounded-xl border border-border/70 bg-card p-3 space-y-1.5">
              <p className="font-semibold text-foreground">Positioning tips for best accuracy</p>
              <ul className="space-y-1 text-muted-foreground list-disc pl-4">
                <li>Place the camera at or slightly below eye level facing you.</li>
                <li>Avoid strong backlighting (sun or bright window directly behind you).</li>
                <li>Sit comfortably as you would while driving and keep both eyes visible.</li>
              </ul>
            </div>

            <Button
              type="button"
              size="default"
              onClick={onStartCameraCheck}
              disabled={isCameraLoading}
              className="w-full min-h-[44px] font-semibold"
            >
              {isCameraLoading ? (
                <>
                  <Loader2 className="size-4 mr-2 animate-spin" aria-hidden="true" />
                  Starting Camera…
                </>
              ) : (
                <>
                  <Camera className="size-4 mr-2" aria-hidden="true" />
                  Step 1: Enable Camera &amp; Check Position
                </>
              )}
            </Button>
          </div>
        )}

        {/* STATE B: Camera Positioning & Environment Pre-Check */}
        {isCameraActive && !isCalibrating && !isSuccess && (
          <div className="space-y-4">
            <div className="space-y-2">
              <div className="flex items-center justify-between p-2.5 rounded-xl border border-emerald-500/30 bg-emerald-500/5">
                <span className="flex items-center gap-2 font-medium text-foreground">
                  <Camera className="size-4 text-emerald-500" aria-hidden="true" />
                  Camera feed
                </span>
                <span className="font-mono tabular-nums text-[11px] text-emerald-600 dark:text-emerald-400 font-semibold">
                  {resolution ? `${resolution.width}×${resolution.height}` : "Active"}
                </span>
              </div>

              <div
                className={`flex items-center justify-between p-2.5 rounded-xl border ${
                  isLightingOptimal
                    ? "border-emerald-500/30 bg-emerald-500/5"
                    : "border-amber-500/35 bg-amber-500/10"
                }`}
              >
                <span className="flex items-center gap-2 font-medium text-foreground">
                  <Sun
                    className={`size-4 ${
                      isLightingOptimal ? "text-emerald-500" : "text-amber-500"
                    }`}
                    aria-hidden="true"
                  />
                  Lighting &amp; clarity
                </span>
                <span
                  className={`text-[11px] font-semibold ${
                    isLightingOptimal
                      ? "text-emerald-600 dark:text-emerald-400"
                      : "text-amber-600 dark:text-amber-400"
                  }`}
                >
                  {lightingLabel}
                </span>
              </div>

              <div
                className={`flex items-center justify-between p-2.5 rounded-xl border ${
                  isFaceVisible === true
                    ? "border-emerald-500/30 bg-emerald-500/5"
                    : isFaceVisible === false
                      ? "border-amber-500/35 bg-amber-500/10"
                      : "border-border bg-muted/30"
                }`}
              >
                <span className="flex items-center gap-2 font-medium text-foreground">
                  <ScanFace
                    className={`size-4 ${
                      isFaceVisible === true
                        ? "text-emerald-500"
                        : isFaceVisible === false
                          ? "text-amber-500"
                          : "text-muted-foreground"
                    }`}
                    aria-hidden="true"
                  />
                  Face &amp; AI readiness
                </span>
                <span className="text-[11px] font-semibold text-foreground">
                  {isEngineConnecting
                    ? calibrationEngine === "browser"
                      ? "Loading MediaPipe WASM…"
                      : "Connecting to AI…"
                    : isFaceVisible === true
                      ? "Face locked · Ready"
                      : isFaceVisible === false
                        ? "Center your face"
                        : isLivekitConnected
                          ? "Awaiting first frame…"
                          : calibrationEngine === "browser"
                            ? "Browser AI ready"
                            : "Ready to connect"}
                </span>
              </div>
            </div>

            <div className="rounded-xl border border-border/70 bg-muted/20 p-3 text-muted-foreground leading-relaxed">
              Once your face is centered in the preview and lighting looks clear, start Step 2 to
              record your 4 driving head positions.
            </div>

            <Button
              type="button"
              size="default"
              onClick={onBeginCalibration}
              disabled={isSavingProfile}
              className="w-full min-h-[44px] font-semibold"
            >
              <Play className="size-4 mr-2" aria-hidden="true" />
              Step 2: Begin 4-Zone Calibration (~15s)
            </Button>
          </div>
        )}

        {/* STATE C: Active 4-Zone Calibration Capture */}
        {isCalibrating && calState && !isSuccess && (
          <div className="space-y-4">
            {/* Active Target Prompt */}
            <div className="p-3.5 rounded-xl border border-primary/30 bg-primary/5 flex items-center gap-3">
              <div className="p-2.5 rounded-xl bg-primary/10 shrink-0">
                <ZoneDirectionIcon zone={calState.current_zone} />
              </div>
              <div className="min-w-0 space-y-0.5">
                <span className="text-[11px] font-mono text-muted-foreground block tabular-nums">
                  Zone {calState.phase_index} of {calState.total_phases} · {activeZoneMeta.title}
                </span>
                <p className="font-bold text-foreground text-sm leading-snug">
                  {calState.stability_feedback?.includes("Get ready")
                    ? calState.stability_feedback
                    : activeZoneMeta.activePrompt}
                </p>
                <p className="text-[11px] text-muted-foreground">
                  {activeZoneMeta.hint}
                </p>
              </div>
            </div>

            {/* Alignment Status Pill */}
            <div className="flex items-center justify-between gap-2">
              <span className="text-muted-foreground">Head Alignment</span>
              <Badge
                variant={calState.is_head_in_target_zone ? "outline" : "secondary"}
                className={`text-[11px] ${
                  calState.is_head_in_target_zone
                    ? "border-emerald-500/40 text-emerald-600 dark:text-emerald-400 bg-emerald-500/10"
                    : "border-amber-500/40 text-amber-600 dark:text-amber-400 bg-amber-500/10"
                }`}
              >
                {isFaceVisible === false
                  ? "Face not detected"
                  : calState.is_head_in_target_zone
                    ? "Aligned — hold steady"
                    : calState.stability_feedback || "Adjust head toward target"}
              </Badge>
            </div>

            {/* Current Zone Progress */}
            <div className="space-y-1.5 pt-2 border-t border-border">
              <div className="flex justify-between items-center font-mono tabular-nums">
                <span className="text-muted-foreground font-sans">Zone Capture Progress</span>
                <span className="font-semibold text-foreground">
                  {Math.round(calState.zone_progress_pct)}% ({calState.stable_samples_count}/
                  {calState.target_samples_count})
                </span>
              </div>
              <Progress
                value={calState.zone_progress_pct || 0}
                aria-label={`${activeZoneMeta.title} capture progress`}
                aria-valuetext={`${Math.round(calState.zone_progress_pct)}%`}
                className="h-2.5"
              />
            </div>

            {/* Overall Progress */}
            <div className="space-y-1.5 pt-2 border-t border-border">
              <div className="flex justify-between items-center font-mono tabular-nums">
                <span className="text-muted-foreground font-sans">Overall Progress</span>
                <span className="font-semibold text-foreground">
                  {Math.round(calState.total_progress_pct)}%
                </span>
              </div>
              <Progress
                value={calState.total_progress_pct || 0}
                aria-label="Overall calibration progress"
                aria-valuetext={`${Math.round(calState.total_progress_pct)}%`}
                className="h-1.5"
              />
            </div>

            {/* Controls */}
            <div className="flex items-center gap-2 pt-2">
              <Button
                type="button"
                size="sm"
                variant="destructive"
                onClick={onCancelCalibration}
                className="flex-1 min-h-[40px]"
              >
                <Square className="size-3.5 mr-1.5" aria-hidden="true" />
                Cancel
              </Button>
              <Button
                type="button"
                size="sm"
                variant="outline"
                onClick={onSkipZone}
                disabled={isSavingProfile}
                className="flex-1 min-h-[40px]"
              >
                <FastForward className="size-3.5 mr-1.5 text-primary" aria-hidden="true" />
                Skip Zone
              </Button>
            </div>
          </div>
        )}

        {/* STATE C2: Saving Profile */}
        {isSavingProfile && (
          <div className="p-4 rounded-xl border border-primary/30 bg-primary/5 flex items-center gap-3">
            <Loader2 className="size-5 text-primary animate-spin shrink-0" aria-hidden="true" />
            <div>
              <p className="font-semibold text-foreground">Saving your personal baseline…</p>
              <p className="text-muted-foreground">
                Syncing your 4-zone calibration profile.
              </p>
            </div>
          </div>
        )}

        {/* STATE D: Completed & Saved */}
        {isSuccess && !isSavingProfile && (
          <div className="space-y-4">
            <div className="p-4 rounded-xl border border-emerald-500/35 bg-emerald-500/10 space-y-2">
              <div className="flex items-center justify-between gap-2 text-emerald-600 dark:text-emerald-400 font-semibold text-sm">
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="size-5 shrink-0" aria-hidden="true" />
                  <span>
                    4-Zone {calibrationEngine === "browser" ? "Browser AI" : "Python AI"} Profile Active
                  </span>
                </div>
                {calibrationEngine === "browser" && profileSyncStatus && (
                  <Badge variant="outline" className="text-[10px] font-mono border-emerald-500/40 text-emerald-600 dark:text-emerald-400">
                    {profileSyncStatus === "SYNCED"
                      ? "INDEXEDDB + DB SYNCED"
                      : profileSyncStatus === "PENDING_SYNC"
                        ? "SAVED LOCALLY (SYNC PENDING)"
                        : profileSyncStatus}
                  </Badge>
                )}
              </div>
              <p className="text-foreground/85 leading-relaxed">
                DriveGuard now knows your resting eye openness (
                <strong className="font-mono tabular-nums text-foreground">
                  {savedProfile?.centerEarMedian.toFixed(3) ?? "—"}
                </strong>
                ) and normal glance angles across all 4 cabin zones.
              </p>
            </div>

            {qualityWarnings.length > 0 && (
              <div className="p-3 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-700 dark:text-amber-300 space-y-1">
                <p className="font-semibold">Calibration Advisories</p>
                <ul className="list-disc pl-4 space-y-1">
                  {qualityWarnings.map((w, idx) => (
                    <li key={idx}>{formatQualityWarning(w)}</li>
                  ))}
                </ul>
              </div>
            )}

            <div className="flex flex-col gap-2 pt-1">
              {!embeddedInOnboarding && (
                <Button asChild size="default" className="w-full min-h-[44px] font-semibold">
                  <Link href="/dashboard/live-monitor">
                    <Activity className="size-4 mr-2" aria-hidden="true" />
                    Start Live Monitoring
                  </Link>
                </Button>
              )}
              <Button
                type="button"
                size="sm"
                variant="outline"
                onClick={onBeginCalibration}
                className="w-full min-h-[40px]"
              >
                <RotateCcw className="size-3.5 mr-1.5" aria-hidden="true" />
                Recalibrate All 4 Zones
              </Button>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

"use client";

import React, { useState } from "react";
import { useRouter } from "next/navigation";
import { toast } from "sonner";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { GuidedCalibrationWizard } from "@/components/calibration/guided-calibration-wizard";
import { InitialDriverProfileData } from "@/components/monitor/unified-live-monitor";
import {
  User,
  Car,
  Shield,
  CheckCircle2,
  ArrowRight,
  ArrowLeft,
  Loader2,
  Sparkles,
  Phone,
  Glasses,
  Volume2,
  Eye,
  Camera,
  Activity,
} from "lucide-react";

interface InitialProfile {
  userId: string;
  onboardingStep: number;
  completedSteps: string[];
  phoneNumber?: string | null;
  licenseNumber?: string | null;
  vehicleMake?: string | null;
  vehicleModel?: string | null;
  vehicleYear?: number | null;
  licensePlate?: string | null;
  emergencyContactName?: string | null;
  emergencyContactPhone?: string | null;
  sensitivityLevel?: string | null;
  audioAlertsEnabled?: boolean | null;
  mirrorSuppression?: boolean | null;
  wearsGlasses?: boolean | null;
}

interface OnboardingWizardProps {
  user: {
    id: string;
    name: string;
    email: string;
  };
  initialProfile: InitialProfile;
  initialDriverProfile?: InitialDriverProfileData | null;
}

export function OnboardingWizard({
  user,
  initialProfile,
  initialDriverProfile = null,
}: OnboardingWizardProps) {
  const router = useRouter();

  // Resume at the saved step (bounded between 1 and 5)
  const [currentStep, setCurrentStep] = useState<number>(
    Math.min(5, Math.max(1, initialProfile.onboardingStep || 1))
  );
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Calibrated Biometric Profile State
  const [calibratedDriverProfile, setCalibratedDriverProfile] =
    useState<InitialDriverProfileData | null>(initialDriverProfile);

  // Form State
  const [fullName, setFullName] = useState(user.name || "");
  const [phoneNumber, setPhoneNumber] = useState(initialProfile.phoneNumber || "");
  const [licenseNumber, setLicenseNumber] = useState(initialProfile.licenseNumber || "");

  const [vehicleMake, setVehicleMake] = useState(initialProfile.vehicleMake || "");
  const [vehicleModel, setVehicleModel] = useState(initialProfile.vehicleModel || "");
  const [vehicleYear, setVehicleYear] = useState(initialProfile.vehicleYear?.toString() || "");
  const [licensePlate, setLicensePlate] = useState(initialProfile.licensePlate || "");

  const [emergencyName, setEmergencyName] = useState(initialProfile.emergencyContactName || "");
  const [emergencyPhone, setEmergencyPhone] = useState(initialProfile.emergencyContactPhone || "");

  const [sensitivityLevel, setSensitivityLevel] = useState<"STANDARD" | "HIGH" | "HIGHWAY">(
    (initialProfile.sensitivityLevel as "STANDARD" | "HIGH" | "HIGHWAY") || "STANDARD"
  );
  const [audioAlerts, setAudioAlerts] = useState(initialProfile.audioAlertsEnabled ?? true);
  const [mirrorSuppression, setMirrorSuppression] = useState(initialProfile.mirrorSuppression ?? true);
  const [wearsGlasses, setWearsGlasses] = useState(initialProfile.wearsGlasses ?? false);

  const stepsList = [
    { number: 1, title: "Driver Profile", icon: User },
    { number: 2, title: "Vehicle Info", icon: Car },
    { number: 3, title: "Safety Prefs", icon: Shield },
    { number: 4, title: "4-Zone & Blink Cal", icon: Camera },
    { number: 5, title: "Review & Activate", icon: CheckCircle2 },
  ];

  const progressPercentage = ((currentStep - 1) / (stepsList.length - 1)) * 100;

  const saveStepData = async (stepNum: number, isFinal: boolean = false) => {
    setIsSubmitting(true);
    try {
      let payload: Record<string, unknown> = {
        step: stepNum,
        isFinalStep: isFinal,
      };

      if (stepNum === 1) {
        payload = {
          ...payload,
          completedStepKey: "DRIVER_PROFILE",
          phoneNumber,
          licenseNumber,
          emergencyContactName: emergencyName,
          emergencyContactPhone: emergencyPhone,
        };
      } else if (stepNum === 2) {
        payload = {
          ...payload,
          completedStepKey: "VEHICLE_INFO",
          vehicleMake,
          vehicleModel,
          vehicleYear: vehicleYear ? parseInt(vehicleYear, 10) : undefined,
          licensePlate,
        };
      } else if (stepNum === 3) {
        payload = {
          ...payload,
          completedStepKey: "SAFETY_PREFERENCES",
          sensitivityLevel,
          audioAlertsEnabled: audioAlerts,
          mirrorSuppression,
          wearsGlasses,
        };
      } else if (stepNum === 4) {
        payload = {
          ...payload,
          completedStepKey: "BIOMETRIC_BLINK_CALIBRATION",
        };
      } else if (stepNum === 5 || isFinal) {
        payload = {
          ...payload,
          step: 5,
          completedStepKey: "ONBOARDING_COMPLETED",
          isFinalStep: true,
        };
      }

      const res = await fetch("/api/onboarding", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        throw new Error("Failed to save onboarding progress");
      }

      if (isFinal) {
        toast.success("Onboarding completed! Welcome to DriveGuard.");
        router.push("/dashboard");
        router.refresh();
      } else {
        setCurrentStep((prev) => Math.min(5, prev + 1));
      }
    } catch (err) {
      console.error(err);
      toast.error("Could not save progress. Please check your connection.");
    } finally {
      setIsSubmitting(false);
    }
  };

  // Extract temporal blink metrics for Step 5 Review card
  const temporalSummary = (() => {
    const rawZones = (calibratedDriverProfile?.zoneBaselines || {}) as Record<string, any>;
    const meta = (rawZones._temporalBaseline || {}) as Record<string, any>;
    return {
      blinkRatePerMin:
        typeof meta.blinkFreqMedianPerMin === "number" ? meta.blinkFreqMedianPerMin : 16.0,
      blinkDurationMs: calibratedDriverProfile?.blinkDurationMedianMs ?? 235.0,
      blinkRcr: calibratedDriverProfile?.blinkRcrMedian ?? 0.95,
      restingPerclosPct:
        typeof meta.baselinePerclosPct === "number" ? meta.baselinePerclosPct : 3.5,
      blinksSampled:
        typeof meta.temporalBlinksSampled === "number" ? meta.temporalBlinksSampled : null,
    };
  })();

  return (
    <Card className="border-border/60 shadow-lg backdrop-blur-sm">
      <CardHeader className="pb-4">
        {/* Progress Tracker */}
        <div className="space-y-3">
          <div className="flex items-center justify-between text-xs font-medium text-muted-foreground">
            <span>
              Step {currentStep} of {stepsList.length}
            </span>
            <span>{Math.round(progressPercentage)}% completed</span>
          </div>
          <Progress value={progressPercentage} className="h-1.5" />

          {/* Stepper Pills */}
          <div className="grid grid-cols-5 gap-2 pt-2">
            {stepsList.map((s) => {
              const Icon = s.icon;
              const isActive = s.number === currentStep;
              const isDone = s.number < currentStep;

              return (
                <div
                  key={s.number}
                  className={`flex items-center gap-1.5 p-2 rounded-lg border text-xs transition-colors ${
                    isActive
                      ? "border-primary bg-primary/10 text-primary font-medium"
                      : isDone
                      ? "border-border/80 bg-muted/40 text-muted-foreground"
                      : "border-transparent text-muted-foreground/60"
                  }`}
                >
                  <Icon className="size-3.5 shrink-0" />
                  <span className="truncate hidden sm:inline">{s.title}</span>
                </div>
              );
            })}
          </div>
        </div>

        <CardTitle className="text-xl sm:text-2xl pt-4">
          {currentStep === 1 && "Personal & Driver Contact Information"}
          {currentStep === 2 && "Vehicle Details"}
          {currentStep === 3 && "AI Detection & Alert Preferences"}
          {currentStep === 4 && "4-Zone Spatial + 2-Minute Personal Blink Rate Calibration"}
          {currentStep === 5 && "Review & System Activation"}
        </CardTitle>
        <CardDescription>
          {currentStep === 1 &&
            "Let's set up your profile and primary contact details for safety notifications."}
          {currentStep === 2 &&
            "Enter your primary vehicle's details to calibrate interior cabin perspective."}
          {currentStep === 3 &&
            "Customise the sensitivity of the early warning cascade and fatigue alerts."}
          {currentStep === 4 &&
            "Calibrate your 4 cabin gaze zones (~20s) followed by a 2-minute natural blink stream so DriveGuard adapts to your personal blink rate and duration."}
          {currentStep === 5 &&
            "Confirm your personal baselines and vehicle setup. You will be ready to begin live driving telemetry."}
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-4 pt-2">
        {/* STEP 1: DRIVER & CONTACT DETAILS */}
        {currentStep === 1 && (
          <div className="space-y-4">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="fullName">Full Name</Label>
                <Input
                  id="fullName"
                  name="fullName"
                  autoComplete="name"
                  value={fullName}
                  onChange={(e) => setFullName(e.target.value)}
                  placeholder="e.g. John Doe"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="email">Email Address</Label>
                <Input
                  id="email"
                  name="email"
                  autoComplete="email"
                  value={user.email}
                  disabled
                  className="bg-muted/50"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="phone">Phone Number</Label>
                <Input
                  id="phone"
                  name="phoneNumber"
                  type="tel"
                  inputMode="tel"
                  autoComplete="tel"
                  value={phoneNumber}
                  onChange={(e) => setPhoneNumber(e.target.value)}
                  placeholder="+1 (555) 000-0000"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="license">Driving License / Fleet ID</Label>
                <Input
                  id="license"
                  name="licenseNumber"
                  autoComplete="off"
                  autoCapitalize="characters"
                  spellCheck={false}
                  value={licenseNumber}
                  onChange={(e) => setLicenseNumber(e.target.value)}
                  placeholder="Optional (e.g. DL-98214)"
                />
              </div>
            </div>

            <div className="border-t border-border pt-4">
              <h4 className="text-sm font-semibold text-foreground mb-3 flex items-center gap-2">
                <Phone className="size-4 text-primary" />
                Emergency Contact (Optional)
              </h4>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="emName">Contact Name</Label>
                  <Input
                    id="emName"
                    name="emergencyContactName"
                    autoComplete="name"
                    value={emergencyName}
                    onChange={(e) => setEmergencyName(e.target.value)}
                    placeholder="e.g. Jane Doe (Spouse/Dispatch)"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="emPhone">Contact Phone</Label>
                  <Input
                    id="emPhone"
                    name="emergencyContactPhone"
                    type="tel"
                    inputMode="tel"
                    autoComplete="tel"
                    value={emergencyPhone}
                    onChange={(e) => setEmergencyPhone(e.target.value)}
                    placeholder="+1 (555) 999-9999"
                  />
                </div>
              </div>
            </div>
          </div>
        )}

        {/* STEP 2: VEHICLE DETAILS */}
        {currentStep === 2 && (
          <div className="space-y-4">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="vMake">Vehicle Make</Label>
                <Input
                  id="vMake"
                  name="vehicleMake"
                  autoComplete="off"
                  autoCapitalize="words"
                  value={vehicleMake}
                  onChange={(e) => setVehicleMake(e.target.value)}
                  placeholder="e.g. Toyota, Ford, Volvo"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="vModel">Vehicle Model</Label>
                <Input
                  id="vModel"
                  name="vehicleModel"
                  autoComplete="off"
                  autoCapitalize="words"
                  value={vehicleModel}
                  onChange={(e) => setVehicleModel(e.target.value)}
                  placeholder="e.g. Camry, F-150, XC90"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="vYear">Model Year</Label>
                <Input
                  id="vYear"
                  name="vehicleYear"
                  type="number"
                  inputMode="numeric"
                  min={1970}
                  max={2030}
                  value={vehicleYear}
                  onChange={(e) => setVehicleYear(e.target.value)}
                  placeholder="e.g. 2023"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="vPlate">License Plate</Label>
                <Input
                  id="vPlate"
                  name="licensePlate"
                  autoComplete="off"
                  autoCapitalize="characters"
                  spellCheck={false}
                  value={licensePlate}
                  onChange={(e) => setLicensePlate(e.target.value)}
                  placeholder="e.g. ABC-1234"
                />
              </div>
            </div>

            <div className="p-3 bg-muted/40 rounded-lg border border-border text-xs text-muted-foreground flex items-start gap-2">
              <Car className="size-4 text-primary shrink-0 mt-0.5" />
              <span>
                Vehicle details help DriveGuard tailor camera field-of-view perspective angles for
                mirror glances and dashboard instrumentation.
              </span>
            </div>
          </div>
        )}

        {/* STEP 3: SAFETY & SENSITIVITY PREFERENCES */}
        {currentStep === 3 && (
          <div className="space-y-4">
            {/* Sensitivity Levels */}
            <div className="space-y-2">
              <Label id="onboarding-sensitivity-label">Fatigue Sensitivity Profile</Label>
              <div
                role="radiogroup"
                aria-labelledby="onboarding-sensitivity-label"
                className="grid grid-cols-1 sm:grid-cols-3 gap-3"
              >
                {[
                  {
                    level: "STANDARD",
                    title: "Standard (Urban)",
                    desc: "Balanced thresholds suitable for daily commuting and stop-and-go driving.",
                  },
                  {
                    level: "HIGH",
                    title: "High Sensitivity",
                    desc: "Tighter eye-closure and PERCLOS thresholds for maximum responsiveness.",
                  },
                  {
                    level: "HIGHWAY",
                    title: "Highway & Night",
                    desc: "Aggressive micro-sleep detection optimized for long, monotonous routes.",
                  },
                ].map((s) => (
                  <button
                    key={s.level}
                    type="button"
                    role="radio"
                    aria-checked={sensitivityLevel === s.level}
                    onClick={() =>
                      setSensitivityLevel(s.level as "STANDARD" | "HIGH" | "HIGHWAY")
                    }
                    className={`cursor-pointer rounded-lg border p-3.5 transition-all text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring ${
                      sensitivityLevel === s.level
                        ? "border-primary bg-primary/10 shadow-sm"
                        : "border-border hover:border-border/80 bg-background"
                    }`}
                  >
                    <div className="font-semibold text-sm text-foreground mb-1">{s.title}</div>
                    <div className="text-xs text-muted-foreground">{s.desc}</div>
                  </button>
                ))}
              </div>
            </div>

            {/* Toggle Switches */}
            <div className="space-y-3 pt-3 border-t border-border">
              <div className="flex items-center justify-between p-3 rounded-lg border border-border">
                <div className="space-y-0.5">
                  <div className="text-sm font-medium flex items-center gap-2">
                    <Volume2 className="size-4 text-primary" />
                    Audible Warning Chimes
                  </div>
                  <div className="text-xs text-muted-foreground">
                    Play sound notifications on progressive WARNING and CRITICAL fatigue alerts.
                  </div>
                </div>
                <Switch checked={audioAlerts} onCheckedChange={setAudioAlerts} />
              </div>

              <div className="flex items-center justify-between p-3 rounded-lg border border-border">
                <div className="space-y-0.5">
                  <div className="text-sm font-medium flex items-center gap-2">
                    <Eye className="size-4 text-primary" />
                    Context-Aware Mirror Check Suppression
                  </div>
                  <div className="text-xs text-muted-foreground">
                    Suppresses false alarms for natural mirror glances under 3.0 seconds.
                  </div>
                </div>
                <Switch checked={mirrorSuppression} onCheckedChange={setMirrorSuppression} />
              </div>

              <div className="flex items-center justify-between p-3 rounded-lg border border-border">
                <div className="space-y-0.5">
                  <div className="text-sm font-medium flex items-center gap-2">
                    <Glasses className="size-4 text-primary" />
                    I Wear Eyeglasses
                  </div>
                  <div className="text-xs text-muted-foreground">
                    Activates Stage-2 CNN arbitration to eliminate glare and frame edge
                    interference.
                  </div>
                </div>
                <Switch checked={wearsGlasses} onCheckedChange={setWearsGlasses} />
              </div>
            </div>
          </div>
        )}

        {/* STEP 4: 4-ZONE SPATIAL + 2-MINUTE BLINK RATE CALIBRATION */}
        {currentStep === 4 && (
          <div className="space-y-4">
            <GuidedCalibrationWizard
              driverId={user.id}
              initialDriverProfile={calibratedDriverProfile}
              embeddedInOnboarding={true}
              onCalibrationSaved={(savedProf) => {
                setCalibratedDriverProfile(savedProf);
              }}
            />
          </div>
        )}

        {/* STEP 5: REVIEW & ACTIVATE */}
        {currentStep === 5 && (
          <div className="space-y-4">
            <div className="p-4 bg-primary/5 border border-primary/20 rounded-xl space-y-2">
              <div className="flex items-center gap-2 text-primary font-semibold text-base">
                <Sparkles className="size-5" />
                Your Personal DriveGuard Profile is Ready!
              </div>
              <p className="text-xs text-muted-foreground">
                {calibratedDriverProfile?.isCalibrated
                  ? "Your 4-Zone spatial EAR thresholds and 2-minute personal blink rate & duration baselines have been saved to PostgreSQL and synced to the AI engine."
                  : "Your driver profile is configured. You can run or update your 4-Zone + 2-Minute Blink calibration anytime in Calibration Studio."}
              </p>
            </div>

            {/* Personal Biometric & Blink Cadence Summary */}
            <div className="p-3.5 rounded-xl border border-border bg-muted/20 space-y-2.5">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold flex items-center gap-1.5 text-foreground">
                  <Activity className="size-4 text-primary" />
                  Personal Biometric & Blink Cadence Baseline
                </span>
                <Badge
                  variant="outline"
                  className={
                    calibratedDriverProfile?.isCalibrated
                      ? "border-emerald-500/40 text-emerald-500 bg-emerald-500/10 font-mono text-[10px]"
                      : "border-amber-500/40 text-amber-500 bg-amber-500/10 font-mono text-[10px]"
                  }
                >
                  {calibratedDriverProfile?.isCalibrated
                    ? "CALIBRATED & STORED IN DB"
                    : "POPULATION DEFAULTS"}
                </Badge>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 font-mono text-xs">
                <div className="p-2.5 rounded-lg border border-border bg-background">
                  <span className="text-[10px] text-muted-foreground block">CENTER EAR</span>
                  <span className="text-sm font-bold text-foreground">
                    {(calibratedDriverProfile?.centerEarMedian ?? 0.285).toFixed(3)}
                  </span>
                </div>
                <div className="p-2.5 rounded-lg border border-border bg-background">
                  <span className="text-[10px] text-muted-foreground block">BLINK RATE</span>
                  <span className="text-sm font-bold text-sky-500">
                    {temporalSummary.blinkRatePerMin.toFixed(1)} /min
                  </span>
                </div>
                <div className="p-2.5 rounded-lg border border-border bg-background">
                  <span className="text-[10px] text-muted-foreground block">BLINK DURATION</span>
                  <span className="text-sm font-bold text-amber-500">
                    {Math.round(temporalSummary.blinkDurationMs)} ms
                  </span>
                </div>
                <div className="p-2.5 rounded-lg border border-border bg-background">
                  <span className="text-[10px] text-muted-foreground block">
                    RCR / RESTING PERCLOS
                  </span>
                  <span className="text-sm font-bold text-emerald-500">
                    {temporalSummary.blinkRcr.toFixed(2)} ·{" "}
                    {temporalSummary.restingPerclosPct.toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
              <div className="p-3 border border-border rounded-lg space-y-1">
                <span className="text-muted-foreground block">Driver</span>
                <span className="font-semibold text-foreground text-sm block">
                  {fullName || user.email}
                </span>
                <span className="text-muted-foreground block">
                  {phoneNumber || "No phone provided"}
                </span>
              </div>

              <div className="p-3 border border-border rounded-lg space-y-1">
                <span className="text-muted-foreground block">Vehicle</span>
                <span className="font-semibold text-foreground text-sm block">
                  {vehicleMake && vehicleModel
                    ? `${vehicleMake} ${vehicleModel} (${vehicleYear || "N/A"})`
                    : "Not specified"}
                </span>
                <span className="text-muted-foreground block">
                  {licensePlate ? `Plate: ${licensePlate}` : "Plate: N/A"}
                </span>
              </div>

              <div className="p-3 border border-border rounded-lg space-y-1">
                <span className="text-muted-foreground block">Sensitivity Mode</span>
                <Badge variant="outline" className="text-xs font-mono">
                  {sensitivityLevel}
                </Badge>
              </div>

              <div className="p-3 border border-border rounded-lg space-y-1">
                <span className="text-muted-foreground block">Safety Features</span>
                <span className="text-foreground block font-medium">
                  Audio: {audioAlerts ? "ON" : "OFF"} | Mirrors:{" "}
                  {mirrorSuppression ? "SUPPRESSED" : "OFF"} | Glasses:{" "}
                  {wearsGlasses ? "YES" : "NO"}
                </span>
              </div>
            </div>
          </div>
        )}
      </CardContent>

      <CardFooter className="flex flex-wrap items-center justify-between gap-2 border-t border-border pt-4">
        <div className="flex items-center gap-2">
          {currentStep > 1 && (
            <Button
              variant="outline"
              size="sm"
              onClick={() => setCurrentStep((prev) => Math.max(1, prev - 1))}
              disabled={isSubmitting}
            >
              <ArrowLeft className="size-4 mr-1.5" />
              Previous
            </Button>
          )}
          {currentStep < 5 && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() =>
                currentStep === 4 ? saveStepData(4, false) : saveStepData(5, true)
              }
              disabled={isSubmitting}
              className="text-xs text-muted-foreground hover:text-foreground"
            >
              {currentStep === 4
                ? "Skip calibration for now (calibrate later)"
                : "Skip for now (use defaults)"}
            </Button>
          )}
        </div>

        {currentStep < 5 ? (
          <Button
            size="sm"
            onClick={() => saveStepData(currentStep, false)}
            disabled={isSubmitting}
          >
            {isSubmitting ? (
              <>
                <Loader2 className="size-4 mr-1.5 animate-spin" />
                Saving...
              </>
            ) : (
              <>
                {currentStep === 4
                  ? calibratedDriverProfile?.isCalibrated
                    ? "Continue to Review"
                    : "Continue Without Calibrating"
                  : "Next Step"}
                <ArrowRight className="size-4 ml-1.5" />
              </>
            )}
          </Button>
        ) : (
          <Button
            size="sm"
            onClick={() => saveStepData(5, true)}
            disabled={isSubmitting}
            className="bg-primary text-primary-foreground hover:bg-primary/90 font-medium"
          >
            {isSubmitting ? (
              <>
                <Loader2 className="size-4 mr-1.5 animate-spin" />
                Activating...
              </>
            ) : (
              <>
                <CheckCircle2 className="size-4 mr-1.5" />
                Complete Onboarding & Enter Dashboard
              </>
            )}
          </Button>
        )}
      </CardFooter>
    </Card>
  );
}

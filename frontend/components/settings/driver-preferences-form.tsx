"use client";

import React, { useState } from "react";
import Link from "next/link";
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
import { Badge } from "@/components/ui/badge";
import {
  Car,
  Shield,
  Phone,
  Glasses,
  Volume2,
  Eye,
  Save,
  Loader2,
  Sliders,
  CheckCircle2,
  AlertTriangle,
} from "lucide-react";

export interface DriverPreferencesData {
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

interface DriverPreferencesFormProps {
  initialProfile: DriverPreferencesData;
  isCalibrated: boolean;
}

export function DriverPreferencesForm({
  initialProfile,
  isCalibrated,
}: DriverPreferencesFormProps) {
  const [isSaving, setIsSaving] = useState(false);

  const [phoneNumber, setPhoneNumber] = useState(initialProfile.phoneNumber || "");
  const [licenseNumber, setLicenseNumber] = useState(initialProfile.licenseNumber || "");
  const [vehicleMake, setVehicleMake] = useState(initialProfile.vehicleMake || "");
  const [vehicleModel, setVehicleModel] = useState(initialProfile.vehicleModel || "");
  const [vehicleYear, setVehicleYear] = useState(
    initialProfile.vehicleYear ? String(initialProfile.vehicleYear) : ""
  );
  const [licensePlate, setLicensePlate] = useState(initialProfile.licensePlate || "");
  const [emergencyContactName, setEmergencyContactName] = useState(
    initialProfile.emergencyContactName || ""
  );
  const [emergencyContactPhone, setEmergencyContactPhone] = useState(
    initialProfile.emergencyContactPhone || ""
  );
  const [sensitivityLevel, setSensitivityLevel] = useState<"STANDARD" | "HIGH" | "HIGHWAY">(
    (initialProfile.sensitivityLevel as "STANDARD" | "HIGH" | "HIGHWAY") || "STANDARD"
  );
  const [audioAlertsEnabled, setAudioAlertsEnabled] = useState(
    initialProfile.audioAlertsEnabled ?? true
  );
  const [mirrorSuppression, setMirrorSuppression] = useState(
    initialProfile.mirrorSuppression ?? true
  );
  const [wearsGlasses, setWearsGlasses] = useState(
    initialProfile.wearsGlasses ?? false
  );

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSaving(true);

    try {
      const parsedYear = vehicleYear ? parseInt(vehicleYear, 10) : undefined;
      if (vehicleYear && (Number.isNaN(parsedYear) || parsedYear! < 1970 || parsedYear! > 2030)) {
        toast.error("Please enter a valid vehicle model year between 1970 and 2030.");
        setIsSaving(false);
        return;
      }

      const res = await fetch("/api/onboarding", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          step: 4,
          isFinalStep: true,
          completedStepKey: "SETTINGS_UPDATED",
          phoneNumber,
          licenseNumber,
          vehicleMake,
          vehicleModel,
          vehicleYear: parsedYear,
          licensePlate,
          emergencyContactName,
          emergencyContactPhone,
          sensitivityLevel,
          audioAlertsEnabled,
          mirrorSuppression,
          wearsGlasses,
        }),
      });

      if (!res.ok) {
        throw new Error("Failed to save driver preferences");
      }

      toast.success("Driver & AI monitoring preferences saved.");
    } catch (error) {
      console.error("Failed to update driver preferences:", error);
      toast.error("Could not save driver preferences. Please try again.");
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <form onSubmit={handleSave} className="space-y-6">
      <Card className="border-border/80 shadow-sm">
        <CardHeader className="pb-4 border-b border-border/60">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div>
              <CardTitle className="text-lg flex items-center gap-2">
                <Shield className="size-5 text-primary" />
                Driver &amp; AI Monitoring Preferences
              </CardTitle>
              <CardDescription className="text-xs mt-1">
                Configure your vehicle cabin profile, fatigue cascade sensitivity, and audio alert defaults.
              </CardDescription>
            </div>
            <div className="flex items-center gap-2">
              {isCalibrated ? (
                <Badge className="bg-emerald-500/15 text-emerald-500 border border-emerald-500/30 text-[11px]">
                  <CheckCircle2 className="size-3 mr-1" />
                  4-Zone Calibrated
                </Badge>
              ) : (
                <Badge variant="outline" className="text-amber-500 border-amber-500/30 text-[11px]">
                  <AlertTriangle className="size-3 mr-1" />
                  Uncalibrated
                </Badge>
              )}
              <Button asChild variant="outline" size="sm" className="h-7 text-xs">
                <Link href="/dashboard/calibration">
                  <Sliders className="size-3 mr-1.5" />
                  Calibration Studio
                </Link>
              </Button>
            </div>
          </div>
        </CardHeader>

        <CardContent className="space-y-6 pt-5">
          {/* Sensitivity Selector */}
          <div className="space-y-2.5">
            <Label id="sensitivity-group-label" className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              Fatigue Cascade Sensitivity Profile
            </Label>
            <div
              role="radiogroup"
              aria-labelledby="sensitivity-group-label"
              className="grid grid-cols-1 sm:grid-cols-3 gap-3"
            >
              {[
                {
                  level: "STANDARD" as const,
                  title: "Standard (Urban)",
                  desc: "Balanced thresholds for daily commuting and city traffic.",
                },
                {
                  level: "HIGH" as const,
                  title: "High Sensitivity",
                  desc: "Tighter eye-closure & PERCLOS thresholds for rapid response.",
                },
                {
                  level: "HIGHWAY" as const,
                  title: "Highway & Night",
                  desc: "Aggressive micro-sleep detection for monotonous routes.",
                },
              ].map((s) => {
                const selected = sensitivityLevel === s.level;
                return (
                  <button
                    key={s.level}
                    type="button"
                    role="radio"
                    aria-checked={selected}
                    onClick={() => setSensitivityLevel(s.level)}
                    className={`rounded-lg border p-3 text-left transition-all cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring ${
                      selected
                        ? "border-primary bg-primary/10 shadow-sm"
                        : "border-border hover:border-border/80 bg-background"
                    }`}
                  >
                    <div className="font-semibold text-xs text-foreground mb-1 flex items-center justify-between">
                      <span>{s.title}</span>
                      {selected && (
                        <span className="size-2 rounded-full bg-primary" aria-hidden="true" />
                      )}
                    </div>
                    <div className="text-[11px] text-muted-foreground leading-relaxed">{s.desc}</div>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Safety Toggles */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-2">
            <div className="flex items-center justify-between p-3 rounded-lg border border-border bg-muted/10">
              <div className="space-y-0.5 pr-2">
                <Label htmlFor="pref-audio" className="text-xs font-medium flex items-center gap-1.5 cursor-pointer">
                  <Volume2 className="size-3.5 text-primary shrink-0" />
                  Audible Alerts
                </Label>
                <p className="text-[11px] text-muted-foreground">
                  4-tier Web Audio synthesis on fatigue states.
                </p>
              </div>
              <Switch
                id="pref-audio"
                checked={audioAlertsEnabled}
                onCheckedChange={setAudioAlertsEnabled}
              />
            </div>

            <div className="flex items-center justify-between p-3 rounded-lg border border-border bg-muted/10">
              <div className="space-y-0.5 pr-2">
                <Label htmlFor="pref-mirror" className="text-xs font-medium flex items-center gap-1.5 cursor-pointer">
                  <Eye className="size-3.5 text-primary shrink-0" />
                  Mirror Suppression
                </Label>
                <p className="text-[11px] text-muted-foreground">
                  Suppress alarms during mirror checks (&lt;3.0s).
                </p>
              </div>
              <Switch
                id="pref-mirror"
                checked={mirrorSuppression}
                onCheckedChange={setMirrorSuppression}
              />
            </div>

            <div className="flex items-center justify-between p-3 rounded-lg border border-border bg-muted/10">
              <div className="space-y-0.5 pr-2">
                <Label htmlFor="pref-glasses" className="text-xs font-medium flex items-center gap-1.5 cursor-pointer">
                  <Glasses className="size-3.5 text-primary shrink-0" />
                  Eyewear / Glasses
                </Label>
                <p className="text-[11px] text-muted-foreground">
                  Prioritize Stage-2 CNN arbitration for glare.
                </p>
              </div>
              <Switch
                id="pref-glasses"
                checked={wearsGlasses}
                onCheckedChange={setWearsGlasses}
              />
            </div>
          </div>

          {/* Driver Contact & Emergency Info */}
          <div className="border-t border-border/60 pt-4 space-y-3">
            <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground flex items-center gap-1.5">
              <Phone className="size-3.5 text-primary" />
              Driver Contact &amp; Emergency Dispatch
            </h4>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="space-y-1.5">
                <Label htmlFor="pref-phone" className="text-xs">Driver Phone Number</Label>
                <Input
                  id="pref-phone"
                  type="tel"
                  autoComplete="tel"
                  value={phoneNumber}
                  onChange={(e) => setPhoneNumber(e.target.value)}
                  placeholder="+1 (555) 000-0000"
                  className="h-9 text-xs"
                />
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="pref-license" className="text-xs">Driving License / Fleet ID</Label>
                <Input
                  id="pref-license"
                  value={licenseNumber}
                  onChange={(e) => setLicenseNumber(e.target.value)}
                  placeholder="Optional (e.g. DL-98214)"
                  className="h-9 text-xs"
                />
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="pref-em-name" className="text-xs">Emergency Contact Name</Label>
                <Input
                  id="pref-em-name"
                  value={emergencyContactName}
                  onChange={(e) => setEmergencyContactName(e.target.value)}
                  placeholder="e.g. Jane Doe (Dispatch)"
                  className="h-9 text-xs"
                />
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="pref-em-phone" className="text-xs">Emergency Contact Phone</Label>
                <Input
                  id="pref-em-phone"
                  type="tel"
                  value={emergencyContactPhone}
                  onChange={(e) => setEmergencyContactPhone(e.target.value)}
                  placeholder="+1 (555) 999-9999"
                  className="h-9 text-xs"
                />
              </div>
            </div>
          </div>

          {/* Vehicle Details */}
          <div className="border-t border-border/60 pt-4 space-y-3">
            <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground flex items-center gap-1.5">
              <Car className="size-3.5 text-primary" />
              Vehicle Cabin Metadata
            </h4>
            <div className="grid grid-cols-1 sm:grid-cols-4 gap-3">
              <div className="space-y-1.5">
                <Label htmlFor="pref-make" className="text-xs">Make</Label>
                <Input
                  id="pref-make"
                  value={vehicleMake}
                  onChange={(e) => setVehicleMake(e.target.value)}
                  placeholder="e.g. Volvo"
                  className="h-9 text-xs"
                />
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="pref-model" className="text-xs">Model</Label>
                <Input
                  id="pref-model"
                  value={vehicleModel}
                  onChange={(e) => setVehicleModel(e.target.value)}
                  placeholder="e.g. XC90"
                  className="h-9 text-xs"
                />
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="pref-year" className="text-xs">Model Year</Label>
                <Input
                  id="pref-year"
                  type="number"
                  min={1970}
                  max={2030}
                  value={vehicleYear}
                  onChange={(e) => setVehicleYear(e.target.value)}
                  placeholder="2025"
                  className="h-9 text-xs"
                />
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="pref-plate" className="text-xs">License Plate</Label>
                <Input
                  id="pref-plate"
                  value={licensePlate}
                  onChange={(e) => setLicensePlate(e.target.value)}
                  placeholder="ABC-1234"
                  className="h-9 text-xs"
                />
              </div>
            </div>
          </div>
        </CardContent>

        <CardFooter className="border-t border-border/60 pt-4 flex items-center justify-between">
          <span className="text-[11px] text-muted-foreground">
            Changes apply immediately to subsequent Live Monitoring sessions.
          </span>
          <Button type="submit" size="sm" disabled={isSaving}>
            {isSaving ? (
              <>
                <Loader2 className="size-3.5 mr-1.5 animate-spin" />
                Saving...
              </>
            ) : (
              <>
                <Save className="size-3.5 mr-1.5" />
                Save Preferences
              </>
            )}
          </Button>
        </CardFooter>
      </Card>
    </form>
  );
}

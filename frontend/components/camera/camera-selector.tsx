"use client";

import React from "react";
import { CameraDevice } from "@/types/camera";
import { Camera, RefreshCw, Smartphone, Usb } from "lucide-react";
import { Button } from "@/components/ui/button";

interface CameraSelectorProps {
  devices: CameraDevice[];
  selectedDeviceId: string;
  onSelectDevice: (deviceId: string) => void;
  onRefreshDevices: () => void;
  disabled?: boolean;
}

export function CameraSelector({
  devices,
  selectedDeviceId,
  onSelectDevice,
  onRefreshDevices,
  disabled = false,
}: CameraSelectorProps) {
  const [isRefreshing, setIsRefreshing] = React.useState(false);

  const handleRefresh = async () => {
    setIsRefreshing(true);
    try {
      await onRefreshDevices();
    } finally {
      setIsRefreshing(false);
    }
  };

  return (
    <div className="flex items-center gap-2 w-full">
      <div className="relative flex-1">
        <select
          aria-label="Select camera input device"
          value={selectedDeviceId}
          onChange={(e) => onSelectDevice(e.target.value)}
          disabled={disabled || devices.length === 0}
          className="w-full h-11 px-3.5 pr-8 rounded-xl bg-card border border-border text-xs sm:text-sm font-medium text-foreground focus:outline-none focus:ring-2 focus:ring-primary/40 disabled:opacity-50 appearance-none cursor-pointer"
        >
          {devices.length === 0 ? (
            <option value="">No cameras enumerated</option>
          ) : (
            devices.map((device, idx) => (
              <option key={device.deviceId || idx} value={device.deviceId}>
                {device.label || `Camera ${idx + 1}`}
                {device.facingMode !== "unknown" ? ` (${device.facingMode})` : ""}
              </option>
            ))
          )}
        </select>
        <Camera className="size-4 text-muted-foreground absolute right-3 top-3.5 pointer-events-none" />
      </div>

      <Button
        size="icon"
        variant="outline"
        onClick={handleRefresh}
        disabled={isRefreshing || disabled}
        className="min-h-[44px] min-w-[44px] rounded-xl shrink-0 border-border"
        title="Rescan camera devices"
        aria-label="Rescan available camera devices"
      >
        <RefreshCw className={`size-4 ${isRefreshing ? "animate-spin text-primary" : ""}`} />
      </Button>
    </div>
  );
}

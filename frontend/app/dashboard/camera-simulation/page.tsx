import { redirect } from "next/navigation";

export default function CameraSimulationRedirectPage() {
  redirect("/dashboard/live-monitor");
}

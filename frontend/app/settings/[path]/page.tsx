import { viewPaths } from "@better-auth-ui/core"
import { getServerSession } from "@/lib/get-session";
import { dehydrate, HydrationBoundary } from "@tanstack/react-query"
import { notFound, redirect } from "next/navigation"

import { Settings } from "@/components/auth/settings/settings"
import { getQueryClient } from "@/lib/query-client"

export default async function SettingsPage({
  params
}: {
  params: Promise<{
    path: string
  }>
}) {
  const { path } = await params

  if (!Object.values(viewPaths.settings).includes(path)) {
    notFound()
  }

  const queryClient = getQueryClient()

  const session = await getServerSession();

  if (!session) {
    redirect(
      `/auth/sign-in?redirectTo=${encodeURIComponent(`/settings/${path}`)}`
    )
  }

  return (
    <HydrationBoundary state={dehydrate(queryClient)}>
      <div className="w-full max-w-3xl mx-auto p-4 md:p-6">
        <Settings path={path} />
      </div>
    </HydrationBoundary>
  )
}
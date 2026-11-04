"use client"

import { QueryClientProvider } from "@tanstack/react-query"
import Link from "next/link"
import { useRouter } from "next/navigation"
import type { ReactNode } from "react"
import { useTheme } from "next-themes"
import { deleteUserPlugin } from "@/lib/auth/delete-user-plugin"
import { authClient } from "@/lib/auth-client"
import { lastLoginMethodPlugin } from "@/lib/auth/last-login-method-plugin"
import { multiSessionPlugin } from "@/lib/auth/multi-session-plugin"
import { themePlugin } from "@/lib/auth/theme-plugin"
import { getQueryClient } from "@/lib/query-client"
import { AuthProvider } from "./auth/auth-provider"
import { Toaster } from "./ui/sonner"

import { TooltipProvider } from "./ui/tooltip"

export function Providers({ children }: { children: ReactNode }) {
  const router = useRouter()
  const queryClient = getQueryClient()
  const theme = useTheme()

  return (
    <QueryClientProvider client={queryClient}>
      <AuthProvider
        authClient={authClient}
        redirectTo="/dashboard"
        socialProviders={["google"]}
        navigate={({ to, replace }) =>
          replace ? router.replace(to) : router.push(to)
        }
        plugins={[
          deleteUserPlugin(),
          lastLoginMethodPlugin(),
          multiSessionPlugin(),
          themePlugin({ useTheme: () => theme }),
        ]}
        Link={Link}
      >
        <TooltipProvider>
          {children}
        </TooltipProvider>

        <Toaster />
      </AuthProvider>
    </QueryClientProvider>
  )
}

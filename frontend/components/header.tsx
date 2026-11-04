import { dehydrate, HydrationBoundary } from "@tanstack/react-query";
import Link from "next/link";
import { getServerSession } from "@/lib/get-session";

import { getQueryClient } from "@/lib/query-client";
import { UserButton } from "./auth/user/user-button";
import { ThemeToggle } from "./theme-toggle";
import { Logo } from "./logo";

export async function Header() {
  const queryClient = getQueryClient();
  const session = await getServerSession();

  return (
    <HydrationBoundary state={dehydrate(queryClient)}>
      <header className="sticky top-0 z-10 bg-background/80 backdrop-blur-md border-b border-border">
        <div className="py-3 px-4 md:px-6 mx-auto justify-between flex items-center">
          <Link href="/" className="flex items-center gap-2.5 no-underline">
            <Logo />
            <div className="flex flex-col">
              <span className="font-bold tracking-tight text-sm text-foreground">IntelliGuard</span>
              <span className="text-[10px] font-mono text-muted-foreground uppercase tracking-wider">
                Security System
              </span>
            </div>
          </Link>

          <div className="flex items-center gap-2">
            <ThemeToggle />
            <UserButton size="icon" user={session?.user} />
          </div>
        </div>
      </header>
    </HydrationBoundary>
  );
}

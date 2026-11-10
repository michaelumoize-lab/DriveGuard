"use client";

import * as React from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import {
  Home,
  LayoutDashboard,
  Users,
  UserPlus,
  UserCog,
  ClipboardList,
  ShieldCheck,
  ShieldAlert,
  Camera,
  Activity,
  Eye,
  Settings,
  CircleHelp,
  LogOut,
  LucideIcon,
  Loader2,
  ChevronDown,
  ChevronRight,
  Cpu,
  Sparkles,
  Sun,
  Moon,
  Monitor,
  Check,
} from "lucide-react";
import { useTheme } from "next-themes";
import { signOut } from "@/lib/auth-client";

import { UserAvatar } from "@/components/user-avatar";
import { Logo } from "@/components/logo";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarMenuSub,
  SidebarMenuSubButton,
  SidebarMenuSubItem,
  useSidebar,
} from "@/components/ui/sidebar";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
  DropdownMenuSub,
  DropdownMenuSubTrigger,
  DropdownMenuSubContent,
} from "@/components/ui/dropdown-menu";

type UserRole = "USER" | "ADMIN" | "DRIVER" | string;

interface AdminSidebarProps extends React.ComponentProps<typeof Sidebar> {
  user: {
    id: string;
    email: string;
    name?: string | null;
    image?: string | null;
    role: UserRole;
  };
}

interface NavItem {
  title: string;
  href?: string;
  icon: LucideIcon;
  isSubmenu?: boolean;
  subItems?: Omit<NavItem, "isSubmenu" | "subItems">[];
}

const mainNavItems: NavItem[] = [
  {
    title: "Home",
    href: "/",
    icon: Home,
  },
  {
    title: "Safety Dashboard",
    href: "/dashboard",
    icon: LayoutDashboard,
  },
  {
    title: "Live Monitoring",
    href: "/dashboard/live-monitor",
    icon: Activity,
  },
  {
    title: "Calibration",
    href: "/dashboard/calibration",
    icon: Eye,
  },
  {
    title: "Driving Sessions",
    href: "/dashboard/sessions",
    icon: ClipboardList,
  },
  {
    title: "Fatigue Alerts",
    href: "/dashboard/alerts",
    icon: ShieldAlert,
  },
  {
    title: "System Specs",
    href: "/dashboard/project-overview",
    icon: Sparkles,
  },
];

const supportNavItems: NavItem[] = [
  { title: "Settings", href: "/settings/account", icon: Settings },
  { title: "Help", href: "/dashboard/help", icon: CircleHelp },
];

export function AdminSidebar({ user, className, ...props }: AdminSidebarProps) {
  const pathname = usePathname();
  const router = useRouter();
  const { isMobile, setOpenMobile } = useSidebar();
  const { theme, setTheme } = useTheme();
  const [isLoading, setIsLoading] = React.useState(false);
  const [openSubmenus, setOpenSubmenus] = React.useState<Record<string, boolean>>({});

  const handleLogout = async () => {
    setIsLoading(true);
    try {
      await signOut({
        fetchOptions: {
          onSuccess: () => {
            router.push("/auth/sign-in");
          },
        },
      });
    } catch (error) {
      console.error("Sign out error:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleNavClick = () => {
    if (isMobile) {
      setOpenMobile(false);
    }
  };

  const toggleSubmenu = (title: string) => {
    setOpenSubmenus((prev) => ({
      ...prev,
      [title]: !prev[title],
    }));
  };

  return (
    <Sidebar collapsible="icon" className={className} {...props}>
      {/* Fixed Sticky Header with h-14 height for seamless navbar alignment */}
      <SidebarHeader className="h-14 shrink-0 border-b border-sidebar-border justify-center px-3">
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton size="lg" asChild>
              <Link href="/dashboard" onClick={handleNavClick}>
                <Logo className="size-8" />
                <div className="grid flex-1 text-left text-sm leading-tight">
                  <span className="truncate font-semibold">DriveGuard</span>
                  <span className="truncate text-[10px] text-muted-foreground font-medium uppercase tracking-wider">
                    Driver Safety AI
                  </span>
                </div>
              </Link>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>

      {/* Independently Scrollable Navigation Content */}
      <SidebarContent className="flex-1 overflow-y-auto">
        <SidebarGroup>
          <SidebarGroupLabel>Platform</SidebarGroupLabel>
          <SidebarMenu>
            {mainNavItems.map((item) => {
              const Icon = item.icon;
              const isActive = item.href ? (pathname === item.href || (item.href !== "/dashboard" && pathname.startsWith(item.href + "/"))) : false;

              if (item.isSubmenu && item.subItems) {
                const isSubmenuActive = item.subItems.some(
                  (subItem) => subItem.href && (pathname === subItem.href || (subItem.href !== "/dashboard" && pathname.startsWith(subItem.href + "/")))
                );
                const isOpen = openSubmenus[item.title] ?? isSubmenuActive;

                return (
                  <Collapsible
                    key={item.title}
                    open={isOpen}
                    onOpenChange={() => toggleSubmenu(item.title)}
                    className="w-full"
                  >
                    <SidebarMenuItem>
                      <CollapsibleTrigger asChild>
                        <SidebarMenuButton
                          className="data-[state=open]:bg-sidebar-accent data-[state=open]:text-sidebar-accent-foreground w-full justify-between"
                          isActive={isSubmenuActive}
                        >
                          <div className="flex items-center gap-2">
                            <Icon className="size-4" />
                            <span>{item.title}</span>
                          </div>
                          {isOpen ? (
                            <ChevronDown className="h-4 w-4" />
                          ) : (
                            <ChevronRight className="h-4 w-4" />
                          )}
                        </SidebarMenuButton>
                      </CollapsibleTrigger>
                      <CollapsibleContent>
                        <SidebarMenuSub>
                          {item.subItems.map((subItem) => {
                            const SubIcon = subItem.icon;
                            const isSubActive = pathname === subItem.href;
                            return (
                              <SidebarMenuSubItem key={subItem.href}>
                                <SidebarMenuSubButton
                                  asChild
                                  isActive={isSubActive}
                                >
                                  <Link
                                    href={subItem.href!}
                                    onClick={handleNavClick}
                                  >
                                    <SubIcon className="size-4" />
                                    <span>{subItem.title}</span>
                                  </Link>
                                </SidebarMenuSubButton>
                              </SidebarMenuSubItem>
                            );
                          })}
                        </SidebarMenuSub>
                      </CollapsibleContent>
                    </SidebarMenuItem>
                  </Collapsible>
                );
              }

              return (
                <SidebarMenuItem key={item.href}>
                  <SidebarMenuButton
                    asChild
                    isActive={isActive}
                    tooltip={item.title}
                  >
                    <Link href={item.href!} onClick={handleNavClick}>
                      <Icon className="size-4" />
                      <span>{item.title}</span>
                    </Link>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              );
            })}
          </SidebarMenu>
        </SidebarGroup>

        <SidebarGroup className="mt-auto">
          <SidebarGroupLabel>Support</SidebarGroupLabel>
          <SidebarMenu>
            {supportNavItems.map((item) => {
              const Icon = item.icon;
              const isActive = pathname === item.href || (item.title === "Settings" && pathname.startsWith("/settings"));

              return (
                <SidebarMenuItem key={item.href}>
                  <SidebarMenuButton
                    asChild
                    isActive={isActive}
                    tooltip={item.title}
                  >
                    <Link href={item.href!} onClick={handleNavClick}>
                      <Icon className="size-4" />
                      <span>{item.title}</span>
                    </Link>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              );
            })}
          </SidebarMenu>
        </SidebarGroup>
      </SidebarContent>

      {/* Fixed Sticky Footer with User Profile Dropdown */}
      <SidebarFooter className="shrink-0 border-t border-sidebar-border p-2">
        <SidebarMenu>
          <SidebarMenuItem>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <SidebarMenuButton
                  size="lg"
                  className="w-full data-[state=open]:bg-sidebar-accent data-[state=open]:text-sidebar-accent-foreground cursor-pointer group-data-[collapsible=icon]:p-0 group-data-[collapsible=icon]:justify-center hover:bg-sidebar-accent outline-none focus-visible:ring-2 focus-visible:ring-sidebar-ring"
                  tooltip={user.name || user.email}
                >
                  <UserAvatar user={user} className="h-8 w-8 shrink-0 rounded-lg" />
                  <div className="grid flex-1 text-left text-sm leading-tight group-data-[collapsible=icon]:hidden">
                    <span className="truncate font-semibold text-xs text-foreground">
                      {user.name || user.email}
                    </span>
                    <span className="truncate text-[10px] text-muted-foreground">
                      {user.email}
                    </span>
                    <span className="mt-0.5 truncate text-[10px] font-medium text-muted-foreground/70">
                      {user.role
                        ? user.role.charAt(0) + user.role.slice(1).toLowerCase()
                        : "Admin"}
                    </span>
                  </div>
                  <ChevronDown className="ml-auto size-4 text-muted-foreground group-data-[collapsible=icon]:hidden shrink-0" />
                </SidebarMenuButton>
              </DropdownMenuTrigger>
              <DropdownMenuContent
                className="w-56 rounded-xl shadow-lg border border-border"
                side={isMobile ? "bottom" : "right"}
                align="end"
                sideOffset={8}
              >
                <DropdownMenuLabel className="p-0 font-normal">
                  <div className="flex items-center gap-2.5 px-2.5 py-2 text-left text-sm">
                    <UserAvatar user={user} className="h-8 w-8 rounded-lg shrink-0" />
                    <div className="grid flex-1 text-left text-sm leading-tight">
                      <span className="truncate font-semibold text-xs text-foreground">
                        {user.name || user.email}
                      </span>
                      <span className="truncate text-[10px] text-muted-foreground">
                        {user.email}
                      </span>
                      <span className="mt-0.5 text-[10px] font-medium text-primary">
                        {user.role
                          ? user.role.charAt(0) + user.role.slice(1).toLowerCase()
                          : "Admin"}
                      </span>
                    </div>
                  </div>
                </DropdownMenuLabel>
                <DropdownMenuSeparator />
                <DropdownMenuSub>
                  <DropdownMenuSubTrigger className="cursor-pointer flex items-center gap-2">
                    <Sun className="size-4 rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0 text-amber-500" />
                    <Moon className="absolute size-4 rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100 text-blue-400" />
                    <span className="ml-5">Theme</span>
                  </DropdownMenuSubTrigger>
                  <DropdownMenuSubContent className="rounded-xl border border-border shadow-md min-w-36">
                    <DropdownMenuItem
                      onClick={() => setTheme("light")}
                      className={`cursor-pointer flex items-center justify-between text-xs rounded-lg px-2.5 py-1.5 ${
                        theme === "light" ? "font-semibold text-primary bg-primary/10" : ""
                      }`}
                    >
                      <div className="flex items-center gap-2">
                        <Sun className="size-3.5 text-amber-500" />
                        <span>Light</span>
                      </div>
                      {theme === "light" && <Check className="size-3.5 text-primary" />}
                    </DropdownMenuItem>
                    <DropdownMenuItem
                      onClick={() => setTheme("dark")}
                      className={`cursor-pointer flex items-center justify-between text-xs rounded-lg px-2.5 py-1.5 ${
                        theme === "dark" ? "font-semibold text-primary bg-primary/10" : ""
                      }`}
                    >
                      <div className="flex items-center gap-2">
                        <Moon className="size-3.5 text-blue-400" />
                        <span>Dark</span>
                      </div>
                      {theme === "dark" && <Check className="size-3.5 text-primary" />}
                    </DropdownMenuItem>
                    <DropdownMenuItem
                      onClick={() => setTheme("system")}
                      className={`cursor-pointer flex items-center justify-between text-xs rounded-lg px-2.5 py-1.5 ${
                        theme === "system" ? "font-semibold text-primary bg-primary/10" : ""
                      }`}
                    >
                      <div className="flex items-center gap-2">
                        <Monitor className="size-3.5 text-muted-foreground" />
                        <span>System</span>
                      </div>
                      {theme === "system" && <Check className="size-3.5 text-primary" />}
                    </DropdownMenuItem>
                  </DropdownMenuSubContent>
                </DropdownMenuSub>
                <DropdownMenuItem asChild className="cursor-pointer">
                  <Link href="/settings/account" className="flex items-center gap-2">
                    <Settings className="size-4 text-muted-foreground" />
                    <span>Account Settings</span>
                  </Link>
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem
                  variant="destructive"
                  onClick={handleLogout}
                  disabled={isLoading}
                  className="cursor-pointer text-destructive focus:text-destructive focus:bg-destructive/10 font-medium"
                >
                  {isLoading ? (
                    <Loader2 className="mr-2 size-4 animate-spin" />
                  ) : (
                    <LogOut className="mr-2 size-4" />
                  )}
                  <span>Log out</span>
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarFooter>
    </Sidebar>
  );
}

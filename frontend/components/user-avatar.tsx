"use client";

import * as React from "react";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";

interface UserAvatarProps {
  user: {
    name?: string | null;
    email?: string | null;
    image?: string | null;
  };
  className?: string;
}

export function UserAvatar({ user, className }: UserAvatarProps) {
  const nameOrEmail = user.name || user.email || "User";
  const initials = nameOrEmail
    .split(" ")
    .map((n) => n[0])
    .join("")
    .slice(0, 2)
    .toUpperCase();

  return (
    <Avatar className={className}>
      {user.image && <AvatarImage src={user.image} alt={nameOrEmail} />}
      <AvatarFallback className="bg-primary/10 text-primary font-semibold text-xs">
        {initials}
      </AvatarFallback>
    </Avatar>
  );
}

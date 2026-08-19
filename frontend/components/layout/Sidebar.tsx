"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { LayoutDashboard, Network, Search, BarChart2, Info, Film } from "lucide-react";
import clsx from "clsx";

const navItems = [
  { name: "Overview", href: "/", icon: LayoutDashboard },
  { name: "Clusters", href: "/clusters", icon: Network },
  { name: "Explorer", href: "/explorer", icon: Search },
  { name: "Evaluation", href: "/evaluation", icon: BarChart2 },
  { name: "Methodology", href: "/methodology", icon: Info },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <div className="w-64 h-screen border-r border-border bg-surface flex flex-col shrink-0 sticky top-0">
      <div className="h-16 flex items-center px-6 border-b border-border">
        <Film className="w-6 h-6 text-accent mr-3" />
        <span className="font-bold text-lg tracking-tight text-foreground">
          CineVault
        </span>
      </div>
      
      <div className="flex-1 py-6 px-4 space-y-1">
        <div className="text-xs font-semibold text-muted uppercase tracking-wider mb-4 px-2">
          Content Segmentation
        </div>
        {navItems.map((item) => {
          const isActive = pathname === item.href;
          const Icon = item.icon;
          return (
            <Link
              key={item.name}
              href={item.href}
              className={clsx(
                "flex items-center px-3 py-2.5 rounded-md text-sm font-medium transition-colors",
                isActive
                  ? "bg-accent/10 text-accent"
                  : "text-muted hover:text-foreground hover:bg-surfaceHover"
              )}
            >
              <Icon className={clsx("w-5 h-5 mr-3", isActive ? "text-accent" : "text-muted")} />
              {item.name}
            </Link>
          );
        })}
      </div>
      
      <div className="p-4 border-t border-border">
        <div className="text-xs text-muted flex items-center justify-between">
          <span>Model Status</span>
          <span className="flex items-center">
            <span className="w-2 h-2 rounded-full bg-green-500 mr-2"></span>
            Online
          </span>
        </div>
      </div>
    </div>
  );
}

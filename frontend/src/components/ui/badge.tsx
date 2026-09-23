import type { HTMLAttributes } from "react";

import { cn } from "../../lib/utils";

export function Badge({ className, ...props }: HTMLAttributes<HTMLSpanElement>) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full border border-ink/10 bg-white/70 px-2.5 py-1 text-[11px] font-semibold uppercase tracking-[0.12em] text-ink/65",
        className
      )}
      {...props}
    />
  );
}

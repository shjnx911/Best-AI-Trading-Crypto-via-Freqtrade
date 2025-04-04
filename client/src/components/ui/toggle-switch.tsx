import * as React from "react";
import { cn } from "@/lib/utils";

export interface ToggleSwitchProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  size?: "sm" | "md" | "lg";
}

const sizeClasses = {
  sm: {
    container: "w-8 h-4",
    slider: "h-3 w-3",
    offset: "translate-x-4"
  },
  md: {
    container: "w-11 h-6",
    slider: "h-5 w-5",
    offset: "translate-x-5"
  },
  lg: {
    container: "w-14 h-7",
    slider: "h-6 w-6",
    offset: "translate-x-7"
  }
};

export const ToggleSwitch = React.forwardRef<HTMLInputElement, ToggleSwitchProps>(
  ({ className, label, size = "md", ...props }, ref) => {
    const id = React.useId();
    const sizeClass = sizeClasses[size];

    return (
      <div className={cn("flex items-center", className)}>
        {label && (
          <label 
            htmlFor={id} 
            className="mr-2 text-sm text-textSecondary cursor-pointer"
          >
            {label}
          </label>
        )}
        <label className={cn("relative inline-block", sizeClass.container)}>
          <input
            id={id}
            type="checkbox"
            className="opacity-0 w-0 h-0"
            ref={ref}
            {...props}
          />
          <span 
            className={cn(
              "absolute cursor-pointer inset-0 rounded-full bg-border transition-colors",
              props.checked ? "bg-primary" : "bg-border"
            )}
          >
            <span 
              className={cn(
                "absolute left-0.5 bottom-0.5 bg-textPrimary rounded-full transition-transform",
                sizeClass.slider,
                props.checked ? sizeClass.offset : "translate-x-0"
              )}
            ></span>
          </span>
        </label>
      </div>
    );
  }
);

ToggleSwitch.displayName = "ToggleSwitch";

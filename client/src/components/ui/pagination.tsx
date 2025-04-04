import * as React from "react";
import { ChevronLeft, ChevronRight, MoreHorizontal } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";

export interface PaginationProps extends React.HTMLAttributes<HTMLDivElement> {}

const Pagination = ({ className, ...props }: PaginationProps) => (
  <div className={cn("flex items-center justify-center", className)} {...props} />
);

const PaginationContent = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn("flex flex-row items-center gap-1", className)}
    {...props}
  />
));
PaginationContent.displayName = "PaginationContent";

const PaginationItem = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div ref={ref} className={cn("", className)} {...props} />
));
PaginationItem.displayName = "PaginationItem";

type PaginationLinkProps = {
  isActive?: boolean;
  disabled?: boolean;
  children?: React.ReactNode;
  onClick?: () => void;
} & Omit<React.HTMLAttributes<HTMLButtonElement>, "onClick">;

const PaginationLink = ({
  className,
  isActive,
  disabled,
  children,
  onClick,
  ...props
}: PaginationLinkProps) => (
  <Button
    aria-current={isActive ? "page" : undefined}
    disabled={disabled}
    className={cn(
      "h-9 w-9 p-0 text-center font-medium",
      isActive ? "bg-primary text-primary-foreground hover:bg-primary/90" : "bg-surfaceLight text-textPrimary hover:bg-surfaceLight/80",
      className
    )}
    onClick={onClick}
    {...props}
  >
    {children}
  </Button>
);

const PaginationPrevious = ({
  className,
  onClick,
  disabled,
  ...props
}: React.ComponentProps<typeof PaginationLink>) => (
  <PaginationLink
    aria-label="Go to previous page"
    size="default"
    className={cn("gap-1", className)}
    onClick={onClick}
    disabled={disabled}
    {...props}
  >
    <ChevronLeft className="h-4 w-4" />
  </PaginationLink>
);

const PaginationNext = ({
  className,
  onClick,
  disabled,
  ...props
}: React.ComponentProps<typeof PaginationLink>) => (
  <PaginationLink
    aria-label="Go to next page"
    size="default"
    className={cn("gap-1", className)}
    onClick={onClick}
    disabled={disabled}
    {...props}
  >
    <ChevronRight className="h-4 w-4" />
  </PaginationLink>
);

const PaginationEllipsis = ({
  className,
  ...props
}: React.HTMLAttributes<HTMLSpanElement>) => (
  <span
    aria-hidden
    className={cn("flex h-9 w-9 items-center justify-center", className)}
    {...props}
  >
    <MoreHorizontal className="h-4 w-4 text-textSecondary" />
  </span>
);

Pagination.Content = PaginationContent;
Pagination.Item = PaginationItem;
Pagination.Link = PaginationLink;
Pagination.Previous = PaginationPrevious;
Pagination.Next = PaginationNext;
Pagination.Ellipsis = PaginationEllipsis;

export { Pagination };

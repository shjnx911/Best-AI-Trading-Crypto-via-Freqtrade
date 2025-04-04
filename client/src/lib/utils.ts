import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatCurrency(value: number): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2
  }).format(value);
}

export function formatNumber(value: number, digits = 2): string {
  return new Intl.NumberFormat('en-US', {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits
  }).format(value);
}

export function formatPercentage(value: number): string {
  return new Intl.NumberFormat('en-US', {
    style: 'percent',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2
  }).format(value / 100);
}

export function formatDate(date: string | Date): string {
  return new Date(date).toLocaleDateString('en-US', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  });
}

export function formatDateTime(date: string | Date): string {
  return new Date(date).toLocaleString('en-US', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false
  });
}

export function getStatusColor(value: number, thresholds: [number, number] = [0, 0]): string {
  const [negativeThreshold, positiveThreshold] = thresholds;
  
  if (value < negativeThreshold) return 'text-secondary';
  if (value > positiveThreshold) return 'text-primary';
  return 'text-textPrimary';
}

export function getPnlColor(value: number): string {
  if (value > 0) return 'text-primary';
  if (value < 0) return 'text-secondary';
  return 'text-textPrimary';
}

export function truncateString(str: string, length = 20): string {
  if (str.length <= length) return str;
  return `${str.slice(0, length)}...`;
}

export function calculateWinRate(wins: number, losses: number): number {
  if (wins + losses === 0) return 0;
  return (wins / (wins + losses)) * 100;
}

export function calculateRiskRewardRatio(avgProfit: number, avgLoss: number): number {
  if (avgLoss === 0) return 0;
  return Math.abs(avgProfit / avgLoss);
}

export function parseApiErrors(error: any): string {
  if (error?.message) return error.message;
  if (typeof error === 'string') return error;
  return 'An unknown error occurred';
}

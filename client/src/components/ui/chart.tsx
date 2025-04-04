import { useEffect, useRef, useState } from 'react';
import { createChart, ColorType, IChartApi } from 'lightweight-charts';
import { cn } from '@/lib/utils';

interface ChartProps {
  data: any[];
  width?: number;
  height?: number;
  chartType?: 'candle' | 'line' | 'area';
  timeVisible?: boolean;
  indicators?: {
    name: string;
    data: any[];
    color: string;
  }[];
  onCrosshairMove?: (params: any) => void;
  className?: string;
  toolbarVisible?: boolean;
  xKey?: string;
  yKey?: string;
  yName?: string;
  color?: string;
}

export function Chart({
  data,
  width,
  height = 300,
  chartType = 'candle',
  timeVisible = true,
  indicators = [],
  onCrosshairMove,
  className,
  toolbarVisible = true,
  xKey = 'time',
  yKey,
  yName,
  color = '#2962FF'
}: ChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const [chartApi, setChartApi] = useState<IChartApi | null>(null);
  
  useEffect(() => {
    if (!chartContainerRef.current) return;

    const chart = createChart(chartContainerRef.current, {
      width: width || chartContainerRef.current.clientWidth,
      height: height,
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#848E9C',
        fontSize: 12,
        fontFamily: 'Inter, sans-serif',
      },
      grid: {
        vertLines: {
          color: '#2B2B3B',
          style: 1,
          visible: true,
        },
        horzLines: {
          color: '#2B2B3B',
          style: 1,
          visible: true,
        },
      },
      crosshair: {
        mode: 1,
        vertLine: {
          width: 1,
          color: '#2962FF',
          style: 0,
          visible: true,
          labelVisible: false,
        },
        horzLine: {
          width: 1,
          color: '#2962FF',
          style: 0,
          visible: true,
          labelVisible: true,
        },
      },
      timeScale: {
        timeVisible: timeVisible,
        secondsVisible: false,
        borderColor: '#2B2B3B',
      },
      rightPriceScale: {
        borderColor: '#2B2B3B',
      },
      handleScroll: {
        vertTouchDrag: true,
      },
    });

    setChartApi(chart);

    try {
      // Add series based on chart type
      if (data && data.length > 0) {
        if (chartType === 'candle') {
          try {
            // First try the V5+ API
            const series = (chart as any).createSeries('candlestick', {
              upColor: '#0ECB81',
              downColor: '#F6465D',
              borderVisible: false,
              wickUpColor: '#0ECB81',
              wickDownColor: '#F6465D',
            });
            series.setData(data);
          } catch (e) {
            // Fallback to the older API (V4 and below)
            const series = (chart as any).addCandlestickSeries({
              upColor: '#0ECB81',
              downColor: '#F6465D',
              borderVisible: false,
              wickUpColor: '#0ECB81',
              wickDownColor: '#F6465D',
            });
            series.setData(data);
          }
        } else if (chartType === 'line') {
          let mappedData = data;
          
          // Map the data if keys are provided
          if (yKey) {
            mappedData = data.map(item => ({
              time: item[xKey],
              value: item[yKey]
            }));
          }
          
          try {
            // First try the V5+ API
            const series = (chart as any).createSeries('line', {
              color: color,
              lineWidth: 2,
            });
            series.setData(mappedData);
          } catch (e) {
            // Fallback to the older API (V4 and below)
            const series = (chart as any).addLineSeries({
              color: color,
              lineWidth: 2,
            });
            series.setData(mappedData);
          }
        } else if (chartType === 'area') {
          let mappedData = data;
          
          // Map the data if keys are provided
          if (yKey) {
            mappedData = data.map(item => ({
              time: item[xKey],
              value: item[yKey]
            }));
          }
          
          try {
            // First try the V5+ API
            const series = (chart as any).createSeries('area', {
              topColor: 'rgba(41, 98, 255, 0.3)',
              bottomColor: 'rgba(41, 98, 255, 0.0)',
              lineColor: '#2962FF',
              lineWidth: 2,
            });
            series.setData(mappedData);
          } catch (e) {
            // Fallback to the older API (V4 and below)
            const series = (chart as any).addAreaSeries({
              topColor: 'rgba(41, 98, 255, 0.3)',
              bottomColor: 'rgba(41, 98, 255, 0.0)',
              lineColor: '#2962FF',
              lineWidth: 2,
            });
            series.setData(mappedData);
          }
        }
      }

      // Add indicators if provided
      if (indicators && indicators.length > 0) {
        indicators.forEach((indicator) => {
          if (indicator.data && indicator.data.length > 0) {
            try {
              // First try the V5+ API
              const series = (chart as any).createSeries('line', {
                color: indicator.color,
                lineWidth: 1,
                priceLineVisible: false,
              });
              series.setData(indicator.data);
            } catch (e) {
              // Fallback to the older API (V4 and below)
              const series = (chart as any).addLineSeries({
                color: indicator.color,
                lineWidth: 1,
                priceLineVisible: false,
              });
              series.setData(indicator.data);
            }
          }
        });
      }
    } catch (error) {
      console.error('Error setting up chart series:', error);
    }

    // Set up crosshair move handler
    if (onCrosshairMove) {
      chart.subscribeCrosshairMove(onCrosshairMove);
    }

    // Handle resize
    const handleResize = () => {
      if (chartContainerRef.current && chart) {
        chart.applyOptions({ 
          width: width || chartContainerRef.current.clientWidth 
        });
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
      setChartApi(null);
    };
  }, [data, width, height, chartType, timeVisible, indicators, xKey, yKey, color]);

  return (
    <div className={cn("chart-container relative", className)}>
      <div ref={chartContainerRef} className="chart-placeholder" />
      
      {toolbarVisible && (
        <div className="absolute top-2 right-2 flex space-x-1">
          {chartApi && (
            <>
              <button 
                className="bg-surfaceLight text-textSecondary rounded px-2 py-1 text-xs hover:bg-surfaceLight/80 transition"
                onClick={() => chartApi.timeScale().fitContent()}
              >
                <i className="ri-fullscreen-line"></i>
              </button>
              <button 
                className="bg-surfaceLight text-textSecondary rounded px-2 py-1 text-xs hover:bg-surfaceLight/80 transition"
                onClick={() => chartApi.timeScale().scrollToRealTime()}
              >
                <i className="ri-time-line"></i>
              </button>
            </>
          )}
        </div>
      )}
      
      {/* Indicators legend */}
      {indicators.length > 0 && (
        <div className="absolute top-2 left-2 flex flex-wrap gap-2">
          {indicators.map((indicator, index) => (
            <div key={index} className="px-2 py-1 rounded bg-surfaceLight text-xs flex items-center">
              <span 
                className="inline-block w-2 h-2 rounded-full mr-1" 
                style={{ backgroundColor: indicator.color }}
              />
              <span>{indicator.name}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

'use client';

import React, { useEffect, useRef, useState } from 'react';
import uPlot, { Options, AlignedData } from 'uplot';
import 'uplot/dist/uPlot.min.css';

interface TelemetryChartProps {
    title: string;
    data: AlignedData;
    series: { label: string; stroke: string }[];
}

export default function TelemetryChart({ title, data, series }: TelemetryChartProps) {
    const chartRef = useRef<HTMLDivElement>(null);
    const uPlotRef = useRef<uPlot | null>(null);

    useEffect(() => {
        if (!chartRef.current) return;

        const options: Options = {
            title: title,
            width: 800,
            height: 400,
            series: [
                {}, // x-axis (time)
                ...series.map(s => ({
                    label: s.label,
                    stroke: s.stroke,
                    width: 2,
                })),
            ],
            scales: {
                x: {
                    time: false,
                },
            },
        };

        const u = new uPlot(options, data, chartRef.current);
        uPlotRef.current = u;

        return () => {
            u.destroy();
            uPlotRef.current = null;
        };
    }, [title, series]); // Re-init if config changes

    // Update data efficiently without destroying chart
    useEffect(() => {
        if (uPlotRef.current && data) {
            uPlotRef.current.setData(data);
        }
    }, [data]);

    return (
        <div className="p-4 bg-gray-900 rounded-lg shadow-lg border border-gray-700">
            <div ref={chartRef} className="w-full" />
        </div>
    );
}

'use client';

import React from 'react';
import TelemetryChart from '../../components/telemetry/TelemetryChart';
import LogViewer from '../../components/logs/LogViewer';
import EcuMapChart from '../../components/telemetry/EcuMapChart';

export default function TelemetryPage() {
    // Mock Telemetry Data for uPlot
    // x-axis: timestamps, y-axis: RPM, Speed, AFR
    const len = 1000;
    const x = Array.from({ length: len }, (_, i) => i);
    const rpm = Array.from({ length: len }, (_, i) => 800 + Math.sin(i * 0.05) * 500 + Math.random() * 50);
    const speed = Array.from({ length: len }, (_, i) => 20 + Math.cos(i * 0.02) * 10 + Math.random() * 2);

    const data: any = [x, rpm, speed];

    const series = [
        { label: "RPM", stroke: "red" },
        { label: "Speed (mph)", stroke: "cyan" }
    ];

    // Mock 3D Data for ECharts [x, y, z] -> [RPM, Load, AFR]
    const mapData: number[][] = [];
    for (let r = 0; r <= 20; r++) {
        for (let l = 0; l <= 20; l++) {
            const rpmVal = r * 300 + 1000;
            const loadVal = l * 5;
            // Target AFR: Richer (lower) at high load/rpm
            const afr = 14.7 - (loadVal / 100 * 2.0) - (rpmVal / 7000 * 1.5);
            mapData.push([rpmVal, loadVal, afr]);
        }
    }

    return (
        <div className="min-h-screen bg-gray-950 text-white p-8">
            <header className="mb-8">
                <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
                    HOPE Telemetry Dashboard
                </h1>
                <p className="text-gray-400">Real-time vehicle analytics and diagnostics</p>
            </header>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
                {/* Telemetry Chart View (uPlot) */}
                <div className="bg-gray-900/50 p-6 rounded-xl border border-gray-800">
                    <h2 className="text-xl font-semibold mb-4 text-blue-300">Live Sensor Data (High Frequency)</h2>
                    <TelemetryChart
                        title="Engine Parameters"
                        data={data}
                        series={series}
                    />
                </div>

                {/* 3D ECU Map View (ECharts) */}
                <div className="bg-gray-900/50 p-6 rounded-xl border border-gray-800">
                    <h2 className="text-xl font-semibold mb-4 text-purple-300">Volumetric Efficiency (VE) Map</h2>
                    <div className="bg-gray-900 rounded-lg overflow-hidden border border-gray-700">
                        <EcuMapChart
                            title="Fuel Map (Target AFR)"
                            data={mapData}
                        />
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-1 gap-8">
                {/* Log Viewer Section */}
                <div className="bg-gray-900/50 p-6 rounded-xl border border-gray-800">
                    <h2 className="text-xl font-semibold mb-4 text-green-300">System Logs</h2>
                    <LogViewer />
                </div>
            </div>

            <div className="mt-8 p-4 bg-gray-900 rounded-lg border border-red-900/30">
                <h3 className="text-lg font-semibold text-red-400 mb-2">Active Alerts</h3>
                <ul className="list-disc list-inside text-gray-300 space-y-1">
                    <li>[WARN] O2 Sensor Bank 1 Rich Condition</li>
                    <li>[INFO] Genetic Optimizer: Generation 42 complete</li>
                </ul>
            </div>
        </div>
    );
}

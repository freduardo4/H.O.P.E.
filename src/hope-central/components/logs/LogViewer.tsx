'use client';

import React, { useState } from 'react';
import { AgGridReact } from 'ag-grid-react';
import "ag-grid-community/styles/ag-grid.css";
import "ag-grid-community/styles/ag-theme-quartz.css";
import { ColDef } from 'ag-grid-community';

interface LogEntry {
    timestamp: string;
    level: string;
    module: string;
    message: string;
}

const MOCK_LOGS: LogEntry[] = [
    { timestamp: "2026-01-29 10:00:01", level: "INFO", module: "ECU_Service", message: "Connecting to ECU..." },
    { timestamp: "2026-01-29 10:00:02", level: "INFO", module: "J2534", message: "Link established (ISO 15765-4)" },
    { timestamp: "2026-01-29 10:00:05", level: "WARN", module: "Tuning_Optimizer", message: "Knock threshold approached (Cyl 3)" },
    { timestamp: "2026-01-29 10:00:08", level: "ERROR", module: "UDS_Protocol", message: "Security Access Denied (Seed: 0xDEADBEEF)" },
];

export default function LogViewer() {
    const [rowData] = useState<LogEntry[]>(MOCK_LOGS);

    const [colDefs] = useState<ColDef<LogEntry>[]>([
        { field: "timestamp", sortable: true, filter: true },
        { field: "level", sortable: true, filter: true },
        { field: "module", sortable: true, filter: true },
        { field: "message", sortable: true, filter: true, flex: 1 },
    ]);

    return (
        <div className="ag-theme-quartz-dark h-[400px] w-full border border-gray-700 rounded-lg overflow-hidden">
            <AgGridReact
                rowData={rowData}
                columnDefs={colDefs}
                animateRows={true}
            />
        </div>
    );
}

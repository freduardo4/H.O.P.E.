export declare class LogReadingDto {
    sessionId: string;
    timestamp?: string;
    pid: string;
    name: string;
    value: number;
    unit: string;
    rawResponse?: string;
}
export declare class LogReadingsBatchDto {
    readings: LogReadingDto[];
}

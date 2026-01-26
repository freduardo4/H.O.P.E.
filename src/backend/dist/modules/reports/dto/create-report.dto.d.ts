import { ReportType } from '../entities/report.entity';
export declare class CreateReportDto {
    vehicleId: string;
    customerId?: string;
    diagnosticSessionId?: string;
    reportType: ReportType;
    title: string;
    description?: string;
    data?: {
        dtcCodes?: string[];
        obd2Readings?: any[];
        anomalies?: any[];
        beforePower?: number;
        afterPower?: number;
        beforeTorque?: number;
        afterTorque?: number;
        fuelConsumptionImprovement?: number;
        modifications?: string[];
        [key: string]: any;
    };
}

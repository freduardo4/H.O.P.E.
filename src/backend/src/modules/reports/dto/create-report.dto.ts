import { IsString, IsEnum, IsOptional, IsUUID, IsObject } from 'class-validator';
import { ReportType } from '../entities/report.entity';

export class CreateReportDto {
    @IsUUID()
    vehicleId: string;

    @IsUUID()
    @IsOptional()
    customerId?: string;

    @IsUUID()
    @IsOptional()
    diagnosticSessionId?: string;

    @IsEnum(ReportType)
    reportType: ReportType;

    @IsString()
    title: string;

    @IsString()
    @IsOptional()
    description?: string;

    @IsObject()
    @IsOptional()
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

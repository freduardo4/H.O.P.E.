import { IsString, IsBoolean, IsOptional, IsNumber } from 'class-validator';

export class SafetyEventDto {
    @IsString()
    eventType: string;

    @IsString()
    ecuId: string;

    @IsOptional()
    @IsNumber()
    voltage?: number;

    @IsBoolean()
    success: boolean;

    @IsOptional()
    @IsString()
    message?: string;

    @IsOptional()
    @IsString()
    metadata?: string;
}

import {
    IsUUID,
    IsString,
    IsNumber,
    IsOptional,
    IsDateString,
    IsArray,
    ValidateNested,
} from 'class-validator';
import { Type } from 'class-transformer';

export class LogReadingDto {
    @IsUUID()
    sessionId: string;

    @IsDateString()
    @IsOptional()
    timestamp?: string;

    @IsString()
    pid: string;

    @IsString()
    name: string;

    @IsNumber()
    value: number;

    @IsString()
    unit: string;

    @IsString()
    @IsOptional()
    rawResponse?: string;
}

export class LogReadingsBatchDto {
    @IsArray()
    @ValidateNested({ each: true })
    @Type(() => LogReadingDto)
    readings: LogReadingDto[];
}

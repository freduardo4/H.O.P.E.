import {
    IsString,
    IsOptional,
    IsArray,
    IsObject,
} from 'class-validator';

export class EndSessionDto {
    @IsString()
    @IsOptional()
    notes?: string;

    @IsArray()
    @IsOptional()
    dtcCodes?: string[];

    @IsObject()
    @IsOptional()
    performanceMetrics?: {
        maxRpm?: number;
        maxSpeed?: number;
        maxBoost?: number;
        avgLoad?: number;
    };

    @IsObject()
    @IsOptional()
    ecuSnapshot?: Record<string, any>;
}

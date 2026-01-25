import {
    IsUUID,
    IsEnum,
    IsOptional,
    IsNumber,
    IsString,
    IsArray,
    IsObject,
    Min,
} from 'class-validator';
import { SessionType } from '../entities/diagnostic-session.entity';

export class CreateSessionDto {
    @IsUUID()
    vehicleId: string;

    @IsEnum(SessionType)
    @IsOptional()
    type?: SessionType;

    @IsNumber()
    @IsOptional()
    @Min(0)
    mileageAtSession?: number;

    @IsString()
    @IsOptional()
    notes?: string;
}

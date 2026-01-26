import { PartialType } from '@nestjs/mapped-types';
import { CreateECUCalibrationDto } from './create-ecu-calibration.dto';
import { IsBoolean, IsOptional } from 'class-validator';

export class UpdateECUCalibrationDto extends PartialType(CreateECUCalibrationDto) {
    @IsBoolean()
    @IsOptional()
    isActive?: boolean;
}

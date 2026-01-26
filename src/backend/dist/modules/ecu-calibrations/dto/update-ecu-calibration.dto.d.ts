import { CreateECUCalibrationDto } from './create-ecu-calibration.dto';
declare const UpdateECUCalibrationDto_base: import("@nestjs/mapped-types").MappedType<Partial<CreateECUCalibrationDto>>;
export declare class UpdateECUCalibrationDto extends UpdateECUCalibrationDto_base {
    isActive?: boolean;
}
export {};

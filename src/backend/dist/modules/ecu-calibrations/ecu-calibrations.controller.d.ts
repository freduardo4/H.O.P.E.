import { ECUCalibrationsService, PaginatedCalibrations } from './ecu-calibrations.service';
import { CreateECUCalibrationDto, UpdateECUCalibrationDto } from './dto';
import { ECUCalibration, CalibrationType } from './entities/ecu-calibration.entity';
import { User } from '../auth/entities/user.entity';
export declare class ECUCalibrationsController {
    private readonly calibrationsService;
    constructor(calibrationsService: ECUCalibrationsService);
    uploadFile(user: User, file: Express.Multer.File, dto: CreateECUCalibrationDto): Promise<ECUCalibration>;
    findAll(user: User, vehicleId?: string, customerId?: string, calibrationType?: CalibrationType, page?: number, limit?: number): Promise<PaginatedCalibrations>;
    findOne(user: User, id: string): Promise<ECUCalibration>;
    getDownloadUrl(user: User, id: string): Promise<{
        url: string;
        expiresIn: number;
    }>;
    getVersionHistory(user: User, vehicleId: string): Promise<ECUCalibration[]>;
    update(user: User, id: string, dto: UpdateECUCalibrationDto): Promise<ECUCalibration>;
    remove(user: User, id: string): Promise<{
        message: string;
    }>;
}

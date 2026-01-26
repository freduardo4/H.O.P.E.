import { Repository } from 'typeorm';
import { ECUCalibration } from './entities/ecu-calibration.entity';
import { CreateECUCalibrationDto, UpdateECUCalibrationDto } from './dto';
export interface UploadECUFileParams {
    tenantId: string;
    dto: CreateECUCalibrationDto;
    fileBuffer: Buffer;
    uploadedBy: string;
}
export interface PaginatedCalibrations {
    data: ECUCalibration[];
    total: number;
    page: number;
    limit: number;
    totalPages: number;
}
export declare class ECUCalibrationsService {
    private readonly calibrationRepo;
    private s3Client;
    private bucketName;
    constructor(calibrationRepo: Repository<ECUCalibration>);
    uploadFile(params: UploadECUFileParams): Promise<ECUCalibration>;
    findAll(params: {
        tenantId: string;
        vehicleId?: string;
        customerId?: string;
        calibrationType?: string;
        page?: number;
        limit?: number;
    }): Promise<PaginatedCalibrations>;
    findOne(tenantId: string, id: string): Promise<ECUCalibration>;
    getDownloadUrl(tenantId: string, id: string, expiresIn?: number): Promise<string>;
    update(tenantId: string, id: string, dto: UpdateECUCalibrationDto): Promise<ECUCalibration>;
    remove(tenantId: string, id: string): Promise<void>;
    getVersionHistory(tenantId: string, vehicleId: string): Promise<ECUCalibration[]>;
}

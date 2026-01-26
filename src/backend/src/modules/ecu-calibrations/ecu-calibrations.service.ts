import { Injectable, NotFoundException, BadRequestException } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { S3Client, PutObjectCommand, GetObjectCommand, DeleteObjectCommand } from '@aws-sdk/client-s3';
import { getSignedUrl } from '@aws-sdk/s3-request-presigner';
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

@Injectable()
export class ECUCalibrationsService {
    private s3Client: S3Client;
    private bucketName: string;

    constructor(
        @InjectRepository(ECUCalibration)
        private readonly calibrationRepo: Repository<ECUCalibration>,
    ) {
        this.bucketName = process.env.AWS_S3_BUCKET || 'hope-ecu-calibrations';
        this.s3Client = new S3Client({
            region: process.env.AWS_S3_REGION || 'us-east-1',
        });
    }

    async uploadFile(params: UploadECUFileParams): Promise<ECUCalibration> {
        const { tenantId, dto, fileBuffer, uploadedBy } = params;

        // Generate unique S3 key
        const timestamp = Date.now();
        const s3Key = `${tenantId}/ecu-calibrations/${dto.vehicleId}/${timestamp}-${dto.fileName}`;

        try {
            // Upload to S3
            await this.s3Client.send(
                new PutObjectCommand({
                    Bucket: this.bucketName,
                    Key: s3Key,
                    Body: fileBuffer,
                    ContentType: 'application/octet-stream',
                    Metadata: {
                        vehicleId: dto.vehicleId,
                        calibrationType: dto.calibrationType,
                        checksum: dto.checksum,
                    },
                }),
            );

            // Determine version number
            let version = 1;
            if (dto.previousVersionId) {
                const previousVersion = await this.calibrationRepo.findOne({
                    where: { id: dto.previousVersionId, tenantId },
                });
                if (previousVersion) {
                    version = previousVersion.version + 1;
                }
            }

            // Save metadata to database
            const calibration = this.calibrationRepo.create({
                ...dto,
                tenantId,
                s3Key,
                s3Bucket: this.bucketName,
                version,
                uploadedBy,
            });

            return await this.calibrationRepo.save(calibration);
        } catch (error) {
            throw new BadRequestException(`Failed to upload ECU file: ${error.message}`);
        }
    }

    async findAll(params: {
        tenantId: string;
        vehicleId?: string;
        customerId?: string;
        calibrationType?: string;
        page?: number;
        limit?: number;
    }): Promise<PaginatedCalibrations> {
        const { tenantId, vehicleId, customerId, calibrationType, page = 1, limit = 20 } = params;

        const query = this.calibrationRepo.createQueryBuilder('calibration')
            .where('calibration.tenantId = :tenantId', { tenantId })
            .andWhere('calibration.isActive = :isActive', { isActive: true });

        if (vehicleId) {
            query.andWhere('calibration.vehicleId = :vehicleId', { vehicleId });
        }

        if (customerId) {
            query.andWhere('calibration.customerId = :customerId', { customerId });
        }

        if (calibrationType) {
            query.andWhere('calibration.calibrationType = :calibrationType', { calibrationType });
        }

        const [data, total] = await query
            .orderBy('calibration.createdAt', 'DESC')
            .skip((page - 1) * limit)
            .take(limit)
            .getManyAndCount();

        return {
            data,
            total,
            page,
            limit,
            totalPages: Math.ceil(total / limit),
        };
    }

    async findOne(tenantId: string, id: string): Promise<ECUCalibration> {
        const calibration = await this.calibrationRepo.findOne({
            where: { id, tenantId },
        });

        if (!calibration) {
            throw new NotFoundException(`ECU Calibration with ID ${id} not found`);
        }

        return calibration;
    }

    async getDownloadUrl(tenantId: string, id: string, expiresIn = 3600): Promise<string> {
        const calibration = await this.findOne(tenantId, id);

        const command = new GetObjectCommand({
            Bucket: calibration.s3Bucket,
            Key: calibration.s3Key,
        });

        return await getSignedUrl(this.s3Client, command, { expiresIn });
    }

    async update(tenantId: string, id: string, dto: UpdateECUCalibrationDto): Promise<ECUCalibration> {
        const calibration = await this.findOne(tenantId, id);

        Object.assign(calibration, dto);

        return await this.calibrationRepo.save(calibration);
    }

    async remove(tenantId: string, id: string): Promise<void> {
        const calibration = await this.findOne(tenantId, id);

        // Soft delete by marking as inactive
        calibration.isActive = false;
        await this.calibrationRepo.save(calibration);

        // Optionally delete from S3 (commented out for safety)
        // await this.s3Client.send(
        //     new DeleteObjectCommand({
        //         Bucket: calibration.s3Bucket,
        //         Key: calibration.s3Key,
        //     }),
        // );
    }

    async getVersionHistory(tenantId: string, vehicleId: string): Promise<ECUCalibration[]> {
        return await this.calibrationRepo.find({
            where: { tenantId, vehicleId, isActive: true },
            order: { version: 'DESC', createdAt: 'DESC' },
        });
    }
}

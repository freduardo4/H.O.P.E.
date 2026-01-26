"use strict";
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
var __metadata = (this && this.__metadata) || function (k, v) {
    if (typeof Reflect === "object" && typeof Reflect.metadata === "function") return Reflect.metadata(k, v);
};
var __param = (this && this.__param) || function (paramIndex, decorator) {
    return function (target, key) { decorator(target, key, paramIndex); }
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.ECUCalibrationsService = void 0;
const common_1 = require("@nestjs/common");
const typeorm_1 = require("@nestjs/typeorm");
const typeorm_2 = require("typeorm");
const client_s3_1 = require("@aws-sdk/client-s3");
const s3_request_presigner_1 = require("@aws-sdk/s3-request-presigner");
const ecu_calibration_entity_1 = require("./entities/ecu-calibration.entity");
let ECUCalibrationsService = class ECUCalibrationsService {
    constructor(calibrationRepo) {
        this.calibrationRepo = calibrationRepo;
        this.bucketName = process.env.AWS_S3_BUCKET || 'hope-ecu-calibrations';
        this.s3Client = new client_s3_1.S3Client({
            region: process.env.AWS_S3_REGION || 'us-east-1',
        });
    }
    async uploadFile(params) {
        const { tenantId, dto, fileBuffer, uploadedBy } = params;
        const timestamp = Date.now();
        const s3Key = `${tenantId}/ecu-calibrations/${dto.vehicleId}/${timestamp}-${dto.fileName}`;
        try {
            await this.s3Client.send(new client_s3_1.PutObjectCommand({
                Bucket: this.bucketName,
                Key: s3Key,
                Body: fileBuffer,
                ContentType: 'application/octet-stream',
                Metadata: {
                    vehicleId: dto.vehicleId,
                    calibrationType: dto.calibrationType,
                    checksum: dto.checksum,
                },
            }));
            let version = 1;
            if (dto.previousVersionId) {
                const previousVersion = await this.calibrationRepo.findOne({
                    where: { id: dto.previousVersionId, tenantId },
                });
                if (previousVersion) {
                    version = previousVersion.version + 1;
                }
            }
            const calibration = this.calibrationRepo.create({
                ...dto,
                tenantId,
                s3Key,
                s3Bucket: this.bucketName,
                version,
                uploadedBy,
            });
            return await this.calibrationRepo.save(calibration);
        }
        catch (error) {
            throw new common_1.BadRequestException(`Failed to upload ECU file: ${error.message}`);
        }
    }
    async findAll(params) {
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
    async findOne(tenantId, id) {
        const calibration = await this.calibrationRepo.findOne({
            where: { id, tenantId },
        });
        if (!calibration) {
            throw new common_1.NotFoundException(`ECU Calibration with ID ${id} not found`);
        }
        return calibration;
    }
    async getDownloadUrl(tenantId, id, expiresIn = 3600) {
        const calibration = await this.findOne(tenantId, id);
        const command = new client_s3_1.GetObjectCommand({
            Bucket: calibration.s3Bucket,
            Key: calibration.s3Key,
        });
        return await (0, s3_request_presigner_1.getSignedUrl)(this.s3Client, command, { expiresIn });
    }
    async update(tenantId, id, dto) {
        const calibration = await this.findOne(tenantId, id);
        Object.assign(calibration, dto);
        return await this.calibrationRepo.save(calibration);
    }
    async remove(tenantId, id) {
        const calibration = await this.findOne(tenantId, id);
        calibration.isActive = false;
        await this.calibrationRepo.save(calibration);
    }
    async getVersionHistory(tenantId, vehicleId) {
        return await this.calibrationRepo.find({
            where: { tenantId, vehicleId, isActive: true },
            order: { version: 'DESC', createdAt: 'DESC' },
        });
    }
};
exports.ECUCalibrationsService = ECUCalibrationsService;
exports.ECUCalibrationsService = ECUCalibrationsService = __decorate([
    (0, common_1.Injectable)(),
    __param(0, (0, typeorm_1.InjectRepository)(ecu_calibration_entity_1.ECUCalibration)),
    __metadata("design:paramtypes", [typeorm_2.Repository])
], ECUCalibrationsService);
//# sourceMappingURL=ecu-calibrations.service.js.map
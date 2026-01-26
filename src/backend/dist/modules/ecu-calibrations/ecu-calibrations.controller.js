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
exports.ECUCalibrationsController = void 0;
const common_1 = require("@nestjs/common");
const platform_express_1 = require("@nestjs/platform-express");
const ecu_calibrations_service_1 = require("./ecu-calibrations.service");
const dto_1 = require("./dto");
const ecu_calibration_entity_1 = require("./entities/ecu-calibration.entity");
const auth_1 = require("../auth");
const user_entity_1 = require("../auth/entities/user.entity");
let ECUCalibrationsController = class ECUCalibrationsController {
    constructor(calibrationsService) {
        this.calibrationsService = calibrationsService;
    }
    async uploadFile(user, file, dto) {
        if (!file) {
            throw new common_1.BadRequestException('No file uploaded');
        }
        return this.calibrationsService.uploadFile({
            tenantId: user.tenantId,
            dto,
            fileBuffer: file.buffer,
            uploadedBy: user.id,
        });
    }
    async findAll(user, vehicleId, customerId, calibrationType, page = 1, limit = 20) {
        return this.calibrationsService.findAll({
            tenantId: user.tenantId,
            vehicleId,
            customerId,
            calibrationType,
            page: Number(page),
            limit: Number(limit),
        });
    }
    async findOne(user, id) {
        return this.calibrationsService.findOne(user.tenantId, id);
    }
    async getDownloadUrl(user, id) {
        const url = await this.calibrationsService.getDownloadUrl(user.tenantId, id);
        return { url, expiresIn: 3600 };
    }
    async getVersionHistory(user, vehicleId) {
        return this.calibrationsService.getVersionHistory(user.tenantId, vehicleId);
    }
    async update(user, id, dto) {
        return this.calibrationsService.update(user.tenantId, id, dto);
    }
    async remove(user, id) {
        await this.calibrationsService.remove(user.tenantId, id);
        return { message: 'ECU Calibration deleted successfully' };
    }
};
exports.ECUCalibrationsController = ECUCalibrationsController;
__decorate([
    (0, common_1.Post)('upload'),
    (0, auth_1.Roles)(user_entity_1.UserRole.ADMIN, user_entity_1.UserRole.SHOP_OWNER, user_entity_1.UserRole.TECHNICIAN),
    (0, common_1.UseInterceptors)((0, platform_express_1.FileInterceptor)('file')),
    __param(0, (0, auth_1.CurrentUser)()),
    __param(1, (0, common_1.UploadedFile)()),
    __param(2, (0, common_1.Body)()),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [user_entity_1.User, Object, dto_1.CreateECUCalibrationDto]),
    __metadata("design:returntype", Promise)
], ECUCalibrationsController.prototype, "uploadFile", null);
__decorate([
    (0, common_1.Get)(),
    __param(0, (0, auth_1.CurrentUser)()),
    __param(1, (0, common_1.Query)('vehicleId')),
    __param(2, (0, common_1.Query)('customerId')),
    __param(3, (0, common_1.Query)('calibrationType')),
    __param(4, (0, common_1.Query)('page')),
    __param(5, (0, common_1.Query)('limit')),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [user_entity_1.User, String, String, String, Object, Object]),
    __metadata("design:returntype", Promise)
], ECUCalibrationsController.prototype, "findAll", null);
__decorate([
    (0, common_1.Get)(':id'),
    __param(0, (0, auth_1.CurrentUser)()),
    __param(1, (0, common_1.Param)('id', common_1.ParseUUIDPipe)),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [user_entity_1.User, String]),
    __metadata("design:returntype", Promise)
], ECUCalibrationsController.prototype, "findOne", null);
__decorate([
    (0, common_1.Get)(':id/download-url'),
    __param(0, (0, auth_1.CurrentUser)()),
    __param(1, (0, common_1.Param)('id', common_1.ParseUUIDPipe)),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [user_entity_1.User, String]),
    __metadata("design:returntype", Promise)
], ECUCalibrationsController.prototype, "getDownloadUrl", null);
__decorate([
    (0, common_1.Get)('vehicle/:vehicleId/history'),
    __param(0, (0, auth_1.CurrentUser)()),
    __param(1, (0, common_1.Param)('vehicleId', common_1.ParseUUIDPipe)),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [user_entity_1.User, String]),
    __metadata("design:returntype", Promise)
], ECUCalibrationsController.prototype, "getVersionHistory", null);
__decorate([
    (0, common_1.Patch)(':id'),
    (0, auth_1.Roles)(user_entity_1.UserRole.ADMIN, user_entity_1.UserRole.SHOP_OWNER, user_entity_1.UserRole.TECHNICIAN),
    __param(0, (0, auth_1.CurrentUser)()),
    __param(1, (0, common_1.Param)('id', common_1.ParseUUIDPipe)),
    __param(2, (0, common_1.Body)()),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [user_entity_1.User, String, dto_1.UpdateECUCalibrationDto]),
    __metadata("design:returntype", Promise)
], ECUCalibrationsController.prototype, "update", null);
__decorate([
    (0, common_1.Delete)(':id'),
    (0, auth_1.Roles)(user_entity_1.UserRole.ADMIN, user_entity_1.UserRole.SHOP_OWNER),
    __param(0, (0, auth_1.CurrentUser)()),
    __param(1, (0, common_1.Param)('id', common_1.ParseUUIDPipe)),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [user_entity_1.User, String]),
    __metadata("design:returntype", Promise)
], ECUCalibrationsController.prototype, "remove", null);
exports.ECUCalibrationsController = ECUCalibrationsController = __decorate([
    (0, common_1.Controller)('ecu-calibrations'),
    (0, common_1.UseGuards)(auth_1.JwtAuthGuard, auth_1.RolesGuard),
    __metadata("design:paramtypes", [ecu_calibrations_service_1.ECUCalibrationsService])
], ECUCalibrationsController);
//# sourceMappingURL=ecu-calibrations.controller.js.map
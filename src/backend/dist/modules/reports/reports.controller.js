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
exports.ReportsController = void 0;
const common_1 = require("@nestjs/common");
const reports_service_1 = require("./reports.service");
const dto_1 = require("./dto");
const report_entity_1 = require("./entities/report.entity");
const auth_1 = require("../auth");
const user_entity_1 = require("../auth/entities/user.entity");
let ReportsController = class ReportsController {
    constructor(reportsService) {
        this.reportsService = reportsService;
    }
    async create(user, dto) {
        return this.reportsService.create(user.tenantId, dto, user.id);
    }
    async findAll(user, vehicleId, customerId, reportType, status, page = 1, limit = 20) {
        return this.reportsService.findAll({
            tenantId: user.tenantId,
            vehicleId,
            customerId,
            reportType,
            status,
            page: Number(page),
            limit: Number(limit),
        });
    }
    async findOne(user, id) {
        return this.reportsService.findOne(user.tenantId, id);
    }
    async getDownloadUrl(user, id) {
        const url = await this.reportsService.getDownloadUrl(user.tenantId, id);
        return { url, expiresIn: 3600 };
    }
    async regenerate(user, id) {
        return this.reportsService.regenerate(user.tenantId, id, user.id);
    }
    async remove(user, id) {
        await this.reportsService.remove(user.tenantId, id);
        return { message: 'Report deleted successfully' };
    }
};
exports.ReportsController = ReportsController;
__decorate([
    (0, common_1.Post)(),
    (0, auth_1.Roles)(user_entity_1.UserRole.ADMIN, user_entity_1.UserRole.SHOP_OWNER, user_entity_1.UserRole.TECHNICIAN),
    __param(0, (0, auth_1.CurrentUser)()),
    __param(1, (0, common_1.Body)()),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [user_entity_1.User,
        dto_1.CreateReportDto]),
    __metadata("design:returntype", Promise)
], ReportsController.prototype, "create", null);
__decorate([
    (0, common_1.Get)(),
    __param(0, (0, auth_1.CurrentUser)()),
    __param(1, (0, common_1.Query)('vehicleId')),
    __param(2, (0, common_1.Query)('customerId')),
    __param(3, (0, common_1.Query)('reportType')),
    __param(4, (0, common_1.Query)('status')),
    __param(5, (0, common_1.Query)('page')),
    __param(6, (0, common_1.Query)('limit')),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [user_entity_1.User, String, String, String, String, Object, Object]),
    __metadata("design:returntype", Promise)
], ReportsController.prototype, "findAll", null);
__decorate([
    (0, common_1.Get)(':id'),
    __param(0, (0, auth_1.CurrentUser)()),
    __param(1, (0, common_1.Param)('id', common_1.ParseUUIDPipe)),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [user_entity_1.User, String]),
    __metadata("design:returntype", Promise)
], ReportsController.prototype, "findOne", null);
__decorate([
    (0, common_1.Get)(':id/download-url'),
    __param(0, (0, auth_1.CurrentUser)()),
    __param(1, (0, common_1.Param)('id', common_1.ParseUUIDPipe)),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [user_entity_1.User, String]),
    __metadata("design:returntype", Promise)
], ReportsController.prototype, "getDownloadUrl", null);
__decorate([
    (0, common_1.Post)(':id/regenerate'),
    (0, auth_1.Roles)(user_entity_1.UserRole.ADMIN, user_entity_1.UserRole.SHOP_OWNER, user_entity_1.UserRole.TECHNICIAN),
    __param(0, (0, auth_1.CurrentUser)()),
    __param(1, (0, common_1.Param)('id', common_1.ParseUUIDPipe)),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [user_entity_1.User, String]),
    __metadata("design:returntype", Promise)
], ReportsController.prototype, "regenerate", null);
__decorate([
    (0, common_1.Delete)(':id'),
    (0, auth_1.Roles)(user_entity_1.UserRole.ADMIN, user_entity_1.UserRole.SHOP_OWNER),
    __param(0, (0, auth_1.CurrentUser)()),
    __param(1, (0, common_1.Param)('id', common_1.ParseUUIDPipe)),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [user_entity_1.User, String]),
    __metadata("design:returntype", Promise)
], ReportsController.prototype, "remove", null);
exports.ReportsController = ReportsController = __decorate([
    (0, common_1.Controller)('reports'),
    (0, common_1.UseGuards)(auth_1.JwtAuthGuard, auth_1.RolesGuard),
    __metadata("design:paramtypes", [reports_service_1.ReportsService])
], ReportsController);
//# sourceMappingURL=reports.controller.js.map
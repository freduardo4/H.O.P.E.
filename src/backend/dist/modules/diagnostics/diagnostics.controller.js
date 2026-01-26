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
exports.DiagnosticsController = void 0;
const common_1 = require("@nestjs/common");
const diagnostics_service_1 = require("./diagnostics.service");
const dto_1 = require("./dto");
const diagnostic_session_entity_1 = require("./entities/diagnostic-session.entity");
const auth_1 = require("../auth");
const user_entity_1 = require("../auth/entities/user.entity");
let DiagnosticsController = class DiagnosticsController {
    constructor(diagnosticsService) {
        this.diagnosticsService = diagnosticsService;
    }
    async createSession(user, dto) {
        return this.diagnosticsService.createSession(user.tenantId, user.id, dto);
    }
    async findAllSessions(user, vehicleId, technicianId, type, status, startDate, endDate, page = 1, limit = 20) {
        return this.diagnosticsService.findAllSessions({
            tenantId: user.tenantId,
            vehicleId,
            technicianId,
            type,
            status,
            startDate: startDate ? new Date(startDate) : undefined,
            endDate: endDate ? new Date(endDate) : undefined,
            page: Number(page),
            limit: Number(limit),
        });
    }
    async findSession(user, id) {
        return this.diagnosticsService.findSession(user.tenantId, id);
    }
    async endSession(user, id, dto) {
        return this.diagnosticsService.endSession(user.tenantId, id, dto);
    }
    async cancelSession(user, id) {
        return this.diagnosticsService.cancelSession(user.tenantId, id);
    }
    async logReading(dto) {
        return this.diagnosticsService.logReading(dto);
    }
    async logReadingsBatch(dto) {
        return this.diagnosticsService.logReadingsBatch(dto.readings);
    }
    async getSessionReadings(sessionId, pid, startTime, endTime, limit) {
        return this.diagnosticsService.getSessionReadings(sessionId, {
            pid,
            startTime: startTime ? new Date(startTime) : undefined,
            endTime: endTime ? new Date(endTime) : undefined,
            limit: limit ? Number(limit) : undefined,
        });
    }
    async getLatestReadings(sessionId) {
        const latestMap = await this.diagnosticsService.getLatestReadings(sessionId);
        return Object.fromEntries(latestMap);
    }
    async getAnalytics(user, startDate, endDate) {
        return this.diagnosticsService.getSessionAnalytics(user.tenantId, new Date(startDate), new Date(endDate));
    }
};
exports.DiagnosticsController = DiagnosticsController;
__decorate([
    (0, common_1.Post)('sessions'),
    (0, auth_1.Roles)(user_entity_1.UserRole.ADMIN, user_entity_1.UserRole.SHOP_OWNER, user_entity_1.UserRole.TECHNICIAN),
    __param(0, (0, auth_1.CurrentUser)()),
    __param(1, (0, common_1.Body)()),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [user_entity_1.User,
        dto_1.CreateSessionDto]),
    __metadata("design:returntype", Promise)
], DiagnosticsController.prototype, "createSession", null);
__decorate([
    (0, common_1.Get)('sessions'),
    __param(0, (0, auth_1.CurrentUser)()),
    __param(1, (0, common_1.Query)('vehicleId')),
    __param(2, (0, common_1.Query)('technicianId')),
    __param(3, (0, common_1.Query)('type')),
    __param(4, (0, common_1.Query)('status')),
    __param(5, (0, common_1.Query)('startDate')),
    __param(6, (0, common_1.Query)('endDate')),
    __param(7, (0, common_1.Query)('page')),
    __param(8, (0, common_1.Query)('limit')),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [user_entity_1.User, String, String, String, String, String, String, Object, Object]),
    __metadata("design:returntype", Promise)
], DiagnosticsController.prototype, "findAllSessions", null);
__decorate([
    (0, common_1.Get)('sessions/:id'),
    __param(0, (0, auth_1.CurrentUser)()),
    __param(1, (0, common_1.Param)('id', common_1.ParseUUIDPipe)),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [user_entity_1.User, String]),
    __metadata("design:returntype", Promise)
], DiagnosticsController.prototype, "findSession", null);
__decorate([
    (0, common_1.Post)('sessions/:id/end'),
    (0, auth_1.Roles)(user_entity_1.UserRole.ADMIN, user_entity_1.UserRole.SHOP_OWNER, user_entity_1.UserRole.TECHNICIAN),
    __param(0, (0, auth_1.CurrentUser)()),
    __param(1, (0, common_1.Param)('id', common_1.ParseUUIDPipe)),
    __param(2, (0, common_1.Body)()),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [user_entity_1.User, String, dto_1.EndSessionDto]),
    __metadata("design:returntype", Promise)
], DiagnosticsController.prototype, "endSession", null);
__decorate([
    (0, common_1.Post)('sessions/:id/cancel'),
    (0, auth_1.Roles)(user_entity_1.UserRole.ADMIN, user_entity_1.UserRole.SHOP_OWNER, user_entity_1.UserRole.TECHNICIAN),
    __param(0, (0, auth_1.CurrentUser)()),
    __param(1, (0, common_1.Param)('id', common_1.ParseUUIDPipe)),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [user_entity_1.User, String]),
    __metadata("design:returntype", Promise)
], DiagnosticsController.prototype, "cancelSession", null);
__decorate([
    (0, common_1.Post)('readings'),
    (0, auth_1.Roles)(user_entity_1.UserRole.ADMIN, user_entity_1.UserRole.SHOP_OWNER, user_entity_1.UserRole.TECHNICIAN),
    __param(0, (0, common_1.Body)()),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [dto_1.LogReadingDto]),
    __metadata("design:returntype", Promise)
], DiagnosticsController.prototype, "logReading", null);
__decorate([
    (0, common_1.Post)('readings/batch'),
    (0, auth_1.Roles)(user_entity_1.UserRole.ADMIN, user_entity_1.UserRole.SHOP_OWNER, user_entity_1.UserRole.TECHNICIAN),
    __param(0, (0, common_1.Body)()),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [dto_1.LogReadingsBatchDto]),
    __metadata("design:returntype", Promise)
], DiagnosticsController.prototype, "logReadingsBatch", null);
__decorate([
    (0, common_1.Get)('sessions/:id/readings'),
    __param(0, (0, common_1.Param)('id', common_1.ParseUUIDPipe)),
    __param(1, (0, common_1.Query)('pid')),
    __param(2, (0, common_1.Query)('startTime')),
    __param(3, (0, common_1.Query)('endTime')),
    __param(4, (0, common_1.Query)('limit')),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [String, String, String, String, Number]),
    __metadata("design:returntype", Promise)
], DiagnosticsController.prototype, "getSessionReadings", null);
__decorate([
    (0, common_1.Get)('sessions/:id/readings/latest'),
    __param(0, (0, common_1.Param)('id', common_1.ParseUUIDPipe)),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [String]),
    __metadata("design:returntype", Promise)
], DiagnosticsController.prototype, "getLatestReadings", null);
__decorate([
    (0, common_1.Get)('analytics'),
    (0, auth_1.Roles)(user_entity_1.UserRole.ADMIN, user_entity_1.UserRole.SHOP_OWNER),
    __param(0, (0, auth_1.CurrentUser)()),
    __param(1, (0, common_1.Query)('startDate')),
    __param(2, (0, common_1.Query)('endDate')),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [user_entity_1.User, String, String]),
    __metadata("design:returntype", Promise)
], DiagnosticsController.prototype, "getAnalytics", null);
exports.DiagnosticsController = DiagnosticsController = __decorate([
    (0, common_1.Controller)('diagnostics'),
    (0, common_1.UseGuards)(auth_1.JwtAuthGuard, auth_1.RolesGuard),
    __metadata("design:paramtypes", [diagnostics_service_1.DiagnosticsService])
], DiagnosticsController);
//# sourceMappingURL=diagnostics.controller.js.map
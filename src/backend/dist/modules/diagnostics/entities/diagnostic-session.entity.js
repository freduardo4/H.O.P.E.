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
Object.defineProperty(exports, "__esModule", { value: true });
exports.DiagnosticSession = exports.SessionType = exports.SessionStatus = void 0;
const typeorm_1 = require("typeorm");
var SessionStatus;
(function (SessionStatus) {
    SessionStatus["ACTIVE"] = "active";
    SessionStatus["COMPLETED"] = "completed";
    SessionStatus["CANCELLED"] = "cancelled";
})(SessionStatus || (exports.SessionStatus = SessionStatus = {}));
var SessionType;
(function (SessionType) {
    SessionType["DIAGNOSTIC"] = "diagnostic";
    SessionType["PERFORMANCE"] = "performance";
    SessionType["TUNE"] = "tune";
    SessionType["MAINTENANCE"] = "maintenance";
})(SessionType || (exports.SessionType = SessionType = {}));
let DiagnosticSession = class DiagnosticSession {
    get duration() {
        if (!this.endTime)
            return null;
        return Math.floor((this.endTime.getTime() - this.startTime.getTime()) / 1000);
    }
};
exports.DiagnosticSession = DiagnosticSession;
__decorate([
    (0, typeorm_1.PrimaryGeneratedColumn)('uuid'),
    __metadata("design:type", String)
], DiagnosticSession.prototype, "id", void 0);
__decorate([
    (0, typeorm_1.Column)(),
    __metadata("design:type", String)
], DiagnosticSession.prototype, "tenantId", void 0);
__decorate([
    (0, typeorm_1.Column)(),
    __metadata("design:type", String)
], DiagnosticSession.prototype, "vehicleId", void 0);
__decorate([
    (0, typeorm_1.Column)({ nullable: true }),
    __metadata("design:type", String)
], DiagnosticSession.prototype, "technicianId", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'enum', enum: SessionType, default: SessionType.DIAGNOSTIC }),
    __metadata("design:type", String)
], DiagnosticSession.prototype, "type", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'enum', enum: SessionStatus, default: SessionStatus.ACTIVE }),
    __metadata("design:type", String)
], DiagnosticSession.prototype, "status", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'timestamp' }),
    __metadata("design:type", Date)
], DiagnosticSession.prototype, "startTime", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'timestamp', nullable: true }),
    __metadata("design:type", Date)
], DiagnosticSession.prototype, "endTime", void 0);
__decorate([
    (0, typeorm_1.Column)({ nullable: true }),
    __metadata("design:type", Number)
], DiagnosticSession.prototype, "mileageAtSession", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'text', nullable: true }),
    __metadata("design:type", String)
], DiagnosticSession.prototype, "notes", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'jsonb', nullable: true }),
    __metadata("design:type", Object)
], DiagnosticSession.prototype, "ecuSnapshot", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'jsonb', nullable: true }),
    __metadata("design:type", Array)
], DiagnosticSession.prototype, "dtcCodes", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'jsonb', nullable: true }),
    __metadata("design:type", Object)
], DiagnosticSession.prototype, "performanceMetrics", void 0);
__decorate([
    (0, typeorm_1.CreateDateColumn)(),
    __metadata("design:type", Date)
], DiagnosticSession.prototype, "createdAt", void 0);
__decorate([
    (0, typeorm_1.UpdateDateColumn)(),
    __metadata("design:type", Date)
], DiagnosticSession.prototype, "updatedAt", void 0);
exports.DiagnosticSession = DiagnosticSession = __decorate([
    (0, typeorm_1.Entity)('diagnostic_sessions')
], DiagnosticSession);
//# sourceMappingURL=diagnostic-session.entity.js.map
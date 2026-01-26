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
exports.DiagnosticsService = void 0;
const common_1 = require("@nestjs/common");
const typeorm_1 = require("@nestjs/typeorm");
const typeorm_2 = require("typeorm");
const diagnostic_session_entity_1 = require("./entities/diagnostic-session.entity");
const obd2_reading_entity_1 = require("./entities/obd2-reading.entity");
let DiagnosticsService = class DiagnosticsService {
    constructor(sessionRepository, readingRepository) {
        this.sessionRepository = sessionRepository;
        this.readingRepository = readingRepository;
    }
    async createSession(tenantId, technicianId, dto) {
        const session = this.sessionRepository.create({
            tenantId,
            technicianId,
            vehicleId: dto.vehicleId,
            type: dto.type || diagnostic_session_entity_1.SessionType.DIAGNOSTIC,
            status: diagnostic_session_entity_1.SessionStatus.ACTIVE,
            startTime: new Date(),
            mileageAtSession: dto.mileageAtSession,
            notes: dto.notes,
        });
        return this.sessionRepository.save(session);
    }
    async findAllSessions(options) {
        const { tenantId, vehicleId, technicianId, type, status, startDate, endDate, page = 1, limit = 20, } = options;
        const queryBuilder = this.sessionRepository
            .createQueryBuilder('session')
            .where('session.tenantId = :tenantId', { tenantId });
        if (vehicleId) {
            queryBuilder.andWhere('session.vehicleId = :vehicleId', { vehicleId });
        }
        if (technicianId) {
            queryBuilder.andWhere('session.technicianId = :technicianId', { technicianId });
        }
        if (type) {
            queryBuilder.andWhere('session.type = :type', { type });
        }
        if (status) {
            queryBuilder.andWhere('session.status = :status', { status });
        }
        if (startDate && endDate) {
            queryBuilder.andWhere('session.startTime BETWEEN :startDate AND :endDate', {
                startDate,
                endDate,
            });
        }
        const total = await queryBuilder.getCount();
        const items = await queryBuilder
            .orderBy('session.startTime', 'DESC')
            .skip((page - 1) * limit)
            .take(limit)
            .getMany();
        return {
            items,
            total,
            page,
            limit,
            totalPages: Math.ceil(total / limit),
        };
    }
    async findSession(tenantId, id) {
        const session = await this.sessionRepository.findOne({
            where: { id, tenantId },
        });
        if (!session) {
            throw new common_1.NotFoundException(`Session with ID ${id} not found`);
        }
        return session;
    }
    async endSession(tenantId, id, dto) {
        const session = await this.findSession(tenantId, id);
        if (session.status !== diagnostic_session_entity_1.SessionStatus.ACTIVE) {
            throw new common_1.BadRequestException('Session is not active');
        }
        session.status = diagnostic_session_entity_1.SessionStatus.COMPLETED;
        session.endTime = new Date();
        if (dto.notes) {
            session.notes = dto.notes;
        }
        if (dto.dtcCodes) {
            session.dtcCodes = dto.dtcCodes;
        }
        if (dto.performanceMetrics) {
            session.performanceMetrics = dto.performanceMetrics;
        }
        if (dto.ecuSnapshot) {
            session.ecuSnapshot = dto.ecuSnapshot;
        }
        return this.sessionRepository.save(session);
    }
    async cancelSession(tenantId, id) {
        const session = await this.findSession(tenantId, id);
        if (session.status !== diagnostic_session_entity_1.SessionStatus.ACTIVE) {
            throw new common_1.BadRequestException('Session is not active');
        }
        session.status = diagnostic_session_entity_1.SessionStatus.CANCELLED;
        session.endTime = new Date();
        return this.sessionRepository.save(session);
    }
    async logReading(dto) {
        const reading = this.readingRepository.create({
            sessionId: dto.sessionId,
            timestamp: dto.timestamp ? new Date(dto.timestamp) : new Date(),
            pid: dto.pid,
            name: dto.name,
            value: dto.value,
            unit: dto.unit,
            rawResponse: dto.rawResponse,
        });
        return this.readingRepository.save(reading);
    }
    async logReadingsBatch(readings) {
        const entities = readings.map(dto => this.readingRepository.create({
            sessionId: dto.sessionId,
            timestamp: dto.timestamp ? new Date(dto.timestamp) : new Date(),
            pid: dto.pid,
            name: dto.name,
            value: dto.value,
            unit: dto.unit,
            rawResponse: dto.rawResponse,
        }));
        return this.readingRepository.save(entities);
    }
    async getSessionReadings(sessionId, options) {
        const queryBuilder = this.readingRepository
            .createQueryBuilder('reading')
            .where('reading.sessionId = :sessionId', { sessionId });
        if (options?.pid) {
            queryBuilder.andWhere('reading.pid = :pid', { pid: options.pid });
        }
        if (options?.startTime && options?.endTime) {
            queryBuilder.andWhere('reading.timestamp BETWEEN :startTime AND :endTime', {
                startTime: options.startTime,
                endTime: options.endTime,
            });
        }
        queryBuilder.orderBy('reading.timestamp', 'ASC');
        if (options?.limit) {
            queryBuilder.take(options.limit);
        }
        return queryBuilder.getMany();
    }
    async getLatestReadings(sessionId) {
        const readings = await this.readingRepository
            .createQueryBuilder('reading')
            .where('reading.sessionId = :sessionId', { sessionId })
            .orderBy('reading.timestamp', 'DESC')
            .getMany();
        const latestByPid = new Map();
        for (const reading of readings) {
            if (!latestByPid.has(reading.pid)) {
                latestByPid.set(reading.pid, reading);
            }
        }
        return latestByPid;
    }
    async getSessionAnalytics(tenantId, startDate, endDate) {
        const sessions = await this.sessionRepository.find({
            where: {
                tenantId,
                startTime: (0, typeorm_2.Between)(startDate, endDate),
            },
        });
        const totalSessions = sessions.length;
        let totalDuration = 0;
        const byType = {};
        const byStatus = {};
        const dtcCounts = {};
        for (const session of sessions) {
            if (session.endTime) {
                totalDuration += Math.floor((session.endTime.getTime() - session.startTime.getTime()) / 1000);
            }
            byType[session.type] = (byType[session.type] || 0) + 1;
            byStatus[session.status] = (byStatus[session.status] || 0) + 1;
            if (session.dtcCodes) {
                for (const code of session.dtcCodes) {
                    dtcCounts[code] = (dtcCounts[code] || 0) + 1;
                }
            }
        }
        return {
            totalSessions,
            totalDuration,
            avgDuration: totalSessions > 0 ? Math.floor(totalDuration / totalSessions) : 0,
            byType: Object.entries(byType).map(([type, count]) => ({ type, count })),
            byStatus: Object.entries(byStatus).map(([status, count]) => ({ status, count })),
            commonDtcCodes: Object.entries(dtcCounts)
                .map(([code, count]) => ({ code, count }))
                .sort((a, b) => b.count - a.count)
                .slice(0, 10),
        };
    }
};
exports.DiagnosticsService = DiagnosticsService;
exports.DiagnosticsService = DiagnosticsService = __decorate([
    (0, common_1.Injectable)(),
    __param(0, (0, typeorm_1.InjectRepository)(diagnostic_session_entity_1.DiagnosticSession)),
    __param(1, (0, typeorm_1.InjectRepository)(obd2_reading_entity_1.OBD2Reading)),
    __metadata("design:paramtypes", [typeorm_2.Repository,
        typeorm_2.Repository])
], DiagnosticsService);
//# sourceMappingURL=diagnostics.service.js.map
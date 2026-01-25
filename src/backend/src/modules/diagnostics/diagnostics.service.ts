import { Injectable, NotFoundException, BadRequestException } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository, Between } from 'typeorm';
import { DiagnosticSession, SessionStatus, SessionType } from './entities/diagnostic-session.entity';
import { OBD2Reading } from './entities/obd2-reading.entity';
import { CreateSessionDto, LogReadingDto, EndSessionDto } from './dto';

export interface SessionSearchOptions {
    tenantId: string;
    vehicleId?: string;
    technicianId?: string;
    type?: SessionType;
    status?: SessionStatus;
    startDate?: Date;
    endDate?: Date;
    page?: number;
    limit?: number;
}

export interface PaginatedSessions {
    items: DiagnosticSession[];
    total: number;
    page: number;
    limit: number;
    totalPages: number;
}

export interface SessionAnalytics {
    totalSessions: number;
    totalDuration: number;
    avgDuration: number;
    byType: { type: string; count: number }[];
    byStatus: { status: string; count: number }[];
    commonDtcCodes: { code: string; count: number }[];
}

@Injectable()
export class DiagnosticsService {
    constructor(
        @InjectRepository(DiagnosticSession)
        private readonly sessionRepository: Repository<DiagnosticSession>,
        @InjectRepository(OBD2Reading)
        private readonly readingRepository: Repository<OBD2Reading>,
    ) {}

    async createSession(
        tenantId: string,
        technicianId: string,
        dto: CreateSessionDto,
    ): Promise<DiagnosticSession> {
        const session = this.sessionRepository.create({
            tenantId,
            technicianId,
            vehicleId: dto.vehicleId,
            type: dto.type || SessionType.DIAGNOSTIC,
            status: SessionStatus.ACTIVE,
            startTime: new Date(),
            mileageAtSession: dto.mileageAtSession,
            notes: dto.notes,
        });

        return this.sessionRepository.save(session);
    }

    async findAllSessions(options: SessionSearchOptions): Promise<PaginatedSessions> {
        const {
            tenantId,
            vehicleId,
            technicianId,
            type,
            status,
            startDate,
            endDate,
            page = 1,
            limit = 20,
        } = options;

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

    async findSession(tenantId: string, id: string): Promise<DiagnosticSession> {
        const session = await this.sessionRepository.findOne({
            where: { id, tenantId },
        });

        if (!session) {
            throw new NotFoundException(`Session with ID ${id} not found`);
        }

        return session;
    }

    async endSession(
        tenantId: string,
        id: string,
        dto: EndSessionDto,
    ): Promise<DiagnosticSession> {
        const session = await this.findSession(tenantId, id);

        if (session.status !== SessionStatus.ACTIVE) {
            throw new BadRequestException('Session is not active');
        }

        session.status = SessionStatus.COMPLETED;
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

    async cancelSession(tenantId: string, id: string): Promise<DiagnosticSession> {
        const session = await this.findSession(tenantId, id);

        if (session.status !== SessionStatus.ACTIVE) {
            throw new BadRequestException('Session is not active');
        }

        session.status = SessionStatus.CANCELLED;
        session.endTime = new Date();

        return this.sessionRepository.save(session);
    }

    async logReading(dto: LogReadingDto): Promise<OBD2Reading> {
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

    async logReadingsBatch(readings: LogReadingDto[]): Promise<OBD2Reading[]> {
        const entities = readings.map(dto =>
            this.readingRepository.create({
                sessionId: dto.sessionId,
                timestamp: dto.timestamp ? new Date(dto.timestamp) : new Date(),
                pid: dto.pid,
                name: dto.name,
                value: dto.value,
                unit: dto.unit,
                rawResponse: dto.rawResponse,
            }),
        );

        return this.readingRepository.save(entities);
    }

    async getSessionReadings(
        sessionId: string,
        options?: {
            pid?: string;
            startTime?: Date;
            endTime?: Date;
            limit?: number;
        },
    ): Promise<OBD2Reading[]> {
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

    async getLatestReadings(sessionId: string): Promise<Map<string, OBD2Reading>> {
        const readings = await this.readingRepository
            .createQueryBuilder('reading')
            .where('reading.sessionId = :sessionId', { sessionId })
            .orderBy('reading.timestamp', 'DESC')
            .getMany();

        const latestByPid = new Map<string, OBD2Reading>();
        for (const reading of readings) {
            if (!latestByPid.has(reading.pid)) {
                latestByPid.set(reading.pid, reading);
            }
        }

        return latestByPid;
    }

    async getSessionAnalytics(
        tenantId: string,
        startDate: Date,
        endDate: Date,
    ): Promise<SessionAnalytics> {
        const sessions = await this.sessionRepository.find({
            where: {
                tenantId,
                startTime: Between(startDate, endDate),
            },
        });

        const totalSessions = sessions.length;
        let totalDuration = 0;

        const byType: { [key: string]: number } = {};
        const byStatus: { [key: string]: number } = {};
        const dtcCounts: { [key: string]: number } = {};

        for (const session of sessions) {
            // Calculate duration
            if (session.endTime) {
                totalDuration += Math.floor(
                    (session.endTime.getTime() - session.startTime.getTime()) / 1000,
                );
            }

            // Count by type
            byType[session.type] = (byType[session.type] || 0) + 1;

            // Count by status
            byStatus[session.status] = (byStatus[session.status] || 0) + 1;

            // Count DTC codes
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
}

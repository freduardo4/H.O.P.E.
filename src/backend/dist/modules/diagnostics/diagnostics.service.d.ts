import { Repository } from 'typeorm';
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
    byType: {
        type: string;
        count: number;
    }[];
    byStatus: {
        status: string;
        count: number;
    }[];
    commonDtcCodes: {
        code: string;
        count: number;
    }[];
}
export declare class DiagnosticsService {
    private readonly sessionRepository;
    private readonly readingRepository;
    constructor(sessionRepository: Repository<DiagnosticSession>, readingRepository: Repository<OBD2Reading>);
    createSession(tenantId: string, technicianId: string, dto: CreateSessionDto): Promise<DiagnosticSession>;
    findAllSessions(options: SessionSearchOptions): Promise<PaginatedSessions>;
    findSession(tenantId: string, id: string): Promise<DiagnosticSession>;
    endSession(tenantId: string, id: string, dto: EndSessionDto): Promise<DiagnosticSession>;
    cancelSession(tenantId: string, id: string): Promise<DiagnosticSession>;
    logReading(dto: LogReadingDto): Promise<OBD2Reading>;
    logReadingsBatch(readings: LogReadingDto[]): Promise<OBD2Reading[]>;
    getSessionReadings(sessionId: string, options?: {
        pid?: string;
        startTime?: Date;
        endTime?: Date;
        limit?: number;
    }): Promise<OBD2Reading[]>;
    getLatestReadings(sessionId: string): Promise<Map<string, OBD2Reading>>;
    getSessionAnalytics(tenantId: string, startDate: Date, endDate: Date): Promise<SessionAnalytics>;
}

import { DiagnosticsService, PaginatedSessions, SessionAnalytics } from './diagnostics.service';
import { CreateSessionDto, LogReadingDto, LogReadingsBatchDto, EndSessionDto } from './dto';
import { DiagnosticSession, SessionStatus, SessionType } from './entities/diagnostic-session.entity';
import { OBD2Reading } from './entities/obd2-reading.entity';
import { User } from '../auth/entities/user.entity';
export declare class DiagnosticsController {
    private readonly diagnosticsService;
    constructor(diagnosticsService: DiagnosticsService);
    createSession(user: User, dto: CreateSessionDto): Promise<DiagnosticSession>;
    findAllSessions(user: User, vehicleId?: string, technicianId?: string, type?: SessionType, status?: SessionStatus, startDate?: string, endDate?: string, page?: number, limit?: number): Promise<PaginatedSessions>;
    findSession(user: User, id: string): Promise<DiagnosticSession>;
    endSession(user: User, id: string, dto: EndSessionDto): Promise<DiagnosticSession>;
    cancelSession(user: User, id: string): Promise<DiagnosticSession>;
    logReading(dto: LogReadingDto): Promise<OBD2Reading>;
    logReadingsBatch(dto: LogReadingsBatchDto): Promise<OBD2Reading[]>;
    getSessionReadings(sessionId: string, pid?: string, startTime?: string, endTime?: string, limit?: number): Promise<OBD2Reading[]>;
    getLatestReadings(sessionId: string): Promise<Record<string, OBD2Reading>>;
    getAnalytics(user: User, startDate: string, endDate: string): Promise<SessionAnalytics>;
}

"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const testing_1 = require("@nestjs/testing");
const typeorm_1 = require("@nestjs/typeorm");
const common_1 = require("@nestjs/common");
const diagnostics_service_1 = require("./diagnostics.service");
const diagnostic_session_entity_1 = require("./entities/diagnostic-session.entity");
const obd2_reading_entity_1 = require("./entities/obd2-reading.entity");
describe('DiagnosticsService', () => {
    let service;
    let sessionRepository;
    let readingRepository;
    const mockSession = {
        id: 'session-uuid',
        tenantId: 'tenant-uuid',
        vehicleId: 'vehicle-uuid',
        technicianId: 'technician-uuid',
        type: diagnostic_session_entity_1.SessionType.DIAGNOSTIC,
        status: diagnostic_session_entity_1.SessionStatus.ACTIVE,
        startTime: new Date('2024-01-15T10:00:00Z'),
        mileageAtSession: 50000,
        notes: 'Initial diagnostic session',
        createdAt: new Date(),
        updatedAt: new Date(),
    };
    const mockCompletedSession = {
        ...mockSession,
        id: 'completed-session-uuid',
        status: diagnostic_session_entity_1.SessionStatus.COMPLETED,
        endTime: new Date('2024-01-15T11:00:00Z'),
        dtcCodes: ['P0300', 'P0171'],
        performanceMetrics: {
            maxRpm: 6500,
            maxSpeed: 120,
            avgLoad: 45,
        },
    };
    const mockReading = {
        id: 'reading-uuid',
        sessionId: 'session-uuid',
        timestamp: new Date('2024-01-15T10:05:00Z'),
        pid: '010C',
        name: 'Engine RPM',
        value: 2500,
        unit: 'RPM',
        rawResponse: '41 0C 09 C4',
        createdAt: new Date(),
    };
    const mockSessionQueryBuilder = {
        where: jest.fn().mockReturnThis(),
        andWhere: jest.fn().mockReturnThis(),
        orderBy: jest.fn().mockReturnThis(),
        skip: jest.fn().mockReturnThis(),
        take: jest.fn().mockReturnThis(),
        getCount: jest.fn(),
        getMany: jest.fn(),
    };
    const mockReadingQueryBuilder = {
        where: jest.fn().mockReturnThis(),
        andWhere: jest.fn().mockReturnThis(),
        orderBy: jest.fn().mockReturnThis(),
        take: jest.fn().mockReturnThis(),
        getMany: jest.fn(),
    };
    beforeEach(async () => {
        const mockSessionRepository = {
            findOne: jest.fn(),
            find: jest.fn(),
            create: jest.fn(),
            save: jest.fn(),
            createQueryBuilder: jest.fn(() => mockSessionQueryBuilder),
        };
        const mockReadingRepository = {
            find: jest.fn(),
            create: jest.fn(),
            save: jest.fn(),
            createQueryBuilder: jest.fn(() => mockReadingQueryBuilder),
        };
        const module = await testing_1.Test.createTestingModule({
            providers: [
                diagnostics_service_1.DiagnosticsService,
                {
                    provide: (0, typeorm_1.getRepositoryToken)(diagnostic_session_entity_1.DiagnosticSession),
                    useValue: mockSessionRepository,
                },
                {
                    provide: (0, typeorm_1.getRepositoryToken)(obd2_reading_entity_1.OBD2Reading),
                    useValue: mockReadingRepository,
                },
            ],
        }).compile();
        service = module.get(diagnostics_service_1.DiagnosticsService);
        sessionRepository = module.get((0, typeorm_1.getRepositoryToken)(diagnostic_session_entity_1.DiagnosticSession));
        readingRepository = module.get((0, typeorm_1.getRepositoryToken)(obd2_reading_entity_1.OBD2Reading));
        jest.clearAllMocks();
    });
    describe('createSession', () => {
        it('should create a new diagnostic session', async () => {
            const dto = {
                vehicleId: 'vehicle-uuid',
                mileageAtSession: 50000,
                notes: 'Initial diagnostic session',
            };
            sessionRepository.create.mockReturnValue(mockSession);
            sessionRepository.save.mockResolvedValue(mockSession);
            const result = await service.createSession('tenant-uuid', 'technician-uuid', dto);
            expect(sessionRepository.create).toHaveBeenCalledWith(expect.objectContaining({
                tenantId: 'tenant-uuid',
                technicianId: 'technician-uuid',
                vehicleId: dto.vehicleId,
                type: diagnostic_session_entity_1.SessionType.DIAGNOSTIC,
                status: diagnostic_session_entity_1.SessionStatus.ACTIVE,
            }));
            expect(result.status).toBe(diagnostic_session_entity_1.SessionStatus.ACTIVE);
        });
        it('should create a performance session', async () => {
            const dto = {
                vehicleId: 'vehicle-uuid',
                type: diagnostic_session_entity_1.SessionType.PERFORMANCE,
            };
            const performanceSession = {
                ...mockSession,
                type: diagnostic_session_entity_1.SessionType.PERFORMANCE,
            };
            sessionRepository.create.mockReturnValue(performanceSession);
            sessionRepository.save.mockResolvedValue(performanceSession);
            const result = await service.createSession('tenant-uuid', 'technician-uuid', dto);
            expect(result.type).toBe(diagnostic_session_entity_1.SessionType.PERFORMANCE);
        });
        it('should create a tuning session', async () => {
            const dto = {
                vehicleId: 'vehicle-uuid',
                type: diagnostic_session_entity_1.SessionType.TUNE,
                notes: 'Stage 2 tune installation',
            };
            const tuneSession = {
                ...mockSession,
                type: diagnostic_session_entity_1.SessionType.TUNE,
                notes: 'Stage 2 tune installation',
            };
            sessionRepository.create.mockReturnValue(tuneSession);
            sessionRepository.save.mockResolvedValue(tuneSession);
            const result = await service.createSession('tenant-uuid', 'technician-uuid', dto);
            expect(result.type).toBe(diagnostic_session_entity_1.SessionType.TUNE);
        });
    });
    describe('findAllSessions', () => {
        it('should return paginated sessions', async () => {
            mockSessionQueryBuilder.getCount.mockResolvedValue(1);
            mockSessionQueryBuilder.getMany.mockResolvedValue([mockSession]);
            const result = await service.findAllSessions({
                tenantId: 'tenant-uuid',
                page: 1,
                limit: 20,
            });
            expect(result.items).toHaveLength(1);
            expect(result.total).toBe(1);
            expect(result.page).toBe(1);
            expect(result.limit).toBe(20);
            expect(result.totalPages).toBe(1);
        });
        it('should filter by vehicleId', async () => {
            mockSessionQueryBuilder.getCount.mockResolvedValue(1);
            mockSessionQueryBuilder.getMany.mockResolvedValue([mockSession]);
            await service.findAllSessions({
                tenantId: 'tenant-uuid',
                vehicleId: 'vehicle-uuid',
            });
            expect(mockSessionQueryBuilder.andWhere).toHaveBeenCalledWith('session.vehicleId = :vehicleId', { vehicleId: 'vehicle-uuid' });
        });
        it('should filter by technicianId', async () => {
            mockSessionQueryBuilder.getCount.mockResolvedValue(1);
            mockSessionQueryBuilder.getMany.mockResolvedValue([mockSession]);
            await service.findAllSessions({
                tenantId: 'tenant-uuid',
                technicianId: 'technician-uuid',
            });
            expect(mockSessionQueryBuilder.andWhere).toHaveBeenCalledWith('session.technicianId = :technicianId', { technicianId: 'technician-uuid' });
        });
        it('should filter by session type', async () => {
            mockSessionQueryBuilder.getCount.mockResolvedValue(1);
            mockSessionQueryBuilder.getMany.mockResolvedValue([mockSession]);
            await service.findAllSessions({
                tenantId: 'tenant-uuid',
                type: diagnostic_session_entity_1.SessionType.DIAGNOSTIC,
            });
            expect(mockSessionQueryBuilder.andWhere).toHaveBeenCalledWith('session.type = :type', { type: diagnostic_session_entity_1.SessionType.DIAGNOSTIC });
        });
        it('should filter by session status', async () => {
            mockSessionQueryBuilder.getCount.mockResolvedValue(1);
            mockSessionQueryBuilder.getMany.mockResolvedValue([mockCompletedSession]);
            await service.findAllSessions({
                tenantId: 'tenant-uuid',
                status: diagnostic_session_entity_1.SessionStatus.COMPLETED,
            });
            expect(mockSessionQueryBuilder.andWhere).toHaveBeenCalledWith('session.status = :status', { status: diagnostic_session_entity_1.SessionStatus.COMPLETED });
        });
        it('should filter by date range', async () => {
            mockSessionQueryBuilder.getCount.mockResolvedValue(1);
            mockSessionQueryBuilder.getMany.mockResolvedValue([mockSession]);
            const startDate = new Date('2024-01-01');
            const endDate = new Date('2024-01-31');
            await service.findAllSessions({
                tenantId: 'tenant-uuid',
                startDate,
                endDate,
            });
            expect(mockSessionQueryBuilder.andWhere).toHaveBeenCalledWith('session.startTime BETWEEN :startDate AND :endDate', { startDate, endDate });
        });
        it('should handle pagination correctly', async () => {
            mockSessionQueryBuilder.getCount.mockResolvedValue(50);
            mockSessionQueryBuilder.getMany.mockResolvedValue([mockSession]);
            const result = await service.findAllSessions({
                tenantId: 'tenant-uuid',
                page: 3,
                limit: 10,
            });
            expect(mockSessionQueryBuilder.skip).toHaveBeenCalledWith(20);
            expect(mockSessionQueryBuilder.take).toHaveBeenCalledWith(10);
            expect(result.totalPages).toBe(5);
        });
    });
    describe('findSession', () => {
        it('should return a session by id', async () => {
            sessionRepository.findOne.mockResolvedValue(mockSession);
            const result = await service.findSession('tenant-uuid', 'session-uuid');
            expect(sessionRepository.findOne).toHaveBeenCalledWith({
                where: { id: 'session-uuid', tenantId: 'tenant-uuid' },
            });
            expect(result.id).toBe('session-uuid');
        });
        it('should throw NotFoundException if session not found', async () => {
            sessionRepository.findOne.mockResolvedValue(null);
            await expect(service.findSession('tenant-uuid', 'non-existent-uuid')).rejects.toThrow(common_1.NotFoundException);
            await expect(service.findSession('tenant-uuid', 'non-existent-uuid')).rejects.toThrow('Session with ID non-existent-uuid not found');
        });
    });
    describe('endSession', () => {
        it('should end an active session', async () => {
            const dto = {
                notes: 'Session completed successfully',
                dtcCodes: ['P0300'],
                performanceMetrics: { maxRpm: 6500 },
            };
            sessionRepository.findOne.mockResolvedValue({ ...mockSession, status: diagnostic_session_entity_1.SessionStatus.ACTIVE });
            sessionRepository.save.mockResolvedValue({
                ...mockSession,
                status: diagnostic_session_entity_1.SessionStatus.COMPLETED,
                endTime: expect.any(Date),
                notes: dto.notes,
                dtcCodes: dto.dtcCodes,
                performanceMetrics: dto.performanceMetrics,
            });
            const result = await service.endSession('tenant-uuid', 'session-uuid', dto);
            expect(result.status).toBe(diagnostic_session_entity_1.SessionStatus.COMPLETED);
            expect(result.notes).toBe('Session completed successfully');
        });
        it('should throw BadRequestException if session is not active', async () => {
            sessionRepository.findOne.mockResolvedValue(mockCompletedSession);
            await expect(service.endSession('tenant-uuid', 'completed-session-uuid', {})).rejects.toThrow(common_1.BadRequestException);
            await expect(service.endSession('tenant-uuid', 'completed-session-uuid', {})).rejects.toThrow('Session is not active');
        });
        it('should update only provided fields', async () => {
            const dto = {
                dtcCodes: ['P0171', 'P0300'],
            };
            sessionRepository.findOne.mockResolvedValue({ ...mockSession, status: diagnostic_session_entity_1.SessionStatus.ACTIVE });
            sessionRepository.save.mockResolvedValue({
                ...mockSession,
                status: diagnostic_session_entity_1.SessionStatus.COMPLETED,
                dtcCodes: dto.dtcCodes,
            });
            const result = await service.endSession('tenant-uuid', 'session-uuid', dto);
            expect(result.dtcCodes).toEqual(['P0171', 'P0300']);
        });
    });
    describe('cancelSession', () => {
        it('should cancel an active session', async () => {
            sessionRepository.findOne.mockResolvedValue({ ...mockSession, status: diagnostic_session_entity_1.SessionStatus.ACTIVE });
            sessionRepository.save.mockResolvedValue({
                ...mockSession,
                status: diagnostic_session_entity_1.SessionStatus.CANCELLED,
                endTime: expect.any(Date),
            });
            const result = await service.cancelSession('tenant-uuid', 'session-uuid');
            expect(result.status).toBe(diagnostic_session_entity_1.SessionStatus.CANCELLED);
        });
        it('should throw BadRequestException if session is not active', async () => {
            sessionRepository.findOne.mockResolvedValue(mockCompletedSession);
            await expect(service.cancelSession('tenant-uuid', 'completed-session-uuid')).rejects.toThrow(common_1.BadRequestException);
        });
    });
    describe('logReading', () => {
        it('should log a single OBD2 reading', async () => {
            const dto = {
                sessionId: 'session-uuid',
                pid: '010C',
                name: 'Engine RPM',
                value: 2500,
                unit: 'RPM',
                rawResponse: '41 0C 09 C4',
            };
            readingRepository.create.mockReturnValue(mockReading);
            readingRepository.save.mockResolvedValue(mockReading);
            const result = await service.logReading(dto);
            expect(readingRepository.create).toHaveBeenCalledWith(expect.objectContaining({
                sessionId: dto.sessionId,
                pid: dto.pid,
                name: dto.name,
                value: dto.value,
                unit: dto.unit,
            }));
            expect(result.pid).toBe('010C');
        });
        it('should use provided timestamp', async () => {
            const timestamp = '2024-01-15T10:05:00Z';
            const dto = {
                sessionId: 'session-uuid',
                pid: '010D',
                name: 'Vehicle Speed',
                value: 60,
                unit: 'km/h',
                timestamp,
            };
            readingRepository.create.mockReturnValue({
                ...mockReading,
                timestamp: new Date(timestamp),
            });
            readingRepository.save.mockResolvedValue({
                ...mockReading,
                timestamp: new Date(timestamp),
            });
            const result = await service.logReading(dto);
            expect(readingRepository.create).toHaveBeenCalledWith(expect.objectContaining({
                timestamp: new Date(timestamp),
            }));
        });
    });
    describe('logReadingsBatch', () => {
        it('should log multiple readings at once', async () => {
            const readings = [
                {
                    sessionId: 'session-uuid',
                    pid: '010C',
                    name: 'Engine RPM',
                    value: 2500,
                    unit: 'RPM',
                },
                {
                    sessionId: 'session-uuid',
                    pid: '010D',
                    name: 'Vehicle Speed',
                    value: 60,
                    unit: 'km/h',
                },
            ];
            const mockReadings = readings.map((r, i) => ({
                id: `reading-uuid-${i}`,
                ...r,
                timestamp: expect.any(Date),
                createdAt: new Date(),
            }));
            readingRepository.create.mockReturnValueOnce(mockReadings[0]);
            readingRepository.create.mockReturnValueOnce(mockReadings[1]);
            readingRepository.save.mockResolvedValue(mockReadings);
            const result = await service.logReadingsBatch(readings);
            expect(readingRepository.create).toHaveBeenCalledTimes(2);
            expect(readingRepository.save).toHaveBeenCalledWith(expect.any(Array));
            expect(result).toHaveLength(2);
        });
    });
    describe('getSessionReadings', () => {
        it('should return readings for a session', async () => {
            mockReadingQueryBuilder.getMany.mockResolvedValue([mockReading]);
            const result = await service.getSessionReadings('session-uuid');
            expect(mockReadingQueryBuilder.where).toHaveBeenCalledWith('reading.sessionId = :sessionId', { sessionId: 'session-uuid' });
            expect(result).toHaveLength(1);
        });
        it('should filter by PID', async () => {
            mockReadingQueryBuilder.getMany.mockResolvedValue([mockReading]);
            await service.getSessionReadings('session-uuid', { pid: '010C' });
            expect(mockReadingQueryBuilder.andWhere).toHaveBeenCalledWith('reading.pid = :pid', { pid: '010C' });
        });
        it('should filter by time range', async () => {
            mockReadingQueryBuilder.getMany.mockResolvedValue([mockReading]);
            const startTime = new Date('2024-01-15T10:00:00Z');
            const endTime = new Date('2024-01-15T10:30:00Z');
            await service.getSessionReadings('session-uuid', { startTime, endTime });
            expect(mockReadingQueryBuilder.andWhere).toHaveBeenCalledWith('reading.timestamp BETWEEN :startTime AND :endTime', { startTime, endTime });
        });
        it('should limit results', async () => {
            mockReadingQueryBuilder.getMany.mockResolvedValue([mockReading]);
            await service.getSessionReadings('session-uuid', { limit: 100 });
            expect(mockReadingQueryBuilder.take).toHaveBeenCalledWith(100);
        });
    });
    describe('getLatestReadings', () => {
        it('should return latest reading for each PID', async () => {
            const readings = [
                { ...mockReading, pid: '010C', value: 3000 },
                { ...mockReading, pid: '010C', value: 2500 },
                { ...mockReading, pid: '010D', value: 80 },
                { ...mockReading, pid: '010D', value: 60 },
            ];
            mockReadingQueryBuilder.getMany.mockResolvedValue(readings);
            const result = await service.getLatestReadings('session-uuid');
            expect(result.size).toBe(2);
            expect(result.get('010C')?.value).toBe(3000);
            expect(result.get('010D')?.value).toBe(80);
        });
    });
    describe('getSessionAnalytics', () => {
        it('should return session analytics', async () => {
            const sessions = [
                mockSession,
                mockCompletedSession,
                {
                    ...mockSession,
                    id: 'cancelled-session-uuid',
                    status: diagnostic_session_entity_1.SessionStatus.CANCELLED,
                    endTime: new Date('2024-01-15T10:30:00Z'),
                },
            ];
            sessionRepository.find.mockResolvedValue(sessions);
            const startDate = new Date('2024-01-01');
            const endDate = new Date('2024-01-31');
            const result = await service.getSessionAnalytics('tenant-uuid', startDate, endDate);
            expect(result.totalSessions).toBe(3);
            expect(result.byType).toBeDefined();
            expect(result.byStatus).toBeDefined();
            expect(result.commonDtcCodes).toBeDefined();
        });
        it('should calculate correct duration', async () => {
            const sessions = [
                {
                    ...mockSession,
                    startTime: new Date('2024-01-15T10:00:00Z'),
                    endTime: new Date('2024-01-15T11:00:00Z'),
                },
                {
                    ...mockSession,
                    id: 'session-2',
                    startTime: new Date('2024-01-16T10:00:00Z'),
                    endTime: new Date('2024-01-16T10:30:00Z'),
                },
            ];
            sessionRepository.find.mockResolvedValue(sessions);
            const result = await service.getSessionAnalytics('tenant-uuid', new Date('2024-01-01'), new Date('2024-01-31'));
            expect(result.totalDuration).toBe(5400);
            expect(result.avgDuration).toBe(2700);
        });
        it('should count DTC codes correctly', async () => {
            const sessions = [
                {
                    ...mockSession,
                    dtcCodes: ['P0300', 'P0171'],
                },
                {
                    ...mockSession,
                    id: 'session-2',
                    dtcCodes: ['P0300', 'P0420'],
                },
            ];
            sessionRepository.find.mockResolvedValue(sessions);
            const result = await service.getSessionAnalytics('tenant-uuid', new Date('2024-01-01'), new Date('2024-01-31'));
            expect(result.commonDtcCodes).toContainEqual({ code: 'P0300', count: 2 });
            expect(result.commonDtcCodes.find(d => d.code === 'P0300')?.count).toBe(2);
        });
        it('should return top 10 DTC codes', async () => {
            const dtcCodes = Array.from({ length: 15 }, (_, i) => `P0${i.toString().padStart(3, '0')}`);
            const sessions = [
                {
                    ...mockSession,
                    dtcCodes,
                },
            ];
            sessionRepository.find.mockResolvedValue(sessions);
            const result = await service.getSessionAnalytics('tenant-uuid', new Date('2024-01-01'), new Date('2024-01-31'));
            expect(result.commonDtcCodes.length).toBeLessThanOrEqual(10);
        });
        it('should handle empty sessions', async () => {
            sessionRepository.find.mockResolvedValue([]);
            const result = await service.getSessionAnalytics('tenant-uuid', new Date('2024-01-01'), new Date('2024-01-31'));
            expect(result.totalSessions).toBe(0);
            expect(result.totalDuration).toBe(0);
            expect(result.avgDuration).toBe(0);
            expect(result.byType).toHaveLength(0);
            expect(result.byStatus).toHaveLength(0);
            expect(result.commonDtcCodes).toHaveLength(0);
        });
    });
});
//# sourceMappingURL=diagnostics.service.spec.js.map
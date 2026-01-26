"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const testing_1 = require("@nestjs/testing");
const diagnostics_controller_1 = require("./diagnostics.controller");
const diagnostics_service_1 = require("./diagnostics.service");
const diagnostic_session_entity_1 = require("./entities/diagnostic-session.entity");
const user_entity_1 = require("../auth/entities/user.entity");
describe('DiagnosticsController', () => {
    let controller;
    let service;
    const mockUser = {
        id: 'user-uuid',
        email: 'test@example.com',
        firstName: 'Test',
        lastName: 'User',
        role: user_entity_1.UserRole.TECHNICIAN,
        tenantId: 'tenant-uuid',
        isActive: true,
    };
    const mockSession = {
        id: 'session-uuid',
        tenantId: 'tenant-uuid',
        vehicleId: 'vehicle-uuid',
        technicianId: 'user-uuid',
        type: diagnostic_session_entity_1.SessionType.DIAGNOSTIC,
        status: diagnostic_session_entity_1.SessionStatus.ACTIVE,
        startTime: new Date(),
        endTime: null,
        mileageAtSession: 25000,
        createdAt: new Date(),
        updatedAt: new Date(),
    };
    const mockReading = {
        id: 'reading-uuid',
        sessionId: 'session-uuid',
        timestamp: new Date(),
        pid: '010C',
        name: 'Engine RPM',
        value: 3500,
        unit: 'RPM',
        rawResponse: '41 0C 0D AC',
        createdAt: new Date(),
    };
    beforeEach(async () => {
        const mockService = {
            createSession: jest.fn(),
            findAllSessions: jest.fn(),
            findSession: jest.fn(),
            endSession: jest.fn(),
            cancelSession: jest.fn(),
            logReading: jest.fn(),
            logReadingsBatch: jest.fn(),
            getSessionReadings: jest.fn(),
            getLatestReadings: jest.fn(),
            getSessionAnalytics: jest.fn(),
        };
        const module = await testing_1.Test.createTestingModule({
            controllers: [diagnostics_controller_1.DiagnosticsController],
            providers: [
                {
                    provide: diagnostics_service_1.DiagnosticsService,
                    useValue: mockService,
                },
            ],
        }).compile();
        controller = module.get(diagnostics_controller_1.DiagnosticsController);
        service = module.get(diagnostics_service_1.DiagnosticsService);
    });
    describe('createSession', () => {
        it('should create a new diagnostic session', async () => {
            const dto = {
                vehicleId: 'vehicle-uuid',
                type: diagnostic_session_entity_1.SessionType.DIAGNOSTIC,
                mileageAtSession: 25000,
            };
            service.createSession.mockResolvedValue(mockSession);
            const result = await controller.createSession(mockUser, dto);
            expect(service.createSession).toHaveBeenCalledWith('tenant-uuid', 'user-uuid', dto);
            expect(result.status).toBe(diagnostic_session_entity_1.SessionStatus.ACTIVE);
        });
    });
    describe('findAllSessions', () => {
        it('should return paginated sessions', async () => {
            const paginatedResult = {
                items: [mockSession],
                total: 1,
                page: 1,
                limit: 20,
                totalPages: 1,
            };
            service.findAllSessions.mockResolvedValue(paginatedResult);
            const result = await controller.findAllSessions(mockUser);
            expect(service.findAllSessions).toHaveBeenCalledWith(expect.objectContaining({ tenantId: 'tenant-uuid' }));
            expect(result.items).toHaveLength(1);
        });
        it('should filter by vehicle ID', async () => {
            const paginatedResult = {
                items: [mockSession],
                total: 1,
                page: 1,
                limit: 20,
                totalPages: 1,
            };
            service.findAllSessions.mockResolvedValue(paginatedResult);
            await controller.findAllSessions(mockUser, 'vehicle-uuid');
            expect(service.findAllSessions).toHaveBeenCalledWith(expect.objectContaining({ vehicleId: 'vehicle-uuid' }));
        });
    });
    describe('findSession', () => {
        it('should return a single session by ID', async () => {
            service.findSession.mockResolvedValue(mockSession);
            const result = await controller.findSession(mockUser, 'session-uuid');
            expect(service.findSession).toHaveBeenCalledWith('tenant-uuid', 'session-uuid');
            expect(result.id).toBe('session-uuid');
        });
    });
    describe('endSession', () => {
        it('should end an active session', async () => {
            const dto = {
                notes: 'Session completed successfully',
                dtcCodes: ['P0300', 'P0171'],
            };
            const completedSession = {
                ...mockSession,
                status: diagnostic_session_entity_1.SessionStatus.COMPLETED,
                endTime: new Date(),
            };
            service.endSession.mockResolvedValue(completedSession);
            const result = await controller.endSession(mockUser, 'session-uuid', dto);
            expect(service.endSession).toHaveBeenCalledWith('tenant-uuid', 'session-uuid', dto);
            expect(result.status).toBe(diagnostic_session_entity_1.SessionStatus.COMPLETED);
        });
    });
    describe('cancelSession', () => {
        it('should cancel an active session', async () => {
            const cancelledSession = {
                ...mockSession,
                status: diagnostic_session_entity_1.SessionStatus.CANCELLED,
                endTime: new Date(),
            };
            service.cancelSession.mockResolvedValue(cancelledSession);
            const result = await controller.cancelSession(mockUser, 'session-uuid');
            expect(service.cancelSession).toHaveBeenCalledWith('tenant-uuid', 'session-uuid');
            expect(result.status).toBe(diagnostic_session_entity_1.SessionStatus.CANCELLED);
        });
    });
    describe('logReading', () => {
        it('should log a single OBD2 reading', async () => {
            const dto = {
                sessionId: 'session-uuid',
                pid: '010C',
                name: 'Engine RPM',
                value: 3500,
                unit: 'RPM',
            };
            service.logReading.mockResolvedValue(mockReading);
            const result = await controller.logReading(dto);
            expect(service.logReading).toHaveBeenCalledWith(dto);
            expect(result.value).toBe(3500);
        });
    });
    describe('logReadingsBatch', () => {
        it('should log multiple OBD2 readings', async () => {
            const dto = {
                readings: [
                    { sessionId: 'session-uuid', pid: '010C', name: 'RPM', value: 3500, unit: 'RPM' },
                    { sessionId: 'session-uuid', pid: '010D', name: 'Speed', value: 60, unit: 'km/h' },
                ],
            };
            service.logReadingsBatch.mockResolvedValue([mockReading, mockReading]);
            const result = await controller.logReadingsBatch(dto);
            expect(service.logReadingsBatch).toHaveBeenCalledWith(dto.readings);
            expect(result).toHaveLength(2);
        });
    });
    describe('getSessionReadings', () => {
        it('should return readings for a session', async () => {
            service.getSessionReadings.mockResolvedValue([mockReading]);
            const result = await controller.getSessionReadings('session-uuid');
            expect(service.getSessionReadings).toHaveBeenCalledWith('session-uuid', {
                pid: undefined,
                startTime: undefined,
                endTime: undefined,
                limit: undefined,
            });
            expect(result).toHaveLength(1);
        });
    });
    describe('getLatestReadings', () => {
        it('should return latest readings by PID', async () => {
            const latestMap = new Map([['010C', mockReading]]);
            service.getLatestReadings.mockResolvedValue(latestMap);
            const result = await controller.getLatestReadings('session-uuid');
            expect(service.getLatestReadings).toHaveBeenCalledWith('session-uuid');
            expect(result['010C']).toBeDefined();
        });
    });
    describe('getAnalytics', () => {
        it('should return session analytics', async () => {
            const analytics = {
                totalSessions: 10,
                totalDuration: 3600,
                avgDuration: 360,
                byType: [{ type: 'diagnostic', count: 8 }],
                byStatus: [{ status: 'completed', count: 9 }],
                commonDtcCodes: [{ code: 'P0300', count: 3 }],
            };
            service.getSessionAnalytics.mockResolvedValue(analytics);
            const result = await controller.getAnalytics(mockUser, '2024-01-01', '2024-12-31');
            expect(service.getSessionAnalytics).toHaveBeenCalledWith('tenant-uuid', expect.any(Date), expect.any(Date));
            expect(result.totalSessions).toBe(10);
        });
    });
});
//# sourceMappingURL=diagnostics.controller.spec.js.map
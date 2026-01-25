import { Test, TestingModule } from '@nestjs/testing';
import { DiagnosticsController } from './diagnostics.controller';
import { DiagnosticsService } from './diagnostics.service';
import { CreateSessionDto, LogReadingDto, EndSessionDto } from './dto';
import { SessionStatus, SessionType } from './entities/diagnostic-session.entity';
import { UserRole } from '../auth/entities/user.entity';

describe('DiagnosticsController', () => {
    let controller: DiagnosticsController;
    let service: jest.Mocked<DiagnosticsService>;

    const mockUser = {
        id: 'user-uuid',
        email: 'test@example.com',
        firstName: 'Test',
        lastName: 'User',
        role: UserRole.TECHNICIAN,
        tenantId: 'tenant-uuid',
        isActive: true,
    };

    const mockSession = {
        id: 'session-uuid',
        tenantId: 'tenant-uuid',
        vehicleId: 'vehicle-uuid',
        technicianId: 'user-uuid',
        type: SessionType.DIAGNOSTIC,
        status: SessionStatus.ACTIVE,
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

        const module: TestingModule = await Test.createTestingModule({
            controllers: [DiagnosticsController],
            providers: [
                {
                    provide: DiagnosticsService,
                    useValue: mockService,
                },
            ],
        }).compile();

        controller = module.get<DiagnosticsController>(DiagnosticsController);
        service = module.get(DiagnosticsService);
    });

    describe('createSession', () => {
        it('should create a new diagnostic session', async () => {
            const dto: CreateSessionDto = {
                vehicleId: 'vehicle-uuid',
                type: SessionType.DIAGNOSTIC,
                mileageAtSession: 25000,
            };

            service.createSession.mockResolvedValue(mockSession as any);

            const result = await controller.createSession(mockUser as any, dto);

            expect(service.createSession).toHaveBeenCalledWith(
                'tenant-uuid',
                'user-uuid',
                dto,
            );
            expect(result.status).toBe(SessionStatus.ACTIVE);
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

            service.findAllSessions.mockResolvedValue(paginatedResult as any);

            const result = await controller.findAllSessions(mockUser as any);

            expect(service.findAllSessions).toHaveBeenCalledWith(
                expect.objectContaining({ tenantId: 'tenant-uuid' }),
            );
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

            service.findAllSessions.mockResolvedValue(paginatedResult as any);

            await controller.findAllSessions(mockUser as any, 'vehicle-uuid');

            expect(service.findAllSessions).toHaveBeenCalledWith(
                expect.objectContaining({ vehicleId: 'vehicle-uuid' }),
            );
        });
    });

    describe('findSession', () => {
        it('should return a single session by ID', async () => {
            service.findSession.mockResolvedValue(mockSession as any);

            const result = await controller.findSession(mockUser as any, 'session-uuid');

            expect(service.findSession).toHaveBeenCalledWith('tenant-uuid', 'session-uuid');
            expect(result.id).toBe('session-uuid');
        });
    });

    describe('endSession', () => {
        it('should end an active session', async () => {
            const dto: EndSessionDto = {
                notes: 'Session completed successfully',
                dtcCodes: ['P0300', 'P0171'],
            };

            const completedSession = {
                ...mockSession,
                status: SessionStatus.COMPLETED,
                endTime: new Date(),
            };

            service.endSession.mockResolvedValue(completedSession as any);

            const result = await controller.endSession(mockUser as any, 'session-uuid', dto);

            expect(service.endSession).toHaveBeenCalledWith('tenant-uuid', 'session-uuid', dto);
            expect(result.status).toBe(SessionStatus.COMPLETED);
        });
    });

    describe('cancelSession', () => {
        it('should cancel an active session', async () => {
            const cancelledSession = {
                ...mockSession,
                status: SessionStatus.CANCELLED,
                endTime: new Date(),
            };

            service.cancelSession.mockResolvedValue(cancelledSession as any);

            const result = await controller.cancelSession(mockUser as any, 'session-uuid');

            expect(service.cancelSession).toHaveBeenCalledWith('tenant-uuid', 'session-uuid');
            expect(result.status).toBe(SessionStatus.CANCELLED);
        });
    });

    describe('logReading', () => {
        it('should log a single OBD2 reading', async () => {
            const dto: LogReadingDto = {
                sessionId: 'session-uuid',
                pid: '010C',
                name: 'Engine RPM',
                value: 3500,
                unit: 'RPM',
            };

            service.logReading.mockResolvedValue(mockReading as any);

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

            service.logReadingsBatch.mockResolvedValue([mockReading, mockReading] as any);

            const result = await controller.logReadingsBatch(dto as any);

            expect(service.logReadingsBatch).toHaveBeenCalledWith(dto.readings);
            expect(result).toHaveLength(2);
        });
    });

    describe('getSessionReadings', () => {
        it('should return readings for a session', async () => {
            service.getSessionReadings.mockResolvedValue([mockReading] as any);

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
            service.getLatestReadings.mockResolvedValue(latestMap as any);

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

            const result = await controller.getAnalytics(
                mockUser as any,
                '2024-01-01',
                '2024-12-31',
            );

            expect(service.getSessionAnalytics).toHaveBeenCalledWith(
                'tenant-uuid',
                expect.any(Date),
                expect.any(Date),
            );
            expect(result.totalSessions).toBe(10);
        });
    });
});

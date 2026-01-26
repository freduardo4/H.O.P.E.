"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const testing_1 = require("@nestjs/testing");
const reports_controller_1 = require("./reports.controller");
const reports_service_1 = require("./reports.service");
const report_entity_1 = require("./entities/report.entity");
const user_entity_1 = require("../auth/entities/user.entity");
describe('ReportsController', () => {
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
    const mockReport = {
        id: 'report-uuid',
        tenantId: 'tenant-uuid',
        sessionId: 'session-uuid',
        vehicleId: 'vehicle-uuid',
        customerId: 'customer-uuid',
        title: 'Diagnostic Report - 2024-01-15',
        description: 'Full vehicle diagnostic',
        type: report_entity_1.ReportType.DIAGNOSTIC,
        status: report_entity_1.ReportStatus.DRAFT,
        format: report_entity_1.ReportFormat.PDF,
        s3Key: null,
        fileSize: null,
        summary: {
            totalReadings: 150,
            dtcsFound: 2,
            anomaliesDetected: 1,
            recommendations: ['Replace air filter', 'Check coolant level'],
        },
        vehicleInfo: {
            make: 'Volkswagen',
            model: 'Golf GTI',
            year: 2022,
            vin: '1HGBH41JXMN109186',
            mileage: 25000,
        },
        diagnosticData: {
            dtcs: [{ code: 'P0300', description: 'Random misfire', severity: 'warning' }],
            parameters: [{ name: 'RPM', min: 800, max: 4500, avg: 2100, unit: 'RPM' }],
            anomalies: [],
        },
        tuningData: null,
        notes: 'Customer reported rough idle',
        createdById: 'user-uuid',
        approvedById: null,
        approvedAt: null,
        sentAt: null,
        sentTo: null,
        isActive: true,
        createdAt: new Date(),
        updatedAt: new Date(),
    };
    beforeEach(async () => {
        const mockService = {
            create: jest.fn(),
            findAll: jest.fn(),
            findOne: jest.fn(),
            update: jest.fn(),
            updateStatus: jest.fn(),
            markAsSent: jest.fn(),
            remove: jest.fn(),
            generateFromSession: jest.fn(),
            getStats: jest.fn(),
        };
        const module = await testing_1.Test.createTestingModule({
            controllers: [reports_controller_1.ReportsController],
            providers: [
                {
                    provide: reports_service_1.ReportsService,
                    useValue: mockService,
                },
            ],
        }).compile();
        controller = module.get(reports_controller_1.ReportsController);
        service = module.get(reports_service_1.ReportsService);
    });
    describe('create', () => {
        it('should create a new report', async () => {
            const dto = {
                title: 'Diagnostic Report - 2024-01-15',
                type: report_entity_1.ReportType.DIAGNOSTIC,
                vehicleId: 'vehicle-uuid',
                sessionId: 'session-uuid',
            };
            service.create.mockResolvedValue(mockReport);
            const result = await controller.create(mockUser, dto);
            expect(service.create).toHaveBeenCalledWith('tenant-uuid', 'user-uuid', dto);
            expect(result.title).toBe('Diagnostic Report - 2024-01-15');
            expect(result.type).toBe(report_entity_1.ReportType.DIAGNOSTIC);
        });
    });
    describe('generateFromSession', () => {
        it('should generate report from diagnostic session', async () => {
            service.generateFromSession.mockResolvedValue(mockReport);
            const vehicleInfo = {
                make: 'Volkswagen',
                model: 'Golf GTI',
                year: 2022,
            };
            const result = await controller.generateFromSession(mockUser, 'session-uuid', vehicleInfo);
            expect(service.generateFromSession).toHaveBeenCalledWith('tenant-uuid', 'session-uuid', 'user-uuid', vehicleInfo);
            expect(result.sessionId).toBe('session-uuid');
        });
        it('should generate report without vehicle info', async () => {
            service.generateFromSession.mockResolvedValue(mockReport);
            await controller.generateFromSession(mockUser, 'session-uuid');
            expect(service.generateFromSession).toHaveBeenCalledWith('tenant-uuid', 'session-uuid', 'user-uuid', undefined);
        });
    });
    describe('findAll', () => {
        it('should return paginated reports', async () => {
            const paginatedResult = {
                items: [mockReport],
                total: 1,
                page: 1,
                limit: 20,
                totalPages: 1,
            };
            service.findAll.mockResolvedValue(paginatedResult);
            const result = await controller.findAll(mockUser);
            expect(service.findAll).toHaveBeenCalledWith({
                tenantId: 'tenant-uuid',
                vehicleId: undefined,
                customerId: undefined,
                sessionId: undefined,
                type: undefined,
                status: undefined,
                page: 1,
                limit: 20,
            });
            expect(result.items).toHaveLength(1);
        });
        it('should filter by vehicle ID', async () => {
            const paginatedResult = {
                items: [mockReport],
                total: 1,
                page: 1,
                limit: 20,
                totalPages: 1,
            };
            service.findAll.mockResolvedValue(paginatedResult);
            await controller.findAll(mockUser, 'vehicle-uuid');
            expect(service.findAll).toHaveBeenCalledWith(expect.objectContaining({ vehicleId: 'vehicle-uuid' }));
        });
        it('should filter by report type', async () => {
            const paginatedResult = {
                items: [mockReport],
                total: 1,
                page: 1,
                limit: 20,
                totalPages: 1,
            };
            service.findAll.mockResolvedValue(paginatedResult);
            await controller.findAll(mockUser, undefined, undefined, undefined, report_entity_1.ReportType.DIAGNOSTIC);
            expect(service.findAll).toHaveBeenCalledWith(expect.objectContaining({ type: report_entity_1.ReportType.DIAGNOSTIC }));
        });
        it('should filter by status', async () => {
            const paginatedResult = {
                items: [mockReport],
                total: 1,
                page: 1,
                limit: 20,
                totalPages: 1,
            };
            service.findAll.mockResolvedValue(paginatedResult);
            await controller.findAll(mockUser, undefined, undefined, undefined, undefined, report_entity_1.ReportStatus.DRAFT);
            expect(service.findAll).toHaveBeenCalledWith(expect.objectContaining({ status: report_entity_1.ReportStatus.DRAFT }));
        });
    });
    describe('findOne', () => {
        it('should return a single report by id', async () => {
            service.findOne.mockResolvedValue(mockReport);
            const result = await controller.findOne(mockUser, 'report-uuid');
            expect(service.findOne).toHaveBeenCalledWith('tenant-uuid', 'report-uuid');
            expect(result.id).toBe('report-uuid');
        });
    });
    describe('update', () => {
        it('should update a report', async () => {
            const updatedReport = {
                ...mockReport,
                title: 'Updated Report Title',
                notes: 'Updated notes',
            };
            service.update.mockResolvedValue(updatedReport);
            const result = await controller.update(mockUser, 'report-uuid', {
                title: 'Updated Report Title',
                notes: 'Updated notes',
            });
            expect(service.update).toHaveBeenCalledWith('tenant-uuid', 'report-uuid', {
                title: 'Updated Report Title',
                notes: 'Updated notes',
            });
            expect(result.title).toBe('Updated Report Title');
        });
    });
    describe('updateStatus', () => {
        it('should update report status', async () => {
            const approvedReport = {
                ...mockReport,
                status: report_entity_1.ReportStatus.APPROVED,
                approvedById: 'user-uuid',
                approvedAt: new Date(),
            };
            service.updateStatus.mockResolvedValue(approvedReport);
            const result = await controller.updateStatus(mockUser, 'report-uuid', report_entity_1.ReportStatus.APPROVED);
            expect(service.updateStatus).toHaveBeenCalledWith('tenant-uuid', 'report-uuid', report_entity_1.ReportStatus.APPROVED, 'user-uuid');
            expect(result.status).toBe(report_entity_1.ReportStatus.APPROVED);
        });
    });
    describe('markAsSent', () => {
        it('should mark report as sent', async () => {
            const sentReport = {
                ...mockReport,
                status: report_entity_1.ReportStatus.SENT,
                sentAt: new Date(),
                sentTo: 'customer@example.com',
            };
            service.markAsSent.mockResolvedValue(sentReport);
            const result = await controller.markAsSent(mockUser, 'report-uuid', 'customer@example.com');
            expect(service.markAsSent).toHaveBeenCalledWith('tenant-uuid', 'report-uuid', 'customer@example.com');
            expect(result.status).toBe(report_entity_1.ReportStatus.SENT);
            expect(result.sentTo).toBe('customer@example.com');
        });
    });
    describe('remove', () => {
        it('should soft delete a report', async () => {
            service.remove.mockResolvedValue(undefined);
            const result = await controller.remove(mockUser, 'report-uuid');
            expect(service.remove).toHaveBeenCalledWith('tenant-uuid', 'report-uuid');
            expect(result.message).toBe('Report deleted successfully');
        });
    });
    describe('getStats', () => {
        it('should return report statistics', async () => {
            const stats = {
                total: 50,
                byType: [
                    { type: report_entity_1.ReportType.DIAGNOSTIC, count: 30 },
                    { type: report_entity_1.ReportType.TUNING, count: 15 },
                    { type: report_entity_1.ReportType.INSPECTION, count: 5 },
                ],
                byStatus: [
                    { status: report_entity_1.ReportStatus.DRAFT, count: 10 },
                    { status: report_entity_1.ReportStatus.APPROVED, count: 25 },
                    { status: report_entity_1.ReportStatus.SENT, count: 15 },
                ],
                recentReports: [mockReport],
            };
            service.getStats.mockResolvedValue(stats);
            const result = await controller.getStats(mockUser);
            expect(service.getStats).toHaveBeenCalledWith('tenant-uuid');
            expect(result.total).toBe(50);
            expect(result.byType).toHaveLength(3);
            expect(result.byStatus).toHaveLength(3);
        });
    });
});
//# sourceMappingURL=reports.controller.spec.js.map
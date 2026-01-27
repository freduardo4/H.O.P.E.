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
        tenantId: 'tenant-uuid',
        email: 'test@example.com',
        role: user_entity_1.UserRole.TECHNICIAN,
    };
    const mockReport = {
        id: 'report-uuid',
        tenantId: 'tenant-uuid',
        vehicleId: 'vehicle-uuid',
        customerId: 'customer-uuid',
        reportType: report_entity_1.ReportType.DIAGNOSTIC,
        status: report_entity_1.ReportStatus.COMPLETED,
        title: 'Diagnostic Report',
        s3Key: 'tenant-uuid/reports/report.pdf',
        s3Bucket: 'hope-reports',
        generatedBy: 'user-uuid',
        createdAt: new Date(),
        updatedAt: new Date(),
    };
    const mockPaginatedResult = {
        data: [mockReport],
        total: 1,
        page: 1,
        limit: 20,
        totalPages: 1,
    };
    beforeEach(async () => {
        const mockService = {
            create: jest.fn(),
            findAll: jest.fn(),
            findOne: jest.fn(),
            getDownloadUrl: jest.fn(),
            regenerate: jest.fn(),
            remove: jest.fn(),
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
        jest.clearAllMocks();
    });
    describe('create', () => {
        it('should create a new report', async () => {
            const dto = {
                vehicleId: 'vehicle-uuid',
                customerId: 'customer-uuid',
                reportType: report_entity_1.ReportType.DIAGNOSTIC,
                title: 'Diagnostic Report',
            };
            service.create.mockResolvedValue(mockReport);
            const result = await controller.create(mockUser, dto);
            expect(service.create).toHaveBeenCalledWith('tenant-uuid', dto, 'user-uuid');
            expect(result.id).toBe('report-uuid');
        });
    });
    describe('findAll', () => {
        it('should return paginated reports', async () => {
            service.findAll.mockResolvedValue(mockPaginatedResult);
            const result = await controller.findAll(mockUser);
            expect(service.findAll).toHaveBeenCalledWith({
                tenantId: 'tenant-uuid',
                vehicleId: undefined,
                customerId: undefined,
                reportType: undefined,
                status: undefined,
                page: 1,
                limit: 20,
            });
            expect(result.data).toHaveLength(1);
        });
        it('should filter by vehicleId', async () => {
            service.findAll.mockResolvedValue(mockPaginatedResult);
            await controller.findAll(mockUser, 'vehicle-uuid');
            expect(service.findAll).toHaveBeenCalledWith(expect.objectContaining({ vehicleId: 'vehicle-uuid' }));
        });
        it('should filter by report type', async () => {
            service.findAll.mockResolvedValue(mockPaginatedResult);
            await controller.findAll(mockUser, undefined, undefined, report_entity_1.ReportType.TUNING);
            expect(service.findAll).toHaveBeenCalledWith(expect.objectContaining({ reportType: report_entity_1.ReportType.TUNING }));
        });
        it('should filter by status', async () => {
            service.findAll.mockResolvedValue(mockPaginatedResult);
            await controller.findAll(mockUser, undefined, undefined, undefined, report_entity_1.ReportStatus.PENDING);
            expect(service.findAll).toHaveBeenCalledWith(expect.objectContaining({ status: report_entity_1.ReportStatus.PENDING }));
        });
    });
    describe('findOne', () => {
        it('should return a single report', async () => {
            service.findOne.mockResolvedValue(mockReport);
            const result = await controller.findOne(mockUser, 'report-uuid');
            expect(service.findOne).toHaveBeenCalledWith('tenant-uuid', 'report-uuid');
            expect(result.id).toBe('report-uuid');
        });
    });
    describe('getDownloadUrl', () => {
        it('should return download URL with expiry', async () => {
            const downloadUrl = 'https://s3.amazonaws.com/bucket/report.pdf?signed=true';
            service.getDownloadUrl.mockResolvedValue(downloadUrl);
            const result = await controller.getDownloadUrl(mockUser, 'report-uuid');
            expect(service.getDownloadUrl).toHaveBeenCalledWith('tenant-uuid', 'report-uuid');
            expect(result.url).toBe(downloadUrl);
            expect(result.expiresIn).toBe(3600);
        });
    });
    describe('regenerate', () => {
        it('should regenerate a report', async () => {
            const regeneratedReport = { ...mockReport, status: report_entity_1.ReportStatus.PENDING };
            service.regenerate.mockResolvedValue(regeneratedReport);
            const result = await controller.regenerate(mockUser, 'report-uuid');
            expect(service.regenerate).toHaveBeenCalledWith('tenant-uuid', 'report-uuid', 'user-uuid');
            expect(result.status).toBe(report_entity_1.ReportStatus.PENDING);
        });
    });
    describe('remove', () => {
        it('should delete a report and return success message', async () => {
            service.remove.mockResolvedValue(undefined);
            const result = await controller.remove(mockUser, 'report-uuid');
            expect(service.remove).toHaveBeenCalledWith('tenant-uuid', 'report-uuid');
            expect(result.message).toBe('Report deleted successfully');
        });
    });
});
//# sourceMappingURL=reports.controller.spec.js.map
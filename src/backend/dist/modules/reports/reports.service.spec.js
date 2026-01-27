"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const testing_1 = require("@nestjs/testing");
const typeorm_1 = require("@nestjs/typeorm");
const common_1 = require("@nestjs/common");
const reports_service_1 = require("./reports.service");
const report_entity_1 = require("./entities/report.entity");
jest.mock('@aws-sdk/client-s3', () => ({
    S3Client: jest.fn().mockImplementation(() => ({
        send: jest.fn(),
    })),
    PutObjectCommand: jest.fn(),
    GetObjectCommand: jest.fn(),
}));
jest.mock('@aws-sdk/s3-request-presigner', () => ({
    getSignedUrl: jest.fn().mockResolvedValue('https://s3.amazonaws.com/signed-url'),
}));
describe('ReportsService', () => {
    let service;
    let repository;
    const mockReport = {
        id: 'report-uuid',
        tenantId: 'tenant-uuid',
        vehicleId: 'vehicle-uuid',
        customerId: 'customer-uuid',
        reportType: report_entity_1.ReportType.DIAGNOSTIC,
        status: report_entity_1.ReportStatus.PENDING,
        title: 'Diagnostic Report',
        description: 'Full diagnostic analysis',
        s3Key: null,
        s3Bucket: null,
        createdAt: new Date(),
        updatedAt: new Date(),
    };
    const mockCompletedReport = {
        ...mockReport,
        id: 'completed-report-uuid',
        status: report_entity_1.ReportStatus.COMPLETED,
        s3Key: 'tenant-uuid/reports/vehicle-uuid/12345-Diagnostic-Report.pdf',
        s3Bucket: 'hope-reports',
        fileSize: 1024,
        generatedBy: 'user-uuid',
        generatedAt: new Date(),
    };
    const mockQueryBuilder = {
        where: jest.fn().mockReturnThis(),
        andWhere: jest.fn().mockReturnThis(),
        orderBy: jest.fn().mockReturnThis(),
        skip: jest.fn().mockReturnThis(),
        take: jest.fn().mockReturnThis(),
        getManyAndCount: jest.fn(),
    };
    beforeEach(async () => {
        const mockRepository = {
            findOne: jest.fn(),
            find: jest.fn(),
            create: jest.fn(),
            save: jest.fn(),
            update: jest.fn(),
            remove: jest.fn(),
            createQueryBuilder: jest.fn(() => mockQueryBuilder),
        };
        const module = await testing_1.Test.createTestingModule({
            providers: [
                reports_service_1.ReportsService,
                {
                    provide: (0, typeorm_1.getRepositoryToken)(report_entity_1.Report),
                    useValue: mockRepository,
                },
            ],
        }).compile();
        service = module.get(reports_service_1.ReportsService);
        repository = module.get((0, typeorm_1.getRepositoryToken)(report_entity_1.Report));
        jest.clearAllMocks();
    });
    describe('create', () => {
        it('should create a report with pending status', async () => {
            const dto = {
                vehicleId: 'vehicle-uuid',
                reportType: report_entity_1.ReportType.DIAGNOSTIC,
                title: 'Diagnostic Report',
                description: 'Full diagnostic analysis',
            };
            repository.create.mockReturnValue(mockReport);
            repository.save.mockResolvedValue(mockReport);
            const result = await service.create('tenant-uuid', dto, 'user-uuid');
            expect(repository.create).toHaveBeenCalledWith({
                ...dto,
                tenantId: 'tenant-uuid',
                status: report_entity_1.ReportStatus.PENDING,
            });
            expect(result.status).toBe(report_entity_1.ReportStatus.PENDING);
            expect(result.title).toBe('Diagnostic Report');
        });
        it('should create a tuning report', async () => {
            const dto = {
                vehicleId: 'vehicle-uuid',
                reportType: report_entity_1.ReportType.TUNING,
                title: 'Stage 2 Tuning Report',
                data: {
                    beforePower: 300,
                    afterPower: 380,
                    modifications: ['ECU Remap', 'Downpipe'],
                },
            };
            const tuningReport = {
                ...mockReport,
                reportType: report_entity_1.ReportType.TUNING,
                title: 'Stage 2 Tuning Report',
                data: dto.data,
            };
            repository.create.mockReturnValue(tuningReport);
            repository.save.mockResolvedValue(tuningReport);
            const result = await service.create('tenant-uuid', dto, 'user-uuid');
            expect(result.reportType).toBe(report_entity_1.ReportType.TUNING);
        });
    });
    describe('findAll', () => {
        it('should return paginated reports', async () => {
            mockQueryBuilder.getManyAndCount.mockResolvedValue([[mockReport], 1]);
            const result = await service.findAll({
                tenantId: 'tenant-uuid',
                page: 1,
                limit: 20,
            });
            expect(result.data).toHaveLength(1);
            expect(result.total).toBe(1);
            expect(result.page).toBe(1);
            expect(result.limit).toBe(20);
            expect(result.totalPages).toBe(1);
        });
        it('should filter by vehicleId', async () => {
            mockQueryBuilder.getManyAndCount.mockResolvedValue([[mockReport], 1]);
            await service.findAll({
                tenantId: 'tenant-uuid',
                vehicleId: 'vehicle-uuid',
            });
            expect(mockQueryBuilder.andWhere).toHaveBeenCalledWith('report.vehicleId = :vehicleId', { vehicleId: 'vehicle-uuid' });
        });
        it('should filter by reportType', async () => {
            mockQueryBuilder.getManyAndCount.mockResolvedValue([[mockReport], 1]);
            await service.findAll({
                tenantId: 'tenant-uuid',
                reportType: report_entity_1.ReportType.DIAGNOSTIC,
            });
            expect(mockQueryBuilder.andWhere).toHaveBeenCalledWith('report.reportType = :reportType', { reportType: report_entity_1.ReportType.DIAGNOSTIC });
        });
        it('should filter by status', async () => {
            mockQueryBuilder.getManyAndCount.mockResolvedValue([[mockCompletedReport], 1]);
            await service.findAll({
                tenantId: 'tenant-uuid',
                status: report_entity_1.ReportStatus.COMPLETED,
            });
            expect(mockQueryBuilder.andWhere).toHaveBeenCalledWith('report.status = :status', { status: report_entity_1.ReportStatus.COMPLETED });
        });
        it('should filter by customerId', async () => {
            mockQueryBuilder.getManyAndCount.mockResolvedValue([[mockReport], 1]);
            await service.findAll({
                tenantId: 'tenant-uuid',
                customerId: 'customer-uuid',
            });
            expect(mockQueryBuilder.andWhere).toHaveBeenCalledWith('report.customerId = :customerId', { customerId: 'customer-uuid' });
        });
        it('should handle pagination correctly', async () => {
            mockQueryBuilder.getManyAndCount.mockResolvedValue([[mockReport], 50]);
            const result = await service.findAll({
                tenantId: 'tenant-uuid',
                page: 3,
                limit: 10,
            });
            expect(mockQueryBuilder.skip).toHaveBeenCalledWith(20);
            expect(mockQueryBuilder.take).toHaveBeenCalledWith(10);
            expect(result.totalPages).toBe(5);
        });
    });
    describe('findOne', () => {
        it('should return a report by id', async () => {
            repository.findOne.mockResolvedValue(mockReport);
            const result = await service.findOne('tenant-uuid', 'report-uuid');
            expect(repository.findOne).toHaveBeenCalledWith({
                where: { id: 'report-uuid', tenantId: 'tenant-uuid' },
            });
            expect(result.id).toBe('report-uuid');
        });
        it('should throw NotFoundException if report not found', async () => {
            repository.findOne.mockResolvedValue(null);
            await expect(service.findOne('tenant-uuid', 'non-existent-uuid')).rejects.toThrow(common_1.NotFoundException);
            await expect(service.findOne('tenant-uuid', 'non-existent-uuid')).rejects.toThrow('Report with ID non-existent-uuid not found');
        });
    });
    describe('getDownloadUrl', () => {
        it('should return signed URL for completed report', async () => {
            repository.findOne.mockResolvedValue(mockCompletedReport);
            const result = await service.getDownloadUrl('tenant-uuid', 'completed-report-uuid');
            expect(result).toBe('https://s3.amazonaws.com/signed-url');
        });
        it('should throw error if report is not completed', async () => {
            repository.findOne.mockResolvedValue(mockReport);
            await expect(service.getDownloadUrl('tenant-uuid', 'report-uuid')).rejects.toThrow(common_1.InternalServerErrorException);
            await expect(service.getDownloadUrl('tenant-uuid', 'report-uuid')).rejects.toThrow('Report is not ready for download');
        });
        it('should throw error if report has no s3Key', async () => {
            const reportNoKey = { ...mockCompletedReport, s3Key: null };
            repository.findOne.mockResolvedValue(reportNoKey);
            await expect(service.getDownloadUrl('tenant-uuid', 'completed-report-uuid')).rejects.toThrow(common_1.InternalServerErrorException);
        });
    });
    describe('remove', () => {
        it('should remove a report', async () => {
            repository.findOne.mockResolvedValue(mockReport);
            repository.remove.mockResolvedValue(mockReport);
            await service.remove('tenant-uuid', 'report-uuid');
            expect(repository.remove).toHaveBeenCalledWith(mockReport);
        });
        it('should throw NotFoundException if report not found', async () => {
            repository.findOne.mockResolvedValue(null);
            await expect(service.remove('tenant-uuid', 'non-existent-uuid')).rejects.toThrow(common_1.NotFoundException);
        });
    });
    describe('regenerate', () => {
        it('should reset report status to pending', async () => {
            const completedReport = { ...mockCompletedReport };
            repository.findOne.mockResolvedValue(completedReport);
            repository.save.mockResolvedValue({
                ...completedReport,
                status: report_entity_1.ReportStatus.PENDING,
                errorMessage: null,
            });
            const result = await service.regenerate('tenant-uuid', 'completed-report-uuid', 'user-uuid');
            expect(repository.save).toHaveBeenCalledWith(expect.objectContaining({
                status: report_entity_1.ReportStatus.PENDING,
                errorMessage: null,
            }));
            expect(result.status).toBe(report_entity_1.ReportStatus.PENDING);
        });
        it('should throw NotFoundException if report not found', async () => {
            repository.findOne.mockResolvedValue(null);
            await expect(service.regenerate('tenant-uuid', 'non-existent-uuid', 'user-uuid')).rejects.toThrow(common_1.NotFoundException);
        });
    });
});
//# sourceMappingURL=reports.service.spec.js.map
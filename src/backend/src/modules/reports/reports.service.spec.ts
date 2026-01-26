import { Test, TestingModule } from '@nestjs/testing';
import { getRepositoryToken } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { NotFoundException, InternalServerErrorException } from '@nestjs/common';
import { ReportsService } from './reports.service';
import { Report, ReportType, ReportStatus } from './entities/report.entity';
import { CreateReportDto } from './dto';

// Mock AWS SDK
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
    let service: ReportsService;
    let repository: jest.Mocked<Repository<Report>>;

    const mockReport: Partial<Report> = {
        id: 'report-uuid',
        tenantId: 'tenant-uuid',
        vehicleId: 'vehicle-uuid',
        customerId: 'customer-uuid',
        reportType: ReportType.DIAGNOSTIC,
        status: ReportStatus.PENDING,
        title: 'Diagnostic Report',
        description: 'Full diagnostic analysis',
        s3Key: null,
        s3Bucket: null,
        createdAt: new Date(),
        updatedAt: new Date(),
    };

    const mockCompletedReport: Partial<Report> = {
        ...mockReport,
        id: 'completed-report-uuid',
        status: ReportStatus.COMPLETED,
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

        const module: TestingModule = await Test.createTestingModule({
            providers: [
                ReportsService,
                {
                    provide: getRepositoryToken(Report),
                    useValue: mockRepository,
                },
            ],
        }).compile();

        service = module.get<ReportsService>(ReportsService);
        repository = module.get(getRepositoryToken(Report));

        jest.clearAllMocks();
    });

    describe('create', () => {
        it('should create a report with pending status', async () => {
            const dto: CreateReportDto = {
                vehicleId: 'vehicle-uuid',
                reportType: ReportType.DIAGNOSTIC,
                title: 'Diagnostic Report',
                description: 'Full diagnostic analysis',
            };

            repository.create.mockReturnValue(mockReport as Report);
            repository.save.mockResolvedValue(mockReport as Report);

            const result = await service.create('tenant-uuid', dto, 'user-uuid');

            expect(repository.create).toHaveBeenCalledWith({
                ...dto,
                tenantId: 'tenant-uuid',
                status: ReportStatus.PENDING,
            });
            expect(result.status).toBe(ReportStatus.PENDING);
            expect(result.title).toBe('Diagnostic Report');
        });

        it('should create a tuning report', async () => {
            const dto: CreateReportDto = {
                vehicleId: 'vehicle-uuid',
                reportType: ReportType.TUNING,
                title: 'Stage 2 Tuning Report',
                data: {
                    beforePower: 300,
                    afterPower: 380,
                    modifications: ['ECU Remap', 'Downpipe'],
                },
            };

            const tuningReport = {
                ...mockReport,
                reportType: ReportType.TUNING,
                title: 'Stage 2 Tuning Report',
                data: dto.data,
            };

            repository.create.mockReturnValue(tuningReport as Report);
            repository.save.mockResolvedValue(tuningReport as Report);

            const result = await service.create('tenant-uuid', dto, 'user-uuid');

            expect(result.reportType).toBe(ReportType.TUNING);
        });
    });

    describe('findAll', () => {
        it('should return paginated reports', async () => {
            mockQueryBuilder.getManyAndCount.mockResolvedValue([[mockReport as Report], 1]);

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
            mockQueryBuilder.getManyAndCount.mockResolvedValue([[mockReport as Report], 1]);

            await service.findAll({
                tenantId: 'tenant-uuid',
                vehicleId: 'vehicle-uuid',
            });

            expect(mockQueryBuilder.andWhere).toHaveBeenCalledWith(
                'report.vehicleId = :vehicleId',
                { vehicleId: 'vehicle-uuid' },
            );
        });

        it('should filter by reportType', async () => {
            mockQueryBuilder.getManyAndCount.mockResolvedValue([[mockReport as Report], 1]);

            await service.findAll({
                tenantId: 'tenant-uuid',
                reportType: ReportType.DIAGNOSTIC,
            });

            expect(mockQueryBuilder.andWhere).toHaveBeenCalledWith(
                'report.reportType = :reportType',
                { reportType: ReportType.DIAGNOSTIC },
            );
        });

        it('should filter by status', async () => {
            mockQueryBuilder.getManyAndCount.mockResolvedValue([[mockCompletedReport as Report], 1]);

            await service.findAll({
                tenantId: 'tenant-uuid',
                status: ReportStatus.COMPLETED,
            });

            expect(mockQueryBuilder.andWhere).toHaveBeenCalledWith(
                'report.status = :status',
                { status: ReportStatus.COMPLETED },
            );
        });

        it('should filter by customerId', async () => {
            mockQueryBuilder.getManyAndCount.mockResolvedValue([[mockReport as Report], 1]);

            await service.findAll({
                tenantId: 'tenant-uuid',
                customerId: 'customer-uuid',
            });

            expect(mockQueryBuilder.andWhere).toHaveBeenCalledWith(
                'report.customerId = :customerId',
                { customerId: 'customer-uuid' },
            );
        });

        it('should handle pagination correctly', async () => {
            mockQueryBuilder.getManyAndCount.mockResolvedValue([[mockReport as Report], 50]);

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
            repository.findOne.mockResolvedValue(mockReport as Report);

            const result = await service.findOne('tenant-uuid', 'report-uuid');

            expect(repository.findOne).toHaveBeenCalledWith({
                where: { id: 'report-uuid', tenantId: 'tenant-uuid' },
            });
            expect(result.id).toBe('report-uuid');
        });

        it('should throw NotFoundException if report not found', async () => {
            repository.findOne.mockResolvedValue(null);

            await expect(
                service.findOne('tenant-uuid', 'non-existent-uuid'),
            ).rejects.toThrow(NotFoundException);
            await expect(
                service.findOne('tenant-uuid', 'non-existent-uuid'),
            ).rejects.toThrow('Report with ID non-existent-uuid not found');
        });
    });

    describe('getDownloadUrl', () => {
        it('should return signed URL for completed report', async () => {
            repository.findOne.mockResolvedValue(mockCompletedReport as Report);

            const result = await service.getDownloadUrl('tenant-uuid', 'completed-report-uuid');

            expect(result).toBe('https://s3.amazonaws.com/signed-url');
        });

        it('should throw error if report is not completed', async () => {
            repository.findOne.mockResolvedValue(mockReport as Report);

            await expect(
                service.getDownloadUrl('tenant-uuid', 'report-uuid'),
            ).rejects.toThrow(InternalServerErrorException);
            await expect(
                service.getDownloadUrl('tenant-uuid', 'report-uuid'),
            ).rejects.toThrow('Report is not ready for download');
        });

        it('should throw error if report has no s3Key', async () => {
            const reportNoKey = { ...mockCompletedReport, s3Key: null };
            repository.findOne.mockResolvedValue(reportNoKey as Report);

            await expect(
                service.getDownloadUrl('tenant-uuid', 'completed-report-uuid'),
            ).rejects.toThrow(InternalServerErrorException);
        });
    });

    describe('remove', () => {
        it('should remove a report', async () => {
            repository.findOne.mockResolvedValue(mockReport as Report);
            repository.remove.mockResolvedValue(mockReport as Report);

            await service.remove('tenant-uuid', 'report-uuid');

            expect(repository.remove).toHaveBeenCalledWith(mockReport);
        });

        it('should throw NotFoundException if report not found', async () => {
            repository.findOne.mockResolvedValue(null);

            await expect(
                service.remove('tenant-uuid', 'non-existent-uuid'),
            ).rejects.toThrow(NotFoundException);
        });
    });

    describe('regenerate', () => {
        it('should reset report status to pending', async () => {
            const completedReport = { ...mockCompletedReport };
            repository.findOne.mockResolvedValue(completedReport as Report);
            repository.save.mockResolvedValue({
                ...completedReport,
                status: ReportStatus.PENDING,
                errorMessage: null,
            } as Report);

            const result = await service.regenerate('tenant-uuid', 'completed-report-uuid', 'user-uuid');

            expect(repository.save).toHaveBeenCalledWith(
                expect.objectContaining({
                    status: ReportStatus.PENDING,
                    errorMessage: null,
                }),
            );
            expect(result.status).toBe(ReportStatus.PENDING);
        });

        it('should throw NotFoundException if report not found', async () => {
            repository.findOne.mockResolvedValue(null);

            await expect(
                service.regenerate('tenant-uuid', 'non-existent-uuid', 'user-uuid'),
            ).rejects.toThrow(NotFoundException);
        });
    });
});

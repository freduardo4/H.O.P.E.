import { Test, TestingModule } from '@nestjs/testing';
import { ReportsController } from './reports.controller';
import { ReportsService, PaginatedReports } from './reports.service';
import { Report, ReportType, ReportStatus } from './entities/report.entity';
import { User, UserRole } from '../auth/entities/user.entity';

describe('ReportsController', () => {
    let controller: ReportsController;
    let service: jest.Mocked<ReportsService>;

    const mockUser: Partial<User> = {
        id: 'user-uuid',
        tenantId: 'tenant-uuid',
        email: 'test@example.com',
        role: UserRole.TECHNICIAN,
    };

    const mockReport: Partial<Report> = {
        id: 'report-uuid',
        tenantId: 'tenant-uuid',
        vehicleId: 'vehicle-uuid',
        customerId: 'customer-uuid',
        reportType: ReportType.DIAGNOSTIC,
        status: ReportStatus.COMPLETED,
        title: 'Diagnostic Report',
        s3Key: 'tenant-uuid/reports/report.pdf',
        s3Bucket: 'hope-reports',
        generatedBy: 'user-uuid',
        createdAt: new Date(),
        updatedAt: new Date(),
    };

    const mockPaginatedResult: PaginatedReports = {
        data: [mockReport as Report],
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

        const module: TestingModule = await Test.createTestingModule({
            controllers: [ReportsController],
            providers: [
                {
                    provide: ReportsService,
                    useValue: mockService,
                },
            ],
        }).compile();

        controller = module.get<ReportsController>(ReportsController);
        service = module.get(ReportsService);

        jest.clearAllMocks();
    });

    describe('create', () => {
        it('should create a new report', async () => {
            const dto = {
                vehicleId: 'vehicle-uuid',
                customerId: 'customer-uuid',
                reportType: ReportType.DIAGNOSTIC,
                title: 'Diagnostic Report',
            };

            service.create.mockResolvedValue(mockReport as Report);

            const result = await controller.create(mockUser as User, dto);

            expect(service.create).toHaveBeenCalledWith(
                'tenant-uuid',
                dto,
                'user-uuid',
            );
            expect(result.id).toBe('report-uuid');
        });
    });

    describe('findAll', () => {
        it('should return paginated reports', async () => {
            service.findAll.mockResolvedValue(mockPaginatedResult);

            const result = await controller.findAll(mockUser as User);

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

            await controller.findAll(mockUser as User, 'vehicle-uuid');

            expect(service.findAll).toHaveBeenCalledWith(
                expect.objectContaining({ vehicleId: 'vehicle-uuid' }),
            );
        });

        it('should filter by report type', async () => {
            service.findAll.mockResolvedValue(mockPaginatedResult);

            await controller.findAll(
                mockUser as User,
                undefined,
                undefined,
                ReportType.TUNING,
            );

            expect(service.findAll).toHaveBeenCalledWith(
                expect.objectContaining({ reportType: ReportType.TUNING }),
            );
        });

        it('should filter by status', async () => {
            service.findAll.mockResolvedValue(mockPaginatedResult);

            await controller.findAll(
                mockUser as User,
                undefined,
                undefined,
                undefined,
                ReportStatus.PENDING,
            );

            expect(service.findAll).toHaveBeenCalledWith(
                expect.objectContaining({ status: ReportStatus.PENDING }),
            );
        });
    });

    describe('findOne', () => {
        it('should return a single report', async () => {
            service.findOne.mockResolvedValue(mockReport as Report);

            const result = await controller.findOne(mockUser as User, 'report-uuid');

            expect(service.findOne).toHaveBeenCalledWith('tenant-uuid', 'report-uuid');
            expect(result.id).toBe('report-uuid');
        });
    });

    describe('getDownloadUrl', () => {
        it('should return download URL with expiry', async () => {
            const downloadUrl = 'https://s3.amazonaws.com/bucket/report.pdf?signed=true';
            service.getDownloadUrl.mockResolvedValue(downloadUrl);

            const result = await controller.getDownloadUrl(mockUser as User, 'report-uuid');

            expect(service.getDownloadUrl).toHaveBeenCalledWith('tenant-uuid', 'report-uuid');
            expect(result.url).toBe(downloadUrl);
            expect(result.expiresIn).toBe(3600);
        });
    });

    describe('regenerate', () => {
        it('should regenerate a report', async () => {
            const regeneratedReport = { ...mockReport, status: ReportStatus.PENDING };
            service.regenerate.mockResolvedValue(regeneratedReport as Report);

            const result = await controller.regenerate(mockUser as User, 'report-uuid');

            expect(service.regenerate).toHaveBeenCalledWith(
                'tenant-uuid',
                'report-uuid',
                'user-uuid',
            );
            expect(result.status).toBe(ReportStatus.PENDING);
        });
    });

    describe('remove', () => {
        it('should delete a report and return success message', async () => {
            service.remove.mockResolvedValue(undefined);

            const result = await controller.remove(mockUser as User, 'report-uuid');

            expect(service.remove).toHaveBeenCalledWith('tenant-uuid', 'report-uuid');
            expect(result.message).toBe('Report deleted successfully');
        });
    });
});

import { ReportsService, PaginatedReports } from './reports.service';
import { CreateReportDto } from './dto';
import { Report, ReportType, ReportStatus } from './entities/report.entity';
import { User } from '../auth/entities/user.entity';
export declare class ReportsController {
    private readonly reportsService;
    constructor(reportsService: ReportsService);
    create(user: User, dto: CreateReportDto): Promise<Report>;
    findAll(user: User, vehicleId?: string, customerId?: string, reportType?: ReportType, status?: ReportStatus, page?: number, limit?: number): Promise<PaginatedReports>;
    findOne(user: User, id: string): Promise<Report>;
    getDownloadUrl(user: User, id: string): Promise<{
        url: string;
        expiresIn: number;
    }>;
    regenerate(user: User, id: string): Promise<Report>;
    remove(user: User, id: string): Promise<{
        message: string;
    }>;
}

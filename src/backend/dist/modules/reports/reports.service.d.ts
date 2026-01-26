import { Repository } from 'typeorm';
import { Report, ReportType, ReportStatus } from './entities/report.entity';
import { CreateReportDto } from './dto';
export interface PaginatedReports {
    data: Report[];
    total: number;
    page: number;
    limit: number;
    totalPages: number;
}
export declare class ReportsService {
    private readonly reportRepo;
    private s3Client;
    private bucketName;
    constructor(reportRepo: Repository<Report>);
    create(tenantId: string, dto: CreateReportDto, generatedBy: string): Promise<Report>;
    private generatePDF;
    private createSimplePDF;
    findAll(params: {
        tenantId: string;
        vehicleId?: string;
        customerId?: string;
        reportType?: ReportType;
        status?: ReportStatus;
        page?: number;
        limit?: number;
    }): Promise<PaginatedReports>;
    findOne(tenantId: string, id: string): Promise<Report>;
    getDownloadUrl(tenantId: string, id: string, expiresIn?: number): Promise<string>;
    remove(tenantId: string, id: string): Promise<void>;
    regenerate(tenantId: string, id: string, generatedBy: string): Promise<Report>;
}

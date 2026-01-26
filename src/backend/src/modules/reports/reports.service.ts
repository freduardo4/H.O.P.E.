import { Injectable, NotFoundException, InternalServerErrorException } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { S3Client, PutObjectCommand, GetObjectCommand } from '@aws-sdk/client-s3';
import { getSignedUrl } from '@aws-sdk/s3-request-presigner';
import { Report, ReportType, ReportStatus } from './entities/report.entity';
import { CreateReportDto } from './dto';

export interface PaginatedReports {
    data: Report[];
    total: number;
    page: number;
    limit: number;
    totalPages: number;
}

@Injectable()
export class ReportsService {
    private s3Client: S3Client;
    private bucketName: string;

    constructor(
        @InjectRepository(Report)
        private readonly reportRepo: Repository<Report>,
    ) {
        this.bucketName = process.env.AWS_S3_BUCKET || 'hope-reports';
        this.s3Client = new S3Client({
            region: process.env.AWS_S3_REGION || 'us-east-1',
        });
    }

    async create(tenantId: string, dto: CreateReportDto, generatedBy: string): Promise<Report> {
        const report = this.reportRepo.create({
            ...dto,
            tenantId,
            status: ReportStatus.PENDING,
        });

        const savedReport = await this.reportRepo.save(report);

        // Trigger async PDF generation (would be done in a queue in production)
        this.generatePDF(savedReport.id, tenantId, generatedBy).catch(error => {
            console.error(`Failed to generate PDF for report ${savedReport.id}:`, error);
        });

        return savedReport;
    }

    private async generatePDF(reportId: string, tenantId: string, generatedBy: string): Promise<void> {
        try {
            // Update status to generating
            await this.reportRepo.update(reportId, {
                status: ReportStatus.GENERATING,
            });

            const report = await this.findOne(tenantId, reportId);

            // Generate PDF content (simplified placeholder)
            // In production, use a library like PDFKit, Puppeteer, or QuestPDF via C# service
            const pdfContent = this.createSimplePDF(report);
            const pdfBuffer = Buffer.from(pdfContent);

            // Upload to S3
            const timestamp = Date.now();
            const s3Key = `${tenantId}/reports/${report.vehicleId}/${timestamp}-${report.title.replace(/\s+/g, '-')}.pdf`;

            await this.s3Client.send(
                new PutObjectCommand({
                    Bucket: this.bucketName,
                    Key: s3Key,
                    Body: pdfBuffer,
                    ContentType: 'application/pdf',
                }),
            );

            // Update report with S3 info
            await this.reportRepo.update(reportId, {
                status: ReportStatus.COMPLETED,
                s3Key,
                s3Bucket: this.bucketName,
                fileSize: pdfBuffer.length,
                generatedBy,
                generatedAt: new Date(),
            });
        } catch (error) {
            await this.reportRepo.update(reportId, {
                status: ReportStatus.FAILED,
                errorMessage: error.message,
            });
            throw error;
        }
    }

    private createSimplePDF(report: Report): string {
        // Simplified PDF generation - placeholder for actual PDF library
        // In production, this would use a proper PDF generation library
        return `
%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /Resources 4 0 R /MediaBox [0 0 612 792] /Contents 5 0 R >>
endobj
4 0 obj
<< /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >>
endobj
5 0 obj
<< /Length 200 >>
stream
BT
/F1 24 Tf
50 700 Td
(${report.title}) Tj
/F1 12 Tf
50 660 Td
(Report Type: ${report.reportType}) Tj
50 640 Td
(Generated: ${new Date().toISOString()}) Tj
ET
endstream
endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000214 00000 n
0000000304 00000 n
trailer
<< /Size 6 /Root 1 0 R >>
startxref
555
%%EOF
        `.trim();
    }

    async findAll(params: {
        tenantId: string;
        vehicleId?: string;
        customerId?: string;
        reportType?: ReportType;
        status?: ReportStatus;
        page?: number;
        limit?: number;
    }): Promise<PaginatedReports> {
        const { tenantId, vehicleId, customerId, reportType, status, page = 1, limit = 20 } = params;

        const query = this.reportRepo.createQueryBuilder('report')
            .where('report.tenantId = :tenantId', { tenantId });

        if (vehicleId) {
            query.andWhere('report.vehicleId = :vehicleId', { vehicleId });
        }

        if (customerId) {
            query.andWhere('report.customerId = :customerId', { customerId });
        }

        if (reportType) {
            query.andWhere('report.reportType = :reportType', { reportType });
        }

        if (status) {
            query.andWhere('report.status = :status', { status });
        }

        const [data, total] = await query
            .orderBy('report.createdAt', 'DESC')
            .skip((page - 1) * limit)
            .take(limit)
            .getManyAndCount();

        return {
            data,
            total,
            page,
            limit,
            totalPages: Math.ceil(total / limit),
        };
    }

    async findOne(tenantId: string, id: string): Promise<Report> {
        const report = await this.reportRepo.findOne({
            where: { id, tenantId },
        });

        if (!report) {
            throw new NotFoundException(`Report with ID ${id} not found`);
        }

        return report;
    }

    async getDownloadUrl(tenantId: string, id: string, expiresIn = 3600): Promise<string> {
        const report = await this.findOne(tenantId, id);

        if (report.status !== ReportStatus.COMPLETED) {
            throw new InternalServerErrorException('Report is not ready for download');
        }

        if (!report.s3Key) {
            throw new InternalServerErrorException('Report file not found');
        }

        const command = new GetObjectCommand({
            Bucket: report.s3Bucket,
            Key: report.s3Key,
        });

        return await getSignedUrl(this.s3Client, command, { expiresIn });
    }

    async remove(tenantId: string, id: string): Promise<void> {
        const report = await this.findOne(tenantId, id);
        await this.reportRepo.remove(report);
    }

    async regenerate(tenantId: string, id: string, generatedBy: string): Promise<Report> {
        const report = await this.findOne(tenantId, id);

        // Reset status
        report.status = ReportStatus.PENDING;
        report.errorMessage = null;
        await this.reportRepo.save(report);

        // Trigger regeneration
        this.generatePDF(report.id, tenantId, generatedBy).catch(error => {
            console.error(`Failed to regenerate PDF for report ${report.id}:`, error);
        });

        return report;
    }
}

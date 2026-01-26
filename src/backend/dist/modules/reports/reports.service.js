"use strict";
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
var __metadata = (this && this.__metadata) || function (k, v) {
    if (typeof Reflect === "object" && typeof Reflect.metadata === "function") return Reflect.metadata(k, v);
};
var __param = (this && this.__param) || function (paramIndex, decorator) {
    return function (target, key) { decorator(target, key, paramIndex); }
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.ReportsService = void 0;
const common_1 = require("@nestjs/common");
const typeorm_1 = require("@nestjs/typeorm");
const typeorm_2 = require("typeorm");
const client_s3_1 = require("@aws-sdk/client-s3");
const s3_request_presigner_1 = require("@aws-sdk/s3-request-presigner");
const report_entity_1 = require("./entities/report.entity");
let ReportsService = class ReportsService {
    constructor(reportRepo) {
        this.reportRepo = reportRepo;
        this.bucketName = process.env.AWS_S3_BUCKET || 'hope-reports';
        this.s3Client = new client_s3_1.S3Client({
            region: process.env.AWS_S3_REGION || 'us-east-1',
        });
    }
    async create(tenantId, dto, generatedBy) {
        const report = this.reportRepo.create({
            ...dto,
            tenantId,
            status: report_entity_1.ReportStatus.PENDING,
        });
        const savedReport = await this.reportRepo.save(report);
        this.generatePDF(savedReport.id, tenantId, generatedBy).catch(error => {
            console.error(`Failed to generate PDF for report ${savedReport.id}:`, error);
        });
        return savedReport;
    }
    async generatePDF(reportId, tenantId, generatedBy) {
        try {
            await this.reportRepo.update(reportId, {
                status: report_entity_1.ReportStatus.GENERATING,
            });
            const report = await this.findOne(tenantId, reportId);
            const pdfContent = this.createSimplePDF(report);
            const pdfBuffer = Buffer.from(pdfContent);
            const timestamp = Date.now();
            const s3Key = `${tenantId}/reports/${report.vehicleId}/${timestamp}-${report.title.replace(/\s+/g, '-')}.pdf`;
            await this.s3Client.send(new client_s3_1.PutObjectCommand({
                Bucket: this.bucketName,
                Key: s3Key,
                Body: pdfBuffer,
                ContentType: 'application/pdf',
            }));
            await this.reportRepo.update(reportId, {
                status: report_entity_1.ReportStatus.COMPLETED,
                s3Key,
                s3Bucket: this.bucketName,
                fileSize: pdfBuffer.length,
                generatedBy,
                generatedAt: new Date(),
            });
        }
        catch (error) {
            await this.reportRepo.update(reportId, {
                status: report_entity_1.ReportStatus.FAILED,
                errorMessage: error.message,
            });
            throw error;
        }
    }
    createSimplePDF(report) {
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
    async findAll(params) {
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
    async findOne(tenantId, id) {
        const report = await this.reportRepo.findOne({
            where: { id, tenantId },
        });
        if (!report) {
            throw new common_1.NotFoundException(`Report with ID ${id} not found`);
        }
        return report;
    }
    async getDownloadUrl(tenantId, id, expiresIn = 3600) {
        const report = await this.findOne(tenantId, id);
        if (report.status !== report_entity_1.ReportStatus.COMPLETED) {
            throw new common_1.InternalServerErrorException('Report is not ready for download');
        }
        if (!report.s3Key) {
            throw new common_1.InternalServerErrorException('Report file not found');
        }
        const command = new client_s3_1.GetObjectCommand({
            Bucket: report.s3Bucket,
            Key: report.s3Key,
        });
        return await (0, s3_request_presigner_1.getSignedUrl)(this.s3Client, command, { expiresIn });
    }
    async remove(tenantId, id) {
        const report = await this.findOne(tenantId, id);
        await this.reportRepo.remove(report);
    }
    async regenerate(tenantId, id, generatedBy) {
        const report = await this.findOne(tenantId, id);
        report.status = report_entity_1.ReportStatus.PENDING;
        report.errorMessage = null;
        await this.reportRepo.save(report);
        this.generatePDF(report.id, tenantId, generatedBy).catch(error => {
            console.error(`Failed to regenerate PDF for report ${report.id}:`, error);
        });
        return report;
    }
};
exports.ReportsService = ReportsService;
exports.ReportsService = ReportsService = __decorate([
    (0, common_1.Injectable)(),
    __param(0, (0, typeorm_1.InjectRepository)(report_entity_1.Report)),
    __metadata("design:paramtypes", [typeorm_2.Repository])
], ReportsService);
//# sourceMappingURL=reports.service.js.map
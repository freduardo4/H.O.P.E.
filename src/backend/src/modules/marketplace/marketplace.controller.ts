import { Controller, Get, Param, Res, StreamableFile, NotFoundException, UnauthorizedException, Query } from '@nestjs/common';
import { MarketplaceService } from './marketplace.service';
import { Response } from 'express';
import { createReadStream, existsSync } from 'fs';
import { join } from 'path';

@Controller('marketplace')
export class MarketplaceController {
    constructor(private readonly marketplaceService: MarketplaceService) { }

    @Get('listings')
    async findAllListings() {
        return this.marketplaceService.findAllListings();
    }

    @Get('download/:licenseKey')
    async downloadFile(
        @Param('licenseKey') licenseKey: string,
        @Query('hardwareId') hardwareId: string,
        @Res({ passthrough: true }) res: Response
    ): Promise<StreamableFile> {
        if (!hardwareId) {
            throw new UnauthorizedException('Hardware ID is required');
        }

        // 1. Validate License
        const isValid = await this.marketplaceService.validateLicense(hardwareId, licenseKey);
        if (!isValid) {
            throw new UnauthorizedException('Invalid or inactive license');
        }

        // 2. Get Listing Info
        const listing = await this.marketplaceService.findListingByLicense(licenseKey);
        if (!listing) {
            throw new NotFoundException('Listing not found');
        }

        // 3. Serve File (Mocking Storage)
        // In a real app, this would stream from S3 or use a pre-signed URL.
        // For now, we'll serve a dummy file if the path doesn't exist, or the actual file.

        // We'll simulate a file path. Ideally, listing.fileUrl points to a local file for now or an S3 key.
        // Let's assume listing.fileUrl is just a filename.
        const fileName = listing.fileUrl || 'default_calibration.bin';
        const uploadsDir = join(process.cwd(), 'uploads');
        const filePath = join(uploadsDir, fileName);

        if (!existsSync(filePath)) {
            // GENERATE DUMMY CONTENT IF FILE MISSING (For Dev Convenience)
            return new StreamableFile(Buffer.from(`DUMMY ENCRYPTED CONTENT FOR ${listing.title} v${listing.version}\nHardwareID: ${hardwareId}`));
        }

        const file = createReadStream(filePath);
        res.set({
            'Content-Type': 'application/octet-stream',
            'Content-Disposition': `attachment; filename="${fileName}"`,
        });
        return new StreamableFile(file);
    }
}

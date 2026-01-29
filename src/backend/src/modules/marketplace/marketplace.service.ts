import { Injectable, NotFoundException } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { CalibrationListing } from './entities/calibration-listing.entity';
import { License } from './entities/license.entity';
import { User } from '../auth/entities/user.entity';
import { Review } from './entities/review.entity';
import * as crypto from 'crypto';

@Injectable()
export class MarketplaceService {
    constructor(
        @InjectRepository(CalibrationListing)
        private listingRepository: Repository<CalibrationListing>,
        @InjectRepository(License)
        private licenseRepository: Repository<License>,
        @InjectRepository(Review)
        private reviewRepository: Repository<Review>,
    ) { }

    async findAllListings(): Promise<any[]> {
        const listings = await this.listingRepository.find({ relations: ['reviews'] });
        return listings.map(l => {
            const avgRating = l.reviews && l.reviews.length > 0
                ? l.reviews.reduce((acc, r) => acc + r.rating, 0) / l.reviews.length
                : 0;
            return {
                ...l,
                rating: parseFloat(avgRating.toFixed(1)),
                reviewCount: l.reviews?.length || 0
            };
        });
    }

    async findListingById(id: string): Promise<CalibrationListing> {
        const listing = await this.listingRepository.findOne({ where: { id } });
        if (!listing) throw new NotFoundException('Listing not found');
        return listing;
    }

    async createLicense(user: User, listingId: string, hardwareId: string): Promise<License> {
        const listing = await this.findListingById(listingId);

        // License key generation logic
        const licenseKey = crypto.randomBytes(16).toString('hex').toUpperCase();

        const license = this.licenseRepository.create({
            licenseKey,
            hardwareId,
            listing,
            user,
        });

        return this.licenseRepository.save(license);
    }

    async validateLicense(hardwareId: string, licenseKey: string): Promise<boolean> {
        const license = await this.licenseRepository.findOne({
            where: { hardwareId, licenseKey, isActive: true },
            relations: ['listing']
        });
        return !!license;
    }

    async findListingByLicense(licenseKey: string): Promise<CalibrationListing | null> {
        const license = await this.licenseRepository.findOne({
            where: { licenseKey },
            relations: ['listing']
        });
        return license ? license.listing : null;
    }

    async encryptBinary(data: Buffer, secretKey: string): Promise<Buffer> {
        const iv = crypto.randomBytes(12); // GCM standard IV length is 12 bytes
        const cipher = crypto.createCipheriv('aes-256-gcm', Buffer.from(secretKey, 'hex'), iv);
        const encrypted = Buffer.concat([cipher.update(data), cipher.final()]);
        const tag = cipher.getAuthTag();
        // Format: IV (12) + Tag (16) + Encrypted Data
        return Buffer.concat([iv, tag, encrypted]);
    }

    async decryptBinary(encryptedData: Buffer, secretKey: string): Promise<Buffer> {
        const iv = encryptedData.subarray(0, 12);
        const tag = encryptedData.subarray(12, 28);
        const data = encryptedData.subarray(28);

        const decipher = crypto.createDecipheriv('aes-256-gcm', Buffer.from(secretKey, 'hex'), iv);
        decipher.setAuthTag(tag);
        return Buffer.concat([decipher.update(data), decipher.final()]);
    }
}

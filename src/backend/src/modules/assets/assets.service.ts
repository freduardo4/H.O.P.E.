
import { Injectable } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { License } from '../marketplace/entities/license.entity';

@Injectable()
export class AssetsService {
    constructor(
        // We'll inject License repository to get real stats if available, 
        // otherwise we fallback to mock/simulated data if the module isn't fully linked yet.
        // For strict modularity, we might just query the DB directly or use a shared service.
        // To avoid circular dependency with MarketplaceModule, we'll try to use InjectRepository if MarketplaceModule is imported.
        @InjectRepository(License)
        private readonly licenseRepository: Repository<License>,
    ) { }

    getLatestUpdate() {
        return {
            version: '1.2.0',
            url: '/assets/download/hope-setup-1.2.0.exe',
            releaseNotes: 'Includes Wiki-Fix Knowledge Graph and Calibration Marketplace enhancements.',
            releasedAt: new Date().toISOString()
        };
    }

    async getFleetStats() {
        const activeLicenses = await this.licenseRepository.count({ where: { isActive: true } });
        // Simulating "online devices" as a fraction of active licenses for now
        const onlineDevices = Math.floor(activeLicenses * 0.4);

        return {
            activeLicenses,
            onlineDevices,
            totalDownloads: activeLicenses * 2 + 150 // Mock metric
        };
    }
}

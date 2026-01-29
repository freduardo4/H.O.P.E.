
import { Controller, Get } from '@nestjs/common';
import { AssetsService } from './assets.service';

@Controller('assets')
export class AssetsController {
    constructor(private readonly assetsService: AssetsService) { }

    @Get('updates/latest')
    getLatestUpdate() {
        return this.assetsService.getLatestUpdate();
    }

    @Get('fleet/stats')
    async getFleetStats() {
        return this.assetsService.getFleetStats();
    }
}


import { Test, TestingModule } from '@nestjs/testing';
import { AssetsController } from './assets.controller';
import { AssetsService } from './assets.service';

describe('AssetsController', () => {
    let controller: AssetsController;
    let service: AssetsService;

    const mockAssetsService = {
        getLatestUpdate: jest.fn().mockReturnValue({ version: '1.2.0' }),
        getFleetStats: jest.fn().mockResolvedValue({ activeLicenses: 50, onlineDevices: 20 }),
    };

    beforeEach(async () => {
        const module: TestingModule = await Test.createTestingModule({
            controllers: [AssetsController],
            providers: [
                { provide: AssetsService, useValue: mockAssetsService },
            ],
        }).compile();

        controller = module.get<AssetsController>(AssetsController);
        service = module.get<AssetsService>(AssetsService);
    });

    it('should be defined', () => {
        expect(controller).toBeDefined();
    });

    it('should return latest update info', () => {
        expect(controller.getLatestUpdate()).toEqual({ version: '1.2.0' });
    });

    it('should return fleet stats', async () => {
        const stats = await controller.getFleetStats();
        expect(stats).toEqual({ activeLicenses: 50, onlineDevices: 20 });
    });
});

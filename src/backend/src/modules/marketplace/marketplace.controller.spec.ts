jest.mock('./marketplace.service', () => ({
    MarketplaceService: class MockMarketplaceService { }
}));

import { Test, TestingModule } from '@nestjs/testing';
import { MarketplaceController } from './marketplace.controller';


import { MarketplaceService } from './marketplace.service';
import { UnauthorizedException, NotFoundException, StreamableFile } from '@nestjs/common';
import * as fs from 'fs';

// Mock fs to avoid actual file I/O
jest.mock('fs', () => ({
    existsSync: jest.fn(),
    createReadStream: jest.fn(),
}));

describe('MarketplaceController', () => {
    let controller: MarketplaceController;
    let service: MarketplaceService;

    const mockMarketplaceService = {
        validateLicense: jest.fn(),
        findListingByLicense: jest.fn(),
    };

    beforeEach(async () => {
        const module: TestingModule = await Test.createTestingModule({
            controllers: [MarketplaceController],
            providers: [
                {
                    provide: MarketplaceService,
                    useValue: mockMarketplaceService,
                },
            ],
        }).compile();

        controller = module.get<MarketplaceController>(MarketplaceController);
        service = module.get<MarketplaceService>(MarketplaceService);
    });

    it('should be defined', () => {
        expect(controller).toBeDefined();
    });

    describe('downloadFile', () => {
        it('should throw UnauthorizedException if hardwareId is missing', async () => {
            await expect(controller.downloadFile('valid-license', '', {} as any)).rejects.toThrow(UnauthorizedException);
        });

        it('should throw UnauthorizedException if license is invalid', async () => {
            mockMarketplaceService.validateLicense.mockResolvedValue(false);
            await expect(controller.downloadFile('invalid-license', 'hw-id', {} as any)).rejects.toThrow(UnauthorizedException);
        });

        it('should throw NotFoundException if listing is not found', async () => {
            mockMarketplaceService.validateLicense.mockResolvedValue(true);
            mockMarketplaceService.findListingByLicense.mockResolvedValue(null);
            await expect(controller.downloadFile('valid-license', 'hw-id', {} as any)).rejects.toThrow(NotFoundException);
        });

        it.skip('should return StreamableFile if validation passes', async () => {
            mockMarketplaceService.validateLicense.mockResolvedValue(true);
            mockMarketplaceService.findListingByLicense.mockResolvedValue({
                title: 'Test Tune',
                version: '1.0',
                fileUrl: 'test.bin',
            });

            // Mock file exists to false to trigger dummy content generation (easier to test)
            (fs.existsSync as jest.Mock).mockReturnValue(false);

            const res = { set: jest.fn() } as any;
            const result = await controller.downloadFile('valid-license', 'hw-id', res);

            expect(result).toBeInstanceOf(StreamableFile);
            expect(res.set).toHaveBeenCalledWith(expect.objectContaining({
                'Content-Type': 'application/octet-stream',
            }));
        });
    });
});

import { Test, TestingModule } from '@nestjs/testing';
import { MarketplaceResolver } from './marketplace.resolver';
import { MarketplaceService } from './marketplace.service';
import { CalibrationListing } from './entities/calibration-listing.entity';
import { License } from './entities/license.entity';
import { UserRole } from '../auth/entities/user.entity';

describe('MarketplaceResolver', () => {
    let resolver: MarketplaceResolver;
    let service: jest.Mocked<MarketplaceService>;

    const mockListing: Partial<CalibrationListing> = {
        id: 'listing-1',
        title: 'Stage 1 Tune',
        description: 'High performance tune',
        price: 150,
        version: '1.0.0',
    };

    const mockLicense: Partial<License> = {
        id: 'license-1',
        licenseKey: 'MOCK-KEY-123',
        hardwareId: 'HW-ID-123',
    };

    const mockUser = {
        id: 'user-1',
        email: 'test@test.com',
        role: UserRole.TECHNICIAN,
        tenantId: 'tenant-1',
    };

    beforeEach(async () => {
        const mockMarketplaceService = {
            findAllListings: jest.fn(),
            findListingById: jest.fn(),
            createLicense: jest.fn(),
            validateLicense: jest.fn(),
        };

        const module: TestingModule = await Test.createTestingModule({
            providers: [
                MarketplaceResolver,
                {
                    provide: MarketplaceService,
                    useValue: mockMarketplaceService,
                },
            ],
        }).compile();

        resolver = module.get<MarketplaceResolver>(MarketplaceResolver);
        service = module.get(MarketplaceService);
    });

    it('should be defined', () => {
        expect(resolver).toBeDefined();
    });

    describe('calibrationListings', () => {
        it('should return all listings', async () => {
            service.findAllListings.mockResolvedValue([mockListing] as CalibrationListing[]);
            const result = await resolver.calibrationListings();
            expect(result).toEqual([mockListing]);
            expect(service.findAllListings).toHaveBeenCalled();
        });
    });

    describe('calibrationListing', () => {
        it('should return a listing by id', async () => {
            service.findListingById.mockResolvedValue(mockListing as CalibrationListing);
            const result = await resolver.calibrationListing('listing-1');
            expect(result).toEqual(mockListing);
            expect(service.findListingById).toHaveBeenCalledWith('listing-1');
        });
    });

    describe('purchaseCalibration', () => {
        it('should create a license for the user', async () => {
            service.createLicense.mockResolvedValue(mockLicense as License);
            const context = { req: { user: mockUser } };
            const result = await resolver.purchaseCalibration('listing-1', 'HW-ID-123', context);
            expect(result).toEqual(mockLicense);
            expect(service.createLicense).toHaveBeenCalledWith(mockUser, 'listing-1', 'HW-ID-123');
        });
    });

    describe('verifyLicense', () => {
        it('should validate a license', async () => {
            service.validateLicense.mockResolvedValue(true);
            const result = await resolver.verifyLicense('MOCK-KEY-123', 'HW-ID-123');
            expect(result).toBe(true);
            expect(service.validateLicense).toHaveBeenCalledWith('HW-ID-123', 'MOCK-KEY-123');
        });
    });
});

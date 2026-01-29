import { Test, TestingModule } from '@nestjs/testing';
import { getRepositoryToken } from '@nestjs/typeorm';
import { MarketplaceService } from './marketplace.service';
import { CalibrationListing } from './entities/calibration-listing.entity';
import { License } from './entities/license.entity';
import { Review } from './entities/review.entity';
import { NotFoundException } from '@nestjs/common';
import { User } from '../auth/entities/user.entity';
import * as crypto from 'crypto';

describe('MarketplaceService', () => {
    let service: MarketplaceService;
    let listingRepository: any;
    let licenseRepository: any;
    let reviewRepository: any;

    const mockListing = {
        id: 'listing-1',
        title: 'Stage 1 Tune',
        description: 'Test description',
        price: 100,
        version: '1.0.0',
        compatibility: 'EDC17',
        reviews: [
            { id: 'rev-1', rating: 5, comment: 'Great!' },
            { id: 'rev-2', rating: 4, comment: 'Good' },
        ],
    };

    const mockUser = {
        id: 'user-1',
        email: 'test@example.com',
    } as User;

    const mockLicense = {
        id: 'license-1',
        licenseKey: 'ABC123XYZ',
        hardwareId: 'HW-999',
        isActive: true,
        listing: mockListing,
        user: mockUser,
    };

    beforeEach(async () => {
        listingRepository = {
            find: jest.fn().mockResolvedValue([mockListing]),
            findOne: jest.fn().mockResolvedValue(mockListing),
        };

        licenseRepository = {
            create: jest.fn().mockImplementation((dto) => ({ ...dto, id: 'license-new' })),
            save: jest.fn().mockImplementation((license) => Promise.resolve(license)),
            findOne: jest.fn().mockResolvedValue(mockLicense),
        };

        reviewRepository = {
            find: jest.fn().mockResolvedValue([]),
        };

        const module: TestingModule = await Test.createTestingModule({
            providers: [
                MarketplaceService,
                {
                    provide: getRepositoryToken(CalibrationListing),
                    useValue: listingRepository,
                },
                {
                    provide: getRepositoryToken(License),
                    useValue: licenseRepository,
                },
                {
                    provide: getRepositoryToken(Review),
                    useValue: reviewRepository,
                },
            ],
        }).compile();

        service = module.get<MarketplaceService>(MarketplaceService);
    });

    it('should be defined', () => {
        expect(service).toBeDefined();
    });

    describe('findAllListings', () => {
        it('should return an array of listings with calculated ratings', async () => {
            const result = await service.findAllListings();
            expect(result[0].rating).toBe(4.5);
            expect(result[0].reviewCount).toBe(2);
            expect(listingRepository.find).toHaveBeenCalledWith({ relations: ['reviews'] });
        });

        it('should return rating 0 for listings with no reviews', async () => {
            listingRepository.find.mockResolvedValue([{ ...mockListing, reviews: [] }]);
            const result = await service.findAllListings();
            expect(result[0].rating).toBe(0);
            expect(result[0].reviewCount).toBe(0);
        });

        it('should handle undefined reviews property gracefully', async () => {
            listingRepository.find.mockResolvedValue([{ ...mockListing, reviews: undefined }]);
            const result = await service.findAllListings();
            expect(result[0].rating).toBe(0);
            expect(result[0].reviewCount).toBe(0);
        });

        it('should round ratings to 1 decimal place', async () => {
            listingRepository.find.mockResolvedValue([{
                ...mockListing,
                reviews: [
                    { rating: 5 }, { rating: 4 }, { rating: 4 }
                ]
            }]); // (5+4+4)/3 = 4.333...
            const result = await service.findAllListings();
            expect(result[0].rating).toBe(4.3);
        });
    });

    describe('findListingById', () => {
        it('should return a listing if found', async () => {
            const result = await service.findListingById('listing-1');
            expect(result).toEqual(mockListing);
        });

        it('should throw NotFoundException if listing not found', async () => {
            listingRepository.findOne.mockResolvedValue(null);
            await expect(service.findListingById('invalid')).rejects.toThrow(NotFoundException);
        });
    });

    describe('createLicense', () => {
        it('should create and save a new license', async () => {
            const result = await service.createLicense(mockUser, 'listing-1', 'HW-999');
            expect(result.hardwareId).toBe('HW-999');
            expect(result.licenseKey).toBeDefined();
            expect(licenseRepository.create).toHaveBeenCalled();
            expect(licenseRepository.save).toHaveBeenCalled();
        });
    });

    describe('validateLicense', () => {
        it('should return true if license is valid', async () => {
            const result = await service.validateLicense('HW-999', 'ABC123XYZ');
            expect(result).toBe(true);
        });

        it('should return false if license not found', async () => {
            licenseRepository.findOne.mockResolvedValue(null);
            const result = await service.validateLicense('HW-999', 'WRONG');
            expect(result).toBe(false);
        });
    });

    describe('Encryption/Decryption', () => {
        const secretKey = '0123456789abcdef0123456789abcdef'; // 32 hex chars = 16 bytes? Wait, AES-256 needs 32 bytes key. 
        // The service uses Buffer.from(secretKey, 'hex'), so secretKey should be 64 hex chars for 32 bytes.
        const longSecretKey = '0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef';
        const originalData = Buffer.from('hello world');

        it('should encrypt and then decrypt data correctly', async () => {
            const encrypted = await service.encryptBinary(originalData, longSecretKey);
            const decrypted = await service.decryptBinary(encrypted, longSecretKey);
            expect(decrypted.toString()).toBe('hello world');
        });

        it('should fail to decrypt with wrong key', async () => {
            const encrypted = await service.encryptBinary(originalData, longSecretKey);
            const wrongKey = 'f'.repeat(64);
            await expect(service.decryptBinary(encrypted, wrongKey)).rejects.toThrow();
        });
    });
});

import { Test, TestingModule } from '@nestjs/testing';
import { Reflector } from '@nestjs/core';
import { JwtAuthGuard } from './jwt-auth.guard';
import { ExecutionContext } from '@nestjs/common';

describe('JwtAuthGuard', () => {
    let guard: JwtAuthGuard;
    let reflector: Reflector;

    const mockReflector = {
        getAllAndOverride: jest.fn(),
    };

    const mockExecutionContext = {
        getHandler: jest.fn(),
        getClass: jest.fn(),
        switchToHttp: jest.fn().mockReturnValue({
            getRequest: jest.fn().mockReturnValue({}),
        }),
    } as unknown as ExecutionContext;

    beforeEach(async () => {
        const module: TestingModule = await Test.createTestingModule({
            providers: [
                JwtAuthGuard,
                { provide: Reflector, useValue: mockReflector },
            ],
        }).compile();

        guard = module.get<JwtAuthGuard>(JwtAuthGuard);
        reflector = module.get<Reflector>(Reflector);
    });

    it('should be defined', () => {
        expect(guard).toBeDefined();
    });

    it('should return true if route is public', () => {
        mockReflector.getAllAndOverride.mockReturnValue(true);
        expect(guard.canActivate(mockExecutionContext)).toBe(true);
    });

    // Testing logic when not public is tricky because super.canActivate() calls Passport strategy.
    // We can at least ensure it checks public status.
});

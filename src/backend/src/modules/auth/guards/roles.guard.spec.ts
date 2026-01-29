import { Test, TestingModule } from '@nestjs/testing';
import { Reflector } from '@nestjs/core';
import { RolesGuard } from './roles.guard';
import { UserRole } from '../entities/user.entity';
import { ExecutionContext } from '@nestjs/common';

describe('RolesGuard', () => {
    let guard: RolesGuard;
    let reflector: Reflector;

    const mockReflector = {
        getAllAndOverride: jest.fn(),
    };

    const mockExecutionContext = {
        getHandler: jest.fn(),
        getClass: jest.fn(),
        switchToHttp: jest.fn().mockReturnValue({
            getRequest: jest.fn().mockReturnValue({
                user: { role: UserRole.TECHNICIAN },
            }),
        }),
    } as unknown as ExecutionContext;

    beforeEach(async () => {
        const module: TestingModule = await Test.createTestingModule({
            providers: [
                RolesGuard,
                { provide: Reflector, useValue: mockReflector },
            ],
        }).compile();

        guard = module.get<RolesGuard>(RolesGuard);
        reflector = module.get<Reflector>(Reflector);
    });

    it('should be defined', () => {
        expect(guard).toBeDefined();
    });

    it('should return true if no roles are required', () => {
        mockReflector.getAllAndOverride.mockReturnValue(undefined);
        expect(guard.canActivate(mockExecutionContext)).toBe(true);
    });

    it('should return true if user has required role', () => {
        mockReflector.getAllAndOverride.mockReturnValue([UserRole.TECHNICIAN]);
        expect(guard.canActivate(mockExecutionContext)).toBe(true);
    });

    it('should return false if user does not have required role', () => {
        mockReflector.getAllAndOverride.mockReturnValue([UserRole.ADMIN]);
        expect(guard.canActivate(mockExecutionContext)).toBe(false);
    });
});

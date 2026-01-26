"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const testing_1 = require("@nestjs/testing");
const auth_controller_1 = require("./auth.controller");
const auth_service_1 = require("./auth.service");
const user_entity_1 = require("./entities/user.entity");
describe('AuthController', () => {
    let controller;
    let authService;
    const mockUser = {
        id: 'user-uuid',
        email: 'test@example.com',
        firstName: 'Test',
        lastName: 'User',
        role: user_entity_1.UserRole.TECHNICIAN,
        tenantId: 'tenant-uuid',
        isActive: true,
        lastLoginAt: new Date(),
        createdAt: new Date(),
        updatedAt: new Date(),
    };
    const mockTokens = {
        accessToken: 'mock-access-token',
        refreshToken: 'mock-refresh-token',
    };
    const mockAuthResponse = {
        ...mockTokens,
        user: mockUser,
    };
    beforeEach(async () => {
        const mockAuthService = {
            register: jest.fn(),
            login: jest.fn(),
            refreshTokens: jest.fn(),
            logout: jest.fn(),
        };
        const module = await testing_1.Test.createTestingModule({
            controllers: [auth_controller_1.AuthController],
            providers: [
                {
                    provide: auth_service_1.AuthService,
                    useValue: mockAuthService,
                },
            ],
        }).compile();
        controller = module.get(auth_controller_1.AuthController);
        authService = module.get(auth_service_1.AuthService);
    });
    describe('register', () => {
        it('should register a new user and return tokens', async () => {
            const dto = {
                email: 'test@example.com',
                password: 'password123',
                firstName: 'Test',
                lastName: 'User',
            };
            authService.register.mockResolvedValue(mockAuthResponse);
            const result = await controller.register(dto);
            expect(authService.register).toHaveBeenCalledWith(dto);
            expect(result).toEqual(mockAuthResponse);
            expect(result.accessToken).toBeDefined();
            expect(result.refreshToken).toBeDefined();
            expect(result.user.email).toBe(dto.email);
        });
    });
    describe('login', () => {
        it('should authenticate user and return tokens', async () => {
            const dto = {
                email: 'test@example.com',
                password: 'password123',
            };
            authService.login.mockResolvedValue(mockAuthResponse);
            const result = await controller.login(dto);
            expect(authService.login).toHaveBeenCalledWith(dto);
            expect(result).toEqual(mockAuthResponse);
            expect(result.accessToken).toBeDefined();
        });
    });
    describe('refreshTokens', () => {
        it('should refresh tokens for authenticated user', async () => {
            authService.refreshTokens.mockResolvedValue(mockTokens);
            const result = await controller.refreshTokens('user-uuid', 'old-refresh-token');
            expect(authService.refreshTokens).toHaveBeenCalledWith('user-uuid', 'old-refresh-token');
            expect(result).toEqual(mockTokens);
        });
    });
    describe('logout', () => {
        it('should logout user and return success message', async () => {
            authService.logout.mockResolvedValue(undefined);
            const result = await controller.logout('user-uuid');
            expect(authService.logout).toHaveBeenCalledWith('user-uuid');
            expect(result).toEqual({ message: 'Logged out successfully' });
        });
    });
    describe('getCurrentUser', () => {
        it('should return current user without sensitive data', async () => {
            const fullUser = {
                ...mockUser,
                passwordHash: 'hashed',
                refreshToken: 'token',
            };
            const result = await controller.getCurrentUser(fullUser);
            expect(result).not.toHaveProperty('passwordHash');
            expect(result).not.toHaveProperty('refreshToken');
            expect(result.email).toBe(mockUser.email);
        });
    });
});
//# sourceMappingURL=auth.controller.spec.js.map
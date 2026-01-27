"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const testing_1 = require("@nestjs/testing");
const typeorm_1 = require("@nestjs/typeorm");
const jwt_1 = require("@nestjs/jwt");
const common_1 = require("@nestjs/common");
const bcrypt = require("bcrypt");
const auth_service_1 = require("./auth.service");
const user_entity_1 = require("./entities/user.entity");
jest.mock('bcrypt');
describe('AuthService', () => {
    let service;
    let userRepository;
    let jwtService;
    const mockUser = {
        id: 'user-uuid',
        email: 'test@example.com',
        passwordHash: 'hashed-password',
        firstName: 'John',
        lastName: 'Doe',
        role: user_entity_1.UserRole.TECHNICIAN,
        tenantId: 'tenant-uuid',
        isActive: true,
        refreshToken: 'hashed-refresh-token',
        createdAt: new Date(),
        updatedAt: new Date(),
    };
    beforeEach(async () => {
        const mockUserRepository = {
            findOne: jest.fn(),
            create: jest.fn(),
            save: jest.fn(),
            update: jest.fn(),
        };
        const mockJwtService = {
            signAsync: jest.fn(),
        };
        const module = await testing_1.Test.createTestingModule({
            providers: [
                auth_service_1.AuthService,
                {
                    provide: (0, typeorm_1.getRepositoryToken)(user_entity_1.User),
                    useValue: mockUserRepository,
                },
                {
                    provide: jwt_1.JwtService,
                    useValue: mockJwtService,
                },
            ],
        }).compile();
        service = module.get(auth_service_1.AuthService);
        userRepository = module.get((0, typeorm_1.getRepositoryToken)(user_entity_1.User));
        jwtService = module.get(jwt_1.JwtService);
        jest.clearAllMocks();
    });
    describe('register', () => {
        it('should register a new user successfully', async () => {
            const dto = {
                email: 'newuser@example.com',
                password: 'password123',
                firstName: 'Jane',
                lastName: 'Smith',
                role: user_entity_1.UserRole.TECHNICIAN,
            };
            userRepository.findOne.mockResolvedValue(null);
            userRepository.create.mockReturnValue({ ...mockUser, ...dto });
            userRepository.save.mockResolvedValue({ ...mockUser, ...dto });
            userRepository.update.mockResolvedValue({ affected: 1 });
            bcrypt.hash.mockResolvedValue('hashed-password');
            jwtService.signAsync
                .mockResolvedValueOnce('access-token')
                .mockResolvedValueOnce('refresh-token');
            const result = await service.register(dto);
            expect(userRepository.findOne).toHaveBeenCalledWith({
                where: { email: dto.email },
            });
            expect(result.accessToken).toBe('access-token');
            expect(result.refreshToken).toBe('refresh-token');
            expect(result.user.email).toBe(dto.email);
        });
        it('should throw ConflictException if email already exists', async () => {
            const dto = {
                email: 'existing@example.com',
                password: 'password123',
                firstName: 'Jane',
                lastName: 'Smith',
            };
            userRepository.findOne.mockResolvedValue(mockUser);
            await expect(service.register(dto)).rejects.toThrow(common_1.ConflictException);
            await expect(service.register(dto)).rejects.toThrow('User with this email already exists');
        });
    });
    describe('login', () => {
        it('should login successfully with valid credentials', async () => {
            const dto = { email: 'test@example.com', password: 'password123' };
            userRepository.findOne.mockResolvedValue(mockUser);
            userRepository.update.mockResolvedValue({ affected: 1 });
            bcrypt.compare.mockResolvedValue(true);
            bcrypt.hash.mockResolvedValue('hashed-refresh-token');
            jwtService.signAsync
                .mockResolvedValueOnce('access-token')
                .mockResolvedValueOnce('refresh-token');
            const result = await service.login(dto);
            expect(result.accessToken).toBe('access-token');
            expect(result.refreshToken).toBe('refresh-token');
            expect(result.user.email).toBe(dto.email);
        });
        it('should throw UnauthorizedException for invalid email', async () => {
            const dto = { email: 'wrong@example.com', password: 'password123' };
            userRepository.findOne.mockResolvedValue(null);
            await expect(service.login(dto)).rejects.toThrow(common_1.UnauthorizedException);
            await expect(service.login(dto)).rejects.toThrow('Invalid credentials');
        });
        it('should throw UnauthorizedException for invalid password', async () => {
            const dto = { email: 'test@example.com', password: 'wrong-password' };
            userRepository.findOne.mockResolvedValue(mockUser);
            bcrypt.compare.mockResolvedValue(false);
            await expect(service.login(dto)).rejects.toThrow(common_1.UnauthorizedException);
            await expect(service.login(dto)).rejects.toThrow('Invalid credentials');
        });
        it('should throw UnauthorizedException for deactivated account', async () => {
            const dto = { email: 'test@example.com', password: 'password123' };
            const deactivatedUser = { ...mockUser, isActive: false };
            userRepository.findOne.mockResolvedValue(deactivatedUser);
            bcrypt.compare.mockResolvedValue(true);
            await expect(service.login(dto)).rejects.toThrow(common_1.UnauthorizedException);
            await expect(service.login(dto)).rejects.toThrow('Account is deactivated');
        });
    });
    describe('refreshTokens', () => {
        it('should refresh tokens successfully', async () => {
            userRepository.findOne.mockResolvedValue(mockUser);
            userRepository.update.mockResolvedValue({ affected: 1 });
            bcrypt.compare.mockResolvedValue(true);
            bcrypt.hash.mockResolvedValue('new-hashed-refresh-token');
            jwtService.signAsync
                .mockResolvedValueOnce('new-access-token')
                .mockResolvedValueOnce('new-refresh-token');
            const result = await service.refreshTokens('user-uuid', 'valid-refresh-token');
            expect(result.accessToken).toBe('new-access-token');
            expect(result.refreshToken).toBe('new-refresh-token');
        });
        it('should throw UnauthorizedException if user not found', async () => {
            userRepository.findOne.mockResolvedValue(null);
            await expect(service.refreshTokens('invalid-uuid', 'refresh-token')).rejects.toThrow(common_1.UnauthorizedException);
        });
        it('should throw UnauthorizedException for invalid refresh token', async () => {
            userRepository.findOne.mockResolvedValue(mockUser);
            bcrypt.compare.mockResolvedValue(false);
            await expect(service.refreshTokens('user-uuid', 'invalid-refresh-token')).rejects.toThrow(common_1.UnauthorizedException);
        });
    });
    describe('logout', () => {
        it('should clear refresh token on logout', async () => {
            userRepository.update.mockResolvedValue({ affected: 1 });
            await service.logout('user-uuid');
            expect(userRepository.update).toHaveBeenCalledWith('user-uuid', {
                refreshToken: null,
            });
        });
    });
    describe('validateUser', () => {
        it('should return user from token payload', async () => {
            const payload = { sub: 'user-uuid', email: 'test@example.com', role: 'technician' };
            userRepository.findOne.mockResolvedValue(mockUser);
            const result = await service.validateUser(payload);
            expect(result).toEqual(mockUser);
            expect(userRepository.findOne).toHaveBeenCalledWith({
                where: { id: payload.sub },
            });
        });
        it('should return null if user not found', async () => {
            const payload = { sub: 'invalid-uuid', email: 'test@example.com', role: 'technician' };
            userRepository.findOne.mockResolvedValue(null);
            const result = await service.validateUser(payload);
            expect(result).toBeNull();
        });
    });
    describe('findById', () => {
        it('should return user by id', async () => {
            userRepository.findOne.mockResolvedValue(mockUser);
            const result = await service.findById('user-uuid');
            expect(result).toEqual(mockUser);
        });
        it('should return null if user not found', async () => {
            userRepository.findOne.mockResolvedValue(null);
            const result = await service.findById('invalid-uuid');
            expect(result).toBeNull();
        });
    });
});
//# sourceMappingURL=auth.service.spec.js.map
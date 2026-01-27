import { Test, TestingModule } from '@nestjs/testing';
import { getRepositoryToken } from '@nestjs/typeorm';
import { JwtService } from '@nestjs/jwt';
import { Repository } from 'typeorm';
import { ConflictException, UnauthorizedException } from '@nestjs/common';
import * as bcrypt from 'bcrypt';
import { AuthService } from './auth.service';
import { User, UserRole } from './entities/user.entity';

jest.mock('bcrypt');

describe('AuthService', () => {
    let service: AuthService;
    let userRepository: jest.Mocked<Repository<User>>;
    let jwtService: jest.Mocked<JwtService>;

    const mockUser: Partial<User> = {
        id: 'user-uuid',
        email: 'test@example.com',
        passwordHash: 'hashed-password',
        firstName: 'John',
        lastName: 'Doe',
        role: UserRole.TECHNICIAN,
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

        const module: TestingModule = await Test.createTestingModule({
            providers: [
                AuthService,
                {
                    provide: getRepositoryToken(User),
                    useValue: mockUserRepository,
                },
                {
                    provide: JwtService,
                    useValue: mockJwtService,
                },
            ],
        }).compile();

        service = module.get<AuthService>(AuthService);
        userRepository = module.get(getRepositoryToken(User));
        jwtService = module.get(JwtService);

        jest.clearAllMocks();
    });

    describe('register', () => {
        it('should register a new user successfully', async () => {
            const dto = {
                email: 'newuser@example.com',
                password: 'password123',
                firstName: 'Jane',
                lastName: 'Smith',
                role: UserRole.TECHNICIAN,
            };

            userRepository.findOne.mockResolvedValue(null);
            userRepository.create.mockReturnValue({ ...mockUser, ...dto } as User);
            userRepository.save.mockResolvedValue({ ...mockUser, ...dto } as User);
            userRepository.update.mockResolvedValue({ affected: 1 } as any);

            (bcrypt.hash as jest.Mock).mockResolvedValue('hashed-password');
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

            userRepository.findOne.mockResolvedValue(mockUser as User);

            await expect(service.register(dto)).rejects.toThrow(ConflictException);
            await expect(service.register(dto)).rejects.toThrow(
                'User with this email already exists',
            );
        });
    });

    describe('login', () => {
        it('should login successfully with valid credentials', async () => {
            const dto = { email: 'test@example.com', password: 'password123' };

            userRepository.findOne.mockResolvedValue(mockUser as User);
            userRepository.update.mockResolvedValue({ affected: 1 } as any);

            (bcrypt.compare as jest.Mock).mockResolvedValue(true);
            (bcrypt.hash as jest.Mock).mockResolvedValue('hashed-refresh-token');
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

            await expect(service.login(dto)).rejects.toThrow(UnauthorizedException);
            await expect(service.login(dto)).rejects.toThrow('Invalid credentials');
        });

        it('should throw UnauthorizedException for invalid password', async () => {
            const dto = { email: 'test@example.com', password: 'wrong-password' };

            userRepository.findOne.mockResolvedValue(mockUser as User);
            (bcrypt.compare as jest.Mock).mockResolvedValue(false);

            await expect(service.login(dto)).rejects.toThrow(UnauthorizedException);
            await expect(service.login(dto)).rejects.toThrow('Invalid credentials');
        });

        it('should throw UnauthorizedException for deactivated account', async () => {
            const dto = { email: 'test@example.com', password: 'password123' };
            const deactivatedUser = { ...mockUser, isActive: false };

            userRepository.findOne.mockResolvedValue(deactivatedUser as User);
            (bcrypt.compare as jest.Mock).mockResolvedValue(true);

            await expect(service.login(dto)).rejects.toThrow(UnauthorizedException);
            await expect(service.login(dto)).rejects.toThrow('Account is deactivated');
        });
    });

    describe('refreshTokens', () => {
        it('should refresh tokens successfully', async () => {
            userRepository.findOne.mockResolvedValue(mockUser as User);
            userRepository.update.mockResolvedValue({ affected: 1 } as any);

            (bcrypt.compare as jest.Mock).mockResolvedValue(true);
            (bcrypt.hash as jest.Mock).mockResolvedValue('new-hashed-refresh-token');
            jwtService.signAsync
                .mockResolvedValueOnce('new-access-token')
                .mockResolvedValueOnce('new-refresh-token');

            const result = await service.refreshTokens('user-uuid', 'valid-refresh-token');

            expect(result.accessToken).toBe('new-access-token');
            expect(result.refreshToken).toBe('new-refresh-token');
        });

        it('should throw UnauthorizedException if user not found', async () => {
            userRepository.findOne.mockResolvedValue(null);

            await expect(
                service.refreshTokens('invalid-uuid', 'refresh-token'),
            ).rejects.toThrow(UnauthorizedException);
        });

        it('should throw UnauthorizedException for invalid refresh token', async () => {
            userRepository.findOne.mockResolvedValue(mockUser as User);
            (bcrypt.compare as jest.Mock).mockResolvedValue(false);

            await expect(
                service.refreshTokens('user-uuid', 'invalid-refresh-token'),
            ).rejects.toThrow(UnauthorizedException);
        });
    });

    describe('logout', () => {
        it('should clear refresh token on logout', async () => {
            userRepository.update.mockResolvedValue({ affected: 1 } as any);

            await service.logout('user-uuid');

            expect(userRepository.update).toHaveBeenCalledWith('user-uuid', {
                refreshToken: null,
            });
        });
    });

    describe('validateUser', () => {
        it('should return user from token payload', async () => {
            const payload = { sub: 'user-uuid', email: 'test@example.com', role: 'technician' };
            userRepository.findOne.mockResolvedValue(mockUser as User);

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
            userRepository.findOne.mockResolvedValue(mockUser as User);

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

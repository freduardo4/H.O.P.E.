import { Injectable, UnauthorizedException, ConflictException } from '@nestjs/common';
import { JwtService } from '@nestjs/jwt';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import * as bcrypt from 'bcrypt';
import { User } from './entities/user.entity';
import { RegisterDto, LoginDto } from './dto';

export interface TokenPayload {
    sub: string;
    email: string;
    role: string;
    tenantId?: string;
}

export interface AuthTokens {
    accessToken: string;
    refreshToken: string;
}

export interface AuthResponse extends AuthTokens {
    user: Omit<User, 'passwordHash' | 'refreshToken'>;
}

@Injectable()
export class AuthService {
    constructor(
        @InjectRepository(User)
        private readonly userRepository: Repository<User>,
        private readonly jwtService: JwtService,
    ) { }

    async register(dto: RegisterDto): Promise<AuthResponse> {
        const existingUser = await this.userRepository.findOne({
            where: { email: dto.email },
        });

        if (existingUser) {
            throw new ConflictException('User with this email already exists');
        }

        const passwordHash = await bcrypt.hash(dto.password, 12);

        const user = this.userRepository.create({
            email: dto.email,
            passwordHash,
            firstName: dto.firstName,
            lastName: dto.lastName,
            role: dto.role,
            tenantId: dto.tenantId,
        });

        await this.userRepository.save(user);

        const tokens = await this.generateTokens(user);
        await this.updateRefreshToken(user.id, tokens.refreshToken);

        return {
            ...tokens,
            user: this.sanitizeUser(user),
        };
    }

    async login(dto: LoginDto): Promise<AuthResponse> {
        const user = await this.userRepository.findOne({
            where: { email: dto.email },
        });

        if (!user) {
            throw new UnauthorizedException('Invalid credentials');
        }

        const isPasswordValid = await bcrypt.compare(dto.password, user.passwordHash);

        if (!isPasswordValid) {
            throw new UnauthorizedException('Invalid credentials');
        }

        if (!user.isActive) {
            throw new UnauthorizedException('Account is deactivated');
        }

        const tokens = await this.generateTokens(user);
        await this.updateRefreshToken(user.id, tokens.refreshToken);

        // Update last login
        await this.userRepository.update(user.id, { lastLoginAt: new Date() });

        return {
            ...tokens,
            user: this.sanitizeUser(user),
        };
    }

    async refreshTokens(userId: string, refreshToken: string): Promise<AuthTokens> {
        const user = await this.userRepository.findOne({
            where: { id: userId },
        });

        if (!user || !user.refreshToken) {
            throw new UnauthorizedException('Access denied');
        }

        const isRefreshTokenValid = await bcrypt.compare(refreshToken, user.refreshToken);

        if (!isRefreshTokenValid) {
            throw new UnauthorizedException('Access denied');
        }

        const tokens = await this.generateTokens(user);
        await this.updateRefreshToken(user.id, tokens.refreshToken);

        return tokens;
    }

    async logout(userId: string): Promise<void> {
        await this.userRepository.update(userId, { refreshToken: null });
    }

    async acceptLegalTerms(userId: string, version: string): Promise<void> {
        await this.userRepository.update(userId, { acceptedLegalVersion: version });
    }

    async validateUser(payload: TokenPayload): Promise<User | null> {
        return this.userRepository.findOne({
            where: { id: payload.sub },
        });
    }

    async findById(id: string): Promise<User | null> {
        return this.userRepository.findOne({ where: { id } });
    }

    private async generateTokens(user: User): Promise<AuthTokens> {
        const payload: TokenPayload = {
            sub: user.id,
            email: user.email,
            role: user.role,
            tenantId: user.tenantId,
        };

        const [accessToken, refreshToken] = await Promise.all([
            this.jwtService.signAsync(payload, {
                secret: process.env.JWT_SECRET || 'hope-secret-key',
                expiresIn: '15m',
            }),
            this.jwtService.signAsync(payload, {
                secret: process.env.JWT_REFRESH_SECRET || 'hope-refresh-secret',
                expiresIn: '7d',
            }),
        ]);

        return { accessToken, refreshToken };
    }

    private async updateRefreshToken(userId: string, refreshToken: string): Promise<void> {
        const hashedRefreshToken = await bcrypt.hash(refreshToken, 12);
        await this.userRepository.update(userId, { refreshToken: hashedRefreshToken });
    }

    private sanitizeUser(user: User): Omit<User, 'passwordHash' | 'refreshToken'> {
        const { passwordHash, refreshToken, ...sanitized } = user;
        return sanitized as Omit<User, 'passwordHash' | 'refreshToken'>;
    }
}

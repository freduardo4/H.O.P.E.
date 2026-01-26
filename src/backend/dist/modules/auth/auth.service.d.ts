import { JwtService } from '@nestjs/jwt';
import { Repository } from 'typeorm';
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
export declare class AuthService {
    private readonly userRepository;
    private readonly jwtService;
    constructor(userRepository: Repository<User>, jwtService: JwtService);
    register(dto: RegisterDto): Promise<AuthResponse>;
    login(dto: LoginDto): Promise<AuthResponse>;
    refreshTokens(userId: string, refreshToken: string): Promise<AuthTokens>;
    logout(userId: string): Promise<void>;
    validateUser(payload: TokenPayload): Promise<User | null>;
    findById(id: string): Promise<User | null>;
    private generateTokens;
    private updateRefreshToken;
    private sanitizeUser;
}

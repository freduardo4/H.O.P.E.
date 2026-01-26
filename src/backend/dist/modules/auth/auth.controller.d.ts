import { AuthService, AuthResponse, AuthTokens } from './auth.service';
import { RegisterDto, LoginDto } from './dto';
import { User } from './entities/user.entity';
export declare class AuthController {
    private readonly authService;
    constructor(authService: AuthService);
    register(dto: RegisterDto): Promise<AuthResponse>;
    login(dto: LoginDto): Promise<AuthResponse>;
    refreshTokens(userId: string, refreshToken: string): Promise<AuthTokens>;
    logout(userId: string): Promise<{
        message: string;
    }>;
    getCurrentUser(user: User): Promise<Omit<User, 'passwordHash' | 'refreshToken'>>;
}

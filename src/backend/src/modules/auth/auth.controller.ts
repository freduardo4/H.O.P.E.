import {
    Controller,
    Post,
    Body,
    HttpCode,
    HttpStatus,
    UseGuards,
    Get,
    Req,
    Res,
} from '@nestjs/common';
import { AuthGuard } from '@nestjs/passport';
import { Response } from 'express';
import { AuthService, AuthResponse, AuthTokens } from './auth.service';
import { RegisterDto, LoginDto } from './dto';
import { Public } from './decorators/public.decorator';
import { CurrentUser } from './decorators/current-user.decorator';
import { JwtAuthGuard } from './guards/jwt-auth.guard';
import { User } from './entities/user.entity';

@Controller('auth')
export class AuthController {
    constructor(private readonly authService: AuthService) { }

    @Public()
    @Post('register')
    async register(@Body() dto: RegisterDto): Promise<AuthResponse> {
        return this.authService.register(dto);
    }

    @Public()
    @Post('login')
    @HttpCode(HttpStatus.OK)
    async login(@Body() dto: LoginDto): Promise<AuthResponse> {
        return this.authService.login(dto);
    }

    @UseGuards(JwtAuthGuard)
    @Post('refresh')
    @HttpCode(HttpStatus.OK)
    async refreshTokens(
        @CurrentUser('id') userId: string,
        @Body('refreshToken') refreshToken: string,
    ): Promise<AuthTokens> {
        return this.authService.refreshTokens(userId, refreshToken);
    }

    @UseGuards(JwtAuthGuard)
    @Post('logout')
    @HttpCode(HttpStatus.OK)
    async logout(@CurrentUser('id') userId: string): Promise<{ message: string }> {
        await this.authService.logout(userId);
        return { message: 'Logged out successfully' };
    }

    @UseGuards(JwtAuthGuard)
    @Get('me')
    async getCurrentUser(@CurrentUser() user: User): Promise<Omit<User, 'passwordHash' | 'refreshToken'>> {
        const { passwordHash, refreshToken, ...sanitized } = user;
        return sanitized as Omit<User, 'passwordHash' | 'refreshToken'>;
    }

    @Post('accept-legal')
    @HttpCode(HttpStatus.OK)
    async acceptLegalTerms(
        @CurrentUser('id') userId: string,
        @Body('version') version: string,
    ): Promise<{ message: string }> {
        await this.authService.acceptLegalTerms(userId, version);
        return { message: 'Legal terms accepted successfully' };
    }

    @Public()
    @Get('google')
    @UseGuards(AuthGuard('google'))
    async googleAuth(@Req() req) {
        // Initiates Google OAuth2 flow
    }

    @Public()
    @Get('google/callback')
    @UseGuards(AuthGuard('google'))
    async googleAuthRedirect(@Req() req, @Res() res: Response) {
        const result = await this.authService.validateOAuthUser(req.user);
        // In a real app, you might redirect to a frontend URL with tokens
        return res.json(result);
    }

    @Public()
    @Get('github')
    @UseGuards(AuthGuard('github'))
    async githubAuth(@Req() req) {
        // Initiates GitHub OAuth2 flow
    }

    @Public()
    @Get('github/callback')
    @UseGuards(AuthGuard('github'))
    async githubAuthRedirect(@Req() req, @Res() res: Response) {
        const result = await this.authService.validateOAuthUser(req.user);
        return res.json(result);
    }
}

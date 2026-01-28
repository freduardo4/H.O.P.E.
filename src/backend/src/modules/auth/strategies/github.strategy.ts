import { Injectable } from '@nestjs/common';
import { PassportStrategy } from '@nestjs/passport';
import { Strategy } from 'passport-github2';

@Injectable()
export class GithubStrategy extends PassportStrategy(Strategy, 'github') {
    constructor() {
        super({
            clientID: process.env.GITHUB_CLIENT_ID || 'dummy-github-id',
            clientSecret: process.env.GITHUB_CLIENT_SECRET || 'dummy-github-secret',
            callbackURL: 'http://localhost:3000/auth/github/callback',
            scope: ['user:email'],
        });
    }

    async validate(accessToken: string, refreshToken: string, profile: any, done: any): Promise<any> {
        const { username, emails, photos } = profile;
        const user = {
            email: emails[0].value,
            username,
            picture: photos[0].value,
            accessToken,
        };
        done(null, user);
    }
}

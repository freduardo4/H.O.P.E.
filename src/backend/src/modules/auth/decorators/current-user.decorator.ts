import { createParamDecorator, ExecutionContext } from '@nestjs/common';
import { User } from '../entities/user.entity';

export const CurrentUser = createParamDecorator(
    (data: keyof User | undefined, context: ExecutionContext) => {
        const request = context.switchToHttp().getRequest();
        const user = request.user as User;

        return data ? user?.[data] : user;
    },
);

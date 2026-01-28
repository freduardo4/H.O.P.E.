import { createParamDecorator, ExecutionContext } from '@nestjs/common';
import { GqlExecutionContext } from '@nestjs/graphql';
import { User } from '../entities/user.entity';

export const CurrentUser = createParamDecorator(
    (data: keyof User | undefined, context: ExecutionContext) => {
        const ctx = GqlExecutionContext.create(context);
        const request = ctx.getContext().req || context.switchToHttp().getRequest();
        const user = request.user as User;

        return data ? user?.[data] : user;
    },
);

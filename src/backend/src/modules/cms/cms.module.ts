import { Module } from '@nestjs/common';
import { CacheModule } from '@nestjs/cache-manager';
import { CmsService } from './cms.service';
import { CmsController } from './cms.controller';

@Module({
    imports: [
        CacheModule.register({
            ttl: 300, // 5 minutes
            max: 100, // maximum number of items in cache
        }),
    ],
    providers: [CmsService],
    controllers: [CmsController],
    exports: [CmsService],
})
export class CmsModule { }

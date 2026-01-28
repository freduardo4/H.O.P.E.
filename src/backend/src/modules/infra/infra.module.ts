import { Module } from '@nestjs/common';
import { BackupService } from './backup.service';
import { ConfigService } from './config.service';
import { ConfigController } from './config.controller';

@Module({
    providers: [BackupService, ConfigService],
    controllers: [ConfigController],
    exports: [BackupService, ConfigService],
})
export class InfraModule { }

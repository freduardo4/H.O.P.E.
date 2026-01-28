import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { SafetyLogsService } from './safety-logs.service';
import { SafetyLogsController } from './safety-logs.controller';
import { SafetyLog } from './entities/safety-log.entity';

@Module({
    imports: [TypeOrmModule.forFeature([SafetyLog])],
    controllers: [SafetyLogsController],
    providers: [SafetyLogsService],
    exports: [SafetyLogsService],
})
export class SafetyLogsModule { }

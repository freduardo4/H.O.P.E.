import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { DiagnosticsController } from './diagnostics.controller';
import { DiagnosticsService } from './diagnostics.service';
import { DiagnosticSession } from './entities/diagnostic-session.entity';
import { OBD2Reading } from './entities/obd2-reading.entity';
import { AuthModule } from '../auth';

@Module({
    imports: [
        TypeOrmModule.forFeature([DiagnosticSession, OBD2Reading]),
        AuthModule,
    ],
    controllers: [DiagnosticsController],
    providers: [DiagnosticsService],
    exports: [DiagnosticsService],
})
export class DiagnosticsModule {}

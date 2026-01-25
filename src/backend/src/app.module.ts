import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { HealthModule } from './health';
import { AuthModule } from './modules/auth';
import { VehiclesModule } from './modules/vehicles';
import { DiagnosticsModule } from './modules/diagnostics';
import { getDatabaseConfig } from './config';

@Module({
    imports: [
        TypeOrmModule.forRoot(getDatabaseConfig()),
        HealthModule,
        AuthModule,
        VehiclesModule,
        DiagnosticsModule,
    ],
    controllers: [],
    providers: [],
})
export class AppModule {}

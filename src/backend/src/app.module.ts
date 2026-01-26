import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { HealthModule } from './health';
import { AuthModule } from './modules/auth';
import { VehiclesModule } from './modules/vehicles';
import { DiagnosticsModule } from './modules/diagnostics';
import { ECUCalibrationsModule } from './modules/ecu-calibrations';
import { ReportsModule } from './modules/reports';
import { CustomersModule } from './modules/customers';
import { getDatabaseConfig } from './config';

@Module({
    imports: [
        TypeOrmModule.forRoot(getDatabaseConfig()),
        HealthModule,
        AuthModule,
        VehiclesModule,
        DiagnosticsModule,
        ECUCalibrationsModule,
        ReportsModule,
        CustomersModule,
    ],
    controllers: [],
    providers: [],
})
export class AppModule {}

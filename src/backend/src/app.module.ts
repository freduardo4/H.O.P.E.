import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { GraphQLModule } from '@nestjs/graphql';
import { ApolloDriver, ApolloDriverConfig } from '@nestjs/apollo';
import { join } from 'path';
import { LoggerModule } from 'nestjs-pino';
import { SentryModule } from '@sentry/nestjs/setup';
import { ScheduleModule } from '@nestjs/schedule';
import { HealthModule } from './health';
import { AuthModule } from './modules/auth';
import { VehiclesModule } from './modules/vehicles';
import { DiagnosticsModule } from './modules/diagnostics';
import { ECUCalibrationsModule } from './modules/ecu-calibrations';
import { ReportsModule } from './modules/reports';
import { CustomersModule } from './modules/customers';
import { SafetyLogsModule } from './modules/safety-logs';
import { TuningModule } from './modules/tuning/tuning.module';
import { InfraModule } from './modules/infra/infra.module';
import { WikiFixModule } from './modules/wiki-fix/wiki-fix.module';
import { CmsModule } from './modules/cms/cms.module';
import { MarketplaceModule } from './modules/marketplace/marketplace.module';
import { getDatabaseConfig } from './config';

@Module({
    imports: [
        TypeOrmModule.forRoot(getDatabaseConfig()),
        GraphQLModule.forRoot<ApolloDriverConfig>({
            driver: ApolloDriver,
            autoSchemaFile: join(process.cwd(), 'src/schema.gql'),
            sortSchema: true,
            playground: true,
        }),
        LoggerModule.forRoot({
            pinoHttp: {
                transport: process.env.NODE_ENV !== 'production'
                    ? { target: 'pino-pretty', options: { colorize: true } }
                    : undefined,
                level: process.env.LOG_LEVEL || 'info',
            },
        }),
        SentryModule.forRoot(),
        ScheduleModule.forRoot(),
        HealthModule,
        AuthModule,
        VehiclesModule,
        DiagnosticsModule,
        ECUCalibrationsModule,
        ReportsModule,
        CustomersModule,
        SafetyLogsModule,
        TuningModule,
        InfraModule,
        WikiFixModule,
        CmsModule,
        MarketplaceModule,
    ],
    controllers: [],
    providers: [],
})
export class AppModule { }

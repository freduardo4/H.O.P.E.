import { TypeOrmModuleOptions } from '@nestjs/typeorm';
import { join } from 'path';

export const getDatabaseConfig = (): TypeOrmModuleOptions => ({
    type: 'postgres',
    host: process.env.DB_HOST || 'localhost',
    port: parseInt(process.env.DB_PORT || '5432', 10),
    username: process.env.DB_USERNAME || 'hope',
    password: process.env.DB_PASSWORD || 'hope_password',
    database: process.env.DB_DATABASE || 'hope_db',
    autoLoadEntities: true,
    synchronize: false, // Transitioned to migrations for Phase 8.4
    logging: process.env.NODE_ENV === 'development',
    ssl: process.env.DB_SSL === 'true' ? { rejectUnauthorized: false } : false,
    extra: {
        max: parseInt(process.env.DB_POOL_SIZE || '10', 10),
    },
    migrations: [join(__dirname, '/../database/migrations/*{.ts,.js}')],
});

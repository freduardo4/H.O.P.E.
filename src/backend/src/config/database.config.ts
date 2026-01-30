import { TypeOrmModuleOptions } from '@nestjs/typeorm';
import { join } from 'path';

export const getDatabaseConfig = (): TypeOrmModuleOptions => {
    const isTest = process.env.NODE_ENV === 'test';
    const isSqlite = process.env.DB_TYPE === 'sqlite';

    if (isSqlite || isTest) {
        return {
            type: 'better-sqlite3',
            database: ':memory:',
            dropSchema: true,
            synchronize: true,
            autoLoadEntities: true,
            logging: true,
        };
    }

    return {
        type: 'postgres',
        host: process.env.DB_HOST,
        port: parseInt(process.env.DB_PORT || '5432', 10),
        username: process.env.DB_USERNAME,
        password: process.env.DB_PASSWORD,
        database: process.env.DB_DATABASE,
        autoLoadEntities: true,
        synchronize: false, // Transitioned to migrations for Phase 8.4
        logging: process.env.NODE_ENV === 'development',
        ssl: process.env.DB_SSL === 'true' ? { rejectUnauthorized: false } : false,
        extra: {
            max: parseInt(process.env.DB_POOL_SIZE || '10', 10),
        },
        migrations: [join(__dirname, '/../database/migrations/*{.ts,.js}')],
    };
};


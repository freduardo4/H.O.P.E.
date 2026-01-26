"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.InitialSchema1706000000000 = void 0;
const typeorm_1 = require("typeorm");
class InitialSchema1706000000000 {
    constructor() {
        this.name = 'InitialSchema1706000000000';
    }
    async up(queryRunner) {
        await queryRunner.query(`
            CREATE TYPE "user_role_enum" AS ENUM ('admin', 'shop_owner', 'technician', 'viewer')
        `);
        await queryRunner.query(`
            CREATE TYPE "fuel_type_enum" AS ENUM ('gasoline', 'diesel', 'hybrid', 'electric', 'lpg')
        `);
        await queryRunner.query(`
            CREATE TYPE "transmission_type_enum" AS ENUM ('manual', 'automatic', 'dsg', 'cvt')
        `);
        await queryRunner.query(`
            CREATE TYPE "session_status_enum" AS ENUM ('active', 'completed', 'cancelled')
        `);
        await queryRunner.query(`
            CREATE TYPE "session_type_enum" AS ENUM ('diagnostic', 'performance', 'tune', 'maintenance')
        `);
        await queryRunner.createTable(new typeorm_1.Table({
            name: 'users',
            columns: [
                {
                    name: 'id',
                    type: 'uuid',
                    isPrimary: true,
                    generationStrategy: 'uuid',
                    default: 'uuid_generate_v4()',
                },
                {
                    name: 'email',
                    type: 'varchar',
                    isUnique: true,
                },
                {
                    name: 'passwordHash',
                    type: 'varchar',
                },
                {
                    name: 'firstName',
                    type: 'varchar',
                },
                {
                    name: 'lastName',
                    type: 'varchar',
                },
                {
                    name: 'role',
                    type: 'user_role_enum',
                    default: "'technician'",
                },
                {
                    name: 'tenantId',
                    type: 'varchar',
                    isNullable: true,
                },
                {
                    name: 'isActive',
                    type: 'boolean',
                    default: true,
                },
                {
                    name: 'refreshToken',
                    type: 'varchar',
                    isNullable: true,
                },
                {
                    name: 'lastLoginAt',
                    type: 'timestamp',
                    isNullable: true,
                },
                {
                    name: 'createdAt',
                    type: 'timestamp',
                    default: 'CURRENT_TIMESTAMP',
                },
                {
                    name: 'updatedAt',
                    type: 'timestamp',
                    default: 'CURRENT_TIMESTAMP',
                },
            ],
        }), true);
        await queryRunner.createTable(new typeorm_1.Table({
            name: 'vehicles',
            columns: [
                {
                    name: 'id',
                    type: 'uuid',
                    isPrimary: true,
                    generationStrategy: 'uuid',
                    default: 'uuid_generate_v4()',
                },
                {
                    name: 'tenantId',
                    type: 'varchar',
                },
                {
                    name: 'customerId',
                    type: 'varchar',
                    isNullable: true,
                },
                {
                    name: 'vin',
                    type: 'varchar',
                    length: '17',
                    isNullable: true,
                },
                {
                    name: 'make',
                    type: 'varchar',
                },
                {
                    name: 'model',
                    type: 'varchar',
                },
                {
                    name: 'year',
                    type: 'int',
                },
                {
                    name: 'variant',
                    type: 'varchar',
                    isNullable: true,
                },
                {
                    name: 'engineCode',
                    type: 'varchar',
                    isNullable: true,
                },
                {
                    name: 'engineDisplacement',
                    type: 'int',
                    isNullable: true,
                },
                {
                    name: 'enginePower',
                    type: 'int',
                    isNullable: true,
                },
                {
                    name: 'fuelType',
                    type: 'fuel_type_enum',
                    isNullable: true,
                },
                {
                    name: 'transmission',
                    type: 'transmission_type_enum',
                    isNullable: true,
                },
                {
                    name: 'licensePlate',
                    type: 'varchar',
                    isNullable: true,
                },
                {
                    name: 'mileage',
                    type: 'int',
                    isNullable: true,
                },
                {
                    name: 'ecuType',
                    type: 'varchar',
                    isNullable: true,
                },
                {
                    name: 'ecuSoftwareVersion',
                    type: 'varchar',
                    isNullable: true,
                },
                {
                    name: 'notes',
                    type: 'text',
                    isNullable: true,
                },
                {
                    name: 'isActive',
                    type: 'boolean',
                    default: true,
                },
                {
                    name: 'createdAt',
                    type: 'timestamp',
                    default: 'CURRENT_TIMESTAMP',
                },
                {
                    name: 'updatedAt',
                    type: 'timestamp',
                    default: 'CURRENT_TIMESTAMP',
                },
            ],
        }), true);
        await queryRunner.createTable(new typeorm_1.Table({
            name: 'diagnostic_sessions',
            columns: [
                {
                    name: 'id',
                    type: 'uuid',
                    isPrimary: true,
                    generationStrategy: 'uuid',
                    default: 'uuid_generate_v4()',
                },
                {
                    name: 'tenantId',
                    type: 'varchar',
                },
                {
                    name: 'vehicleId',
                    type: 'uuid',
                },
                {
                    name: 'technicianId',
                    type: 'uuid',
                    isNullable: true,
                },
                {
                    name: 'type',
                    type: 'session_type_enum',
                    default: "'diagnostic'",
                },
                {
                    name: 'status',
                    type: 'session_status_enum',
                    default: "'active'",
                },
                {
                    name: 'startTime',
                    type: 'timestamp',
                },
                {
                    name: 'endTime',
                    type: 'timestamp',
                    isNullable: true,
                },
                {
                    name: 'mileageAtSession',
                    type: 'int',
                    isNullable: true,
                },
                {
                    name: 'notes',
                    type: 'text',
                    isNullable: true,
                },
                {
                    name: 'ecuSnapshot',
                    type: 'jsonb',
                    isNullable: true,
                },
                {
                    name: 'dtcCodes',
                    type: 'jsonb',
                    isNullable: true,
                },
                {
                    name: 'performanceMetrics',
                    type: 'jsonb',
                    isNullable: true,
                },
                {
                    name: 'createdAt',
                    type: 'timestamp',
                    default: 'CURRENT_TIMESTAMP',
                },
                {
                    name: 'updatedAt',
                    type: 'timestamp',
                    default: 'CURRENT_TIMESTAMP',
                },
            ],
        }), true);
        await queryRunner.createTable(new typeorm_1.Table({
            name: 'obd2_readings',
            columns: [
                {
                    name: 'id',
                    type: 'uuid',
                    isPrimary: true,
                    generationStrategy: 'uuid',
                    default: 'uuid_generate_v4()',
                },
                {
                    name: 'sessionId',
                    type: 'uuid',
                },
                {
                    name: 'timestamp',
                    type: 'timestamp',
                },
                {
                    name: 'pid',
                    type: 'varchar',
                },
                {
                    name: 'name',
                    type: 'varchar',
                },
                {
                    name: 'value',
                    type: 'decimal',
                    precision: 12,
                    scale: 4,
                },
                {
                    name: 'unit',
                    type: 'varchar',
                },
                {
                    name: 'rawResponse',
                    type: 'text',
                    isNullable: true,
                },
                {
                    name: 'createdAt',
                    type: 'timestamp',
                    default: 'CURRENT_TIMESTAMP',
                },
            ],
        }), true);
        await queryRunner.createIndex('vehicles', new typeorm_1.TableIndex({
            name: 'IDX_vehicles_tenantId',
            columnNames: ['tenantId'],
        }));
        await queryRunner.createIndex('vehicles', new typeorm_1.TableIndex({
            name: 'IDX_vehicles_customerId',
            columnNames: ['customerId'],
        }));
        await queryRunner.createIndex('vehicles', new typeorm_1.TableIndex({
            name: 'IDX_vehicles_vin',
            columnNames: ['vin'],
        }));
        await queryRunner.createIndex('diagnostic_sessions', new typeorm_1.TableIndex({
            name: 'IDX_diagnostic_sessions_vehicleId',
            columnNames: ['vehicleId'],
        }));
        await queryRunner.createIndex('diagnostic_sessions', new typeorm_1.TableIndex({
            name: 'IDX_diagnostic_sessions_tenantId',
            columnNames: ['tenantId'],
        }));
        await queryRunner.createIndex('obd2_readings', new typeorm_1.TableIndex({
            name: 'IDX_obd2_readings_sessionId_timestamp',
            columnNames: ['sessionId', 'timestamp'],
        }));
        await queryRunner.createIndex('obd2_readings', new typeorm_1.TableIndex({
            name: 'IDX_obd2_readings_sessionId_pid',
            columnNames: ['sessionId', 'pid'],
        }));
        await queryRunner.createForeignKey('diagnostic_sessions', new typeorm_1.TableForeignKey({
            name: 'FK_diagnostic_sessions_vehicleId',
            columnNames: ['vehicleId'],
            referencedColumnNames: ['id'],
            referencedTableName: 'vehicles',
            onDelete: 'CASCADE',
        }));
        await queryRunner.createForeignKey('obd2_readings', new typeorm_1.TableForeignKey({
            name: 'FK_obd2_readings_sessionId',
            columnNames: ['sessionId'],
            referencedColumnNames: ['id'],
            referencedTableName: 'diagnostic_sessions',
            onDelete: 'CASCADE',
        }));
        await queryRunner.query('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"');
        try {
            await queryRunner.query(`
                SELECT create_hypertable('obd2_readings', 'timestamp', if_not_exists => TRUE)
            `);
        }
        catch {
            console.log('TimescaleDB not available, using regular table for obd2_readings');
        }
    }
    async down(queryRunner) {
        await queryRunner.dropForeignKey('obd2_readings', 'FK_obd2_readings_sessionId');
        await queryRunner.dropForeignKey('diagnostic_sessions', 'FK_diagnostic_sessions_vehicleId');
        await queryRunner.dropIndex('obd2_readings', 'IDX_obd2_readings_sessionId_pid');
        await queryRunner.dropIndex('obd2_readings', 'IDX_obd2_readings_sessionId_timestamp');
        await queryRunner.dropIndex('diagnostic_sessions', 'IDX_diagnostic_sessions_tenantId');
        await queryRunner.dropIndex('diagnostic_sessions', 'IDX_diagnostic_sessions_vehicleId');
        await queryRunner.dropIndex('vehicles', 'IDX_vehicles_vin');
        await queryRunner.dropIndex('vehicles', 'IDX_vehicles_customerId');
        await queryRunner.dropIndex('vehicles', 'IDX_vehicles_tenantId');
        await queryRunner.dropTable('obd2_readings');
        await queryRunner.dropTable('diagnostic_sessions');
        await queryRunner.dropTable('vehicles');
        await queryRunner.dropTable('users');
        await queryRunner.query('DROP TYPE IF EXISTS "session_type_enum"');
        await queryRunner.query('DROP TYPE IF EXISTS "session_status_enum"');
        await queryRunner.query('DROP TYPE IF EXISTS "transmission_type_enum"');
        await queryRunner.query('DROP TYPE IF EXISTS "fuel_type_enum"');
        await queryRunner.query('DROP TYPE IF EXISTS "user_role_enum"');
    }
}
exports.InitialSchema1706000000000 = InitialSchema1706000000000;
//# sourceMappingURL=1706000000000-InitialSchema.js.map
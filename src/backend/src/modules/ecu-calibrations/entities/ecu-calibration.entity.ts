import {
    Entity,
    PrimaryGeneratedColumn,
    Column,
    CreateDateColumn,
    UpdateDateColumn,
    ManyToOne,
    JoinColumn,
} from 'typeorm';

export enum CalibrationType {
    STOCK = 'stock',
    STAGE_1 = 'stage1',
    STAGE_2 = 'stage2',
    STAGE_3 = 'stage3',
    CUSTOM = 'custom',
}

export enum FileFormat {
    BIN = 'bin',
    HEX = 'hex',
    S19 = 's19',
    UNKNOWN = 'unknown',
}

@Entity('ecu_calibrations')
export class ECUCalibration {
    @PrimaryGeneratedColumn('uuid')
    id: string;

    @Column()
    tenantId: string;

    @Column()
    vehicleId: string;

    @Column({ nullable: true })
    customerId: string;

    @Column()
    fileName: string;

    @Column()
    s3Key: string;

    @Column()
    s3Bucket: string;

    @Column({ type: 'bigint' })
    fileSize: number;

    @Column({ type: 'enum', enum: FileFormat, default: FileFormat.UNKNOWN })
    fileFormat: FileFormat;

    @Column({ type: 'enum', enum: CalibrationType })
    calibrationType: CalibrationType;

    @Column()
    checksum: string;

    @Column({ default: 1 })
    version: number;

    @Column({ nullable: true })
    previousVersionId: string;

    @Column({ nullable: true })
    ecuType: string;

    @Column({ nullable: true })
    ecuSoftwareVersion: string;

    @Column({ type: 'text', nullable: true })
    notes: string;

    @Column({ type: 'jsonb', nullable: true })
    metadata: {
        enginePowerStock?: number;
        enginePowerTuned?: number;
        torqueStock?: number;
        torqueTuned?: number;
        fuelConsumptionImprovement?: number;
        [key: string]: any;
    };

    @Column({ default: true })
    isActive: boolean;

    @Column({ nullable: true })
    uploadedBy: string;

    @CreateDateColumn()
    createdAt: Date;

    @UpdateDateColumn()
    updatedAt: Date;
}

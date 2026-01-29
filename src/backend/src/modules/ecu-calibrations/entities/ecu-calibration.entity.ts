import {
    Entity,
    PrimaryGeneratedColumn,
    Column,
    CreateDateColumn,
    UpdateDateColumn,
} from 'typeorm';
import { ObjectType, Field, ID, Int, Float, registerEnumType } from '@nestjs/graphql';

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

registerEnumType(CalibrationType, { name: 'CalibrationType' });
registerEnumType(FileFormat, { name: 'FileFormat' });

@ObjectType()
@Entity('ecu_calibrations')
export class ECUCalibration {
    @Field(() => ID)
    @PrimaryGeneratedColumn('uuid')
    id: string;

    @Field()
    @Column()
    tenantId: string;

    @Field()
    @Column()
    vehicleId: string;

    @Field({ nullable: true })
    @Column({ nullable: true })
    customerId: string;

    @Field()
    @Column()
    fileName: string;

    @Field()
    @Column()
    s3Key: string;

    @Field()
    @Column()
    s3Bucket: string;

    @Field(() => Int)
    @Column({ type: 'int' })
    fileSize: number;

    @Field(() => FileFormat)
    @Column({ type: 'simple-enum', enum: FileFormat, default: FileFormat.UNKNOWN })
    fileFormat: FileFormat;

    @Field(() => CalibrationType)
    @Column({ type: 'simple-enum', enum: CalibrationType })
    calibrationType: CalibrationType;

    @Field()
    @Column()
    checksum: string;

    @Field(() => Int)
    @Column({ default: 1 })
    version: number;

    @Field({ nullable: true })
    @Column({ nullable: true })
    previousVersionId: string;

    @Field({ nullable: true })
    @Column({ nullable: true })
    ecuType: string;

    @Field({ nullable: true })
    @Column({ nullable: true })
    ecuSoftwareVersion: string;

    @Field({ nullable: true })
    @Column({ type: 'text', nullable: true })
    notes: string;

    @Field(() => String, { nullable: true })
    @Column({ type: 'simple-json', nullable: true })
    metadata: {
        enginePowerStock?: number;
        enginePowerTuned?: number;
        torqueStock?: number;
        torqueTuned?: number;
        fuelConsumptionImprovement?: number;
        [key: string]: any;
    };

    @Field()
    @Column({ default: true })
    isActive: boolean;

    @Field({ nullable: true })
    @Column({ nullable: true })
    uploadedBy: string;

    @Field()
    @CreateDateColumn()
    createdAt: Date;

    @Field()
    @UpdateDateColumn()
    updatedAt: Date;
}

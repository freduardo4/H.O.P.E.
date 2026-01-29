import {
    Entity,
    PrimaryGeneratedColumn,
    Column,
    CreateDateColumn,
    UpdateDateColumn,
} from 'typeorm';
import { ObjectType, Field, ID, Int, registerEnumType } from '@nestjs/graphql';

export enum ReportType {
    DIAGNOSTIC = 'diagnostic',
    TUNING = 'tuning',
    COMPARISON = 'comparison',
    MAINTENANCE = 'maintenance',
}

export enum ReportStatus {
    PENDING = 'pending',
    GENERATING = 'generating',
    COMPLETED = 'completed',
    FAILED = 'failed',
}

registerEnumType(ReportType, { name: 'ReportType' });
registerEnumType(ReportStatus, { name: 'ReportStatus' });

@ObjectType()
@Entity('reports')
export class Report {
    @Field(() => ID)
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

    @Field({ nullable: true })
    @Column({ nullable: true })
    diagnosticSessionId: string;

    @Field(() => ReportType)
    @Column({ type: 'simple-enum', enum: ReportType })
    reportType: ReportType;

    @Field(() => ReportStatus)
    @Column({ type: 'simple-enum', enum: ReportStatus, default: ReportStatus.PENDING })
    status: ReportStatus;

    @Field()
    @Column()
    title: string;

    @Field({ nullable: true })
    @Column({ type: 'text', nullable: true })
    description: string;

    @Field({ nullable: true })
    @Column({ nullable: true })
    s3Key: string;

    @Field({ nullable: true })
    @Column({ nullable: true })
    s3Bucket: string;

    @Field(() => Int, { nullable: true })
    @Column({ type: 'int', nullable: true })
    fileSize: number;

    @Field(() => String, { nullable: true })
    @Column({ type: 'simple-json', nullable: true })
    data: {
        // Diagnostic report data
        dtcCodes?: string[];
        obd2Readings?: any[];
        anomalies?: any[];

        // Tuning report data
        beforePower?: number;
        afterPower?: number;
        beforeTorque?: number;
        afterTorque?: number;
        fuelConsumptionImprovement?: number;
        modifications?: string[];

        // Generic data
        [key: string]: any;
    };

    @Field({ nullable: true })
    @Column({ nullable: true })
    generatedBy: string;

    @Field({ nullable: true })
    @Column({ type: 'datetime', nullable: true })
    generatedAt: Date;

    @Field({ nullable: true })
    @Column({ type: 'text', nullable: true })
    errorMessage: string;

    @Field()
    @CreateDateColumn()
    createdAt: Date;

    @Field()
    @UpdateDateColumn()
    updatedAt: Date;
}

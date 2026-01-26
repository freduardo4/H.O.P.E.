import {
    Entity,
    PrimaryGeneratedColumn,
    Column,
    CreateDateColumn,
    UpdateDateColumn,
} from 'typeorm';

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

@Entity('reports')
export class Report {
    @PrimaryGeneratedColumn('uuid')
    id: string;

    @Column()
    tenantId: string;

    @Column()
    vehicleId: string;

    @Column({ nullable: true })
    customerId: string;

    @Column({ nullable: true })
    diagnosticSessionId: string;

    @Column({ type: 'enum', enum: ReportType })
    reportType: ReportType;

    @Column({ type: 'enum', enum: ReportStatus, default: ReportStatus.PENDING })
    status: ReportStatus;

    @Column()
    title: string;

    @Column({ type: 'text', nullable: true })
    description: string;

    @Column({ nullable: true })
    s3Key: string;

    @Column({ nullable: true })
    s3Bucket: string;

    @Column({ type: 'bigint', nullable: true })
    fileSize: number;

    @Column({ type: 'jsonb', nullable: true })
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

    @Column({ nullable: true })
    generatedBy: string;

    @Column({ nullable: true })
    generatedAt: Date;

    @Column({ type: 'text', nullable: true })
    errorMessage: string;

    @CreateDateColumn()
    createdAt: Date;

    @UpdateDateColumn()
    updatedAt: Date;
}

import {
    Entity,
    PrimaryGeneratedColumn,
    Column,
    CreateDateColumn,
    UpdateDateColumn,
    ManyToOne,
    OneToMany,
    JoinColumn,
} from 'typeorm';

export enum SessionStatus {
    ACTIVE = 'active',
    COMPLETED = 'completed',
    CANCELLED = 'cancelled',
}

export enum SessionType {
    DIAGNOSTIC = 'diagnostic',
    PERFORMANCE = 'performance',
    TUNE = 'tune',
    MAINTENANCE = 'maintenance',
}

@Entity('diagnostic_sessions')
export class DiagnosticSession {
    @PrimaryGeneratedColumn('uuid')
    id: string;

    @Column()
    tenantId: string;

    @Column()
    vehicleId: string;

    @Column({ nullable: true })
    technicianId: string;

    @Column({ type: 'enum', enum: SessionType, default: SessionType.DIAGNOSTIC })
    type: SessionType;

    @Column({ type: 'enum', enum: SessionStatus, default: SessionStatus.ACTIVE })
    status: SessionStatus;

    @Column({ type: 'timestamp' })
    startTime: Date;

    @Column({ type: 'timestamp', nullable: true })
    endTime: Date;

    @Column({ nullable: true })
    mileageAtSession: number;

    @Column({ type: 'text', nullable: true })
    notes: string;

    @Column({ type: 'jsonb', nullable: true })
    ecuSnapshot: Record<string, any>;

    @Column({ type: 'jsonb', nullable: true })
    dtcCodes: string[];

    @Column({ type: 'jsonb', nullable: true })
    performanceMetrics: {
        maxRpm?: number;
        maxSpeed?: number;
        maxBoost?: number;
        avgLoad?: number;
    };

    @CreateDateColumn()
    createdAt: Date;

    @UpdateDateColumn()
    updatedAt: Date;

    get duration(): number | null {
        if (!this.endTime) return null;
        return Math.floor((this.endTime.getTime() - this.startTime.getTime()) / 1000);
    }
}

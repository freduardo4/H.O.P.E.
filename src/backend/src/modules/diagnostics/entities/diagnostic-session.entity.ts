import {
    Entity,
    PrimaryGeneratedColumn,
    Column,
    CreateDateColumn,
    UpdateDateColumn,
} from 'typeorm';
import { ObjectType, Field, ID, registerEnumType } from '@nestjs/graphql';

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

registerEnumType(SessionStatus, { name: 'SessionStatus' });
registerEnumType(SessionType, { name: 'SessionType' });

@ObjectType()
@Entity('diagnostic_sessions')
export class DiagnosticSession {
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
    technicianId: string;

    @Field(() => SessionType)
    @Column({ type: 'simple-enum', enum: SessionType, default: SessionType.DIAGNOSTIC })
    type: SessionType;

    @Field(() => SessionStatus)
    @Column({ type: 'simple-enum', enum: SessionStatus, default: SessionStatus.ACTIVE })
    status: SessionStatus;

    @Field()
    @Column({ type: 'datetime' })
    startTime: Date;

    @Field({ nullable: true })
    @Column({ type: 'datetime', nullable: true })
    endTime: Date;

    @Field({ nullable: true })
    @Column({ nullable: true })
    mileageAtSession: number;

    @Field({ nullable: true })
    @Column({ type: 'text', nullable: true })
    notes: string;

    @Field(() => String, { nullable: true }) // Represent JSON as String in GraphQL
    @Column({ type: 'simple-json', nullable: true })
    ecuSnapshot: Record<string, any>;

    @Field(() => [String], { nullable: true })
    @Column({ type: 'simple-json', nullable: true })
    dtcCodes: string[];

    @Field(() => String, { nullable: true }) // Represent JSON as String in GraphQL
    @Column({ type: 'simple-json', nullable: true })
    performanceMetrics: {
        maxRpm?: number;
        maxSpeed?: number;
        maxBoost?: number;
        avgLoad?: number;
    };

    @Field()
    @CreateDateColumn()
    createdAt: Date;

    @Field()
    @UpdateDateColumn()
    updatedAt: Date;

    @Field({ nullable: true })
    get duration(): number | null {
        if (!this.endTime) return null;
        return Math.floor((this.endTime.getTime() - this.startTime.getTime()) / 1000);
    }
}

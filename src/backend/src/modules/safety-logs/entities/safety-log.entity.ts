import { Entity, Column, PrimaryGeneratedColumn, CreateDateColumn, Index } from 'typeorm';
import { ObjectType, Field, ID, Float } from '@nestjs/graphql';

@ObjectType()
@Entity('safety_logs')
export class SafetyLog {
    @Field(() => ID)
    @PrimaryGeneratedColumn('uuid')
    id: string;

    @Field()
    @Column()
    eventType: string;

    @Field()
    @Index()
    @Column()
    ecuId: string;

    @Field(() => Float, { nullable: true })
    @Column('float', { nullable: true })
    voltage: number | null;

    @Field()
    @Column()
    success: boolean;

    @Field({ nullable: true })
    @Column({ nullable: true })
    message: string;

    @Field({ nullable: true })
    @Column({ type: 'text', nullable: true })
    metadata: string;

    @Field({ nullable: true })
    @Column({ type: 'datetime', default: () => 'CURRENT_TIMESTAMP' })
    timestamp: Date;
}

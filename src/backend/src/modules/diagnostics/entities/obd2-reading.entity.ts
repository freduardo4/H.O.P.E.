import {
    Entity,
    PrimaryGeneratedColumn,
    Column,
    CreateDateColumn,
    Index,
} from 'typeorm';
import { ObjectType, Field, ID, Float } from '@nestjs/graphql';

@ObjectType()
@Entity('obd2_readings')
@Index(['sessionId', 'timestamp'])
@Index(['sessionId', 'pid'])
export class OBD2Reading {
    @Field(() => ID)
    @PrimaryGeneratedColumn('uuid')
    id: string;

    @Field()
    @Column()
    sessionId: string;

    @Field()
    @Column({ type: 'datetime' })
    timestamp: Date;

    @Field()
    @Column()
    pid: string;

    @Field()
    @Column()
    name: string;

    @Field(() => Float)
    @Column({ type: 'float' })
    value: number;

    @Field()
    @Column()
    unit: string;

    @Field({ nullable: true })
    @Column({ type: 'text', nullable: true })
    rawResponse: string;

    @Field()
    @CreateDateColumn()
    createdAt: Date;
}

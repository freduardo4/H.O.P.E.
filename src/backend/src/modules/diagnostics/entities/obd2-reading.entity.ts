import {
    Entity,
    PrimaryGeneratedColumn,
    Column,
    CreateDateColumn,
    Index,
} from 'typeorm';

@Entity('obd2_readings')
@Index(['sessionId', 'timestamp'])
@Index(['sessionId', 'pid'])
export class OBD2Reading {
    @PrimaryGeneratedColumn('uuid')
    id: string;

    @Column()
    sessionId: string;

    @Column({ type: 'timestamp' })
    timestamp: Date;

    @Column()
    pid: string;

    @Column()
    name: string;

    @Column({ type: 'decimal', precision: 12, scale: 4 })
    value: number;

    @Column()
    unit: string;

    @Column({ type: 'text', nullable: true })
    rawResponse: string;

    @CreateDateColumn()
    createdAt: Date;
}

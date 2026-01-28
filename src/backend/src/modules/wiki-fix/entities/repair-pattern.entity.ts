import { Entity, PrimaryGeneratedColumn, Column, ManyToOne, JoinColumn } from 'typeorm';
import { WikiPost } from './wiki-post.entity';

@Entity('repair_patterns')
export class RepairPattern {
    @PrimaryGeneratedColumn('uuid')
    id: string;

    @Column()
    postId: string;

    @ManyToOne('WikiPost', 'patterns')
    @JoinColumn({ name: 'postId' })
    post: WikiPost;

    @Column()
    dtc: string; // e.g. "P0300"

    @Column({ type: 'float', nullable: true })
    minAnomalyScore: number;

    @Column({ type: 'jsonb', nullable: true })
    sensorCriteria: Record<string, any>; // e.g. { "RPM": "> 3000" }

    @Column({ default: 1.0 })
    confidenceScore: number;
}

import { Entity, PrimaryGeneratedColumn, Column, ManyToOne, JoinColumn } from 'typeorm';
import { ObjectType, Field, ID, Float } from '@nestjs/graphql';
import { WikiPost } from './wiki-post.entity';

@ObjectType()
@Entity('repair_patterns')
export class RepairPattern {
    @Field(() => ID)
    @PrimaryGeneratedColumn('uuid')
    id: string;

    @Field()
    @Column()
    postId: string;

    @Field(() => WikiPost)
    @ManyToOne('WikiPost', 'patterns')
    @JoinColumn({ name: 'postId' })
    post: WikiPost;

    @Field()
    @Column()
    dtc: string; // e.g. "P0300"

    @Field(() => Float, { nullable: true })
    @Column({ type: 'float', nullable: true })
    minAnomalyScore: number;

    @Field(() => String, { nullable: true })
    @Column({ type: 'simple-json', nullable: true })
    sensorCriteria: Record<string, any>; // e.g. { "RPM": "> 3000" }

    @Field(() => Float)
    @Column({ default: 1.0 })
    confidenceScore: number;
}

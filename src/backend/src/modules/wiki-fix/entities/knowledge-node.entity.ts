import { Entity, PrimaryGeneratedColumn, Column, CreateDateColumn, UpdateDateColumn, OneToMany } from 'typeorm';
import { ObjectType, Field, ID, Float, registerEnumType } from '@nestjs/graphql';
import type { KnowledgeEdge } from './knowledge-edge.entity';

export enum NodeType {
    DTC = 'DTC',
    SYMPTOM = 'SYMPTOM',
    PART = 'PART',
    REPAIR_STEP = 'REPAIR_STEP',
    VEHICLE_SYSTEM = 'VEHICLE_SYSTEM',
}

registerEnumType(NodeType, { name: 'NodeType' });

@ObjectType()
@Entity('knowledge_nodes')
export class KnowledgeNode {
    @Field(() => ID)
    @PrimaryGeneratedColumn('uuid')
    id: string;

    @Field()
    @Column()
    name: string; // e.g., "P0300", "Rough Idle", "Spark Plug"

    @Field(() => NodeType)
    @Column({ type: 'simple-enum', enum: NodeType })
    type: NodeType;

    @Field({ nullable: true })
    @Column('text', { nullable: true })
    description: string;

    @Field(() => [ID], { nullable: true })
    @OneToMany('KnowledgeEdge', 'sourceNode')
    outEdges: KnowledgeEdge[];

    @Field(() => [ID], { nullable: true })
    @OneToMany('KnowledgeEdge', 'targetNode')
    inEdges: KnowledgeEdge[];

    @Field(() => [Float], { nullable: true })
    @Column('simple-array', { nullable: true })
    embedding: number[];

    @Field()
    @CreateDateColumn()
    createdAt: Date;

    @Field()
    @UpdateDateColumn()
    updatedAt: Date;
}

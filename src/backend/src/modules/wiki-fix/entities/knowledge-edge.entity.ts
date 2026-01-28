import { Entity, PrimaryGeneratedColumn, Column, ManyToOne } from 'typeorm';
import { ObjectType, Field, ID, registerEnumType } from '@nestjs/graphql';
import { KnowledgeNode } from './knowledge-node.entity';

export enum EdgeType {
    CAUSED_BY = 'CAUSED_BY',
    SOLVED_BY = 'SOLVED_BY',
    PART_OF = 'PART_OF',
    SYMPTOM_OF = 'SYMPTOM_OF',
    REPLACES = 'REPLACES',
}

registerEnumType(EdgeType, { name: 'EdgeType' });

@ObjectType()
@Entity('knowledge_edges')
export class KnowledgeEdge {
    @Field(() => ID)
    @PrimaryGeneratedColumn('uuid')
    id: string;

    @Field(() => KnowledgeNode)
    @ManyToOne(() => KnowledgeNode, (node) => node.outEdges)
    sourceNode: KnowledgeNode;

    @Field(() => KnowledgeNode)
    @ManyToOne(() => KnowledgeNode, (node) => node.inEdges)
    targetNode: KnowledgeNode;

    @Field(() => EdgeType)
    @Column({ type: 'simple-enum', enum: EdgeType })
    relation: EdgeType;

    @Field({ nullable: true })
    @Column({ type: 'float', default: 1.0 })
    weight: number; // For ranking relevance
}

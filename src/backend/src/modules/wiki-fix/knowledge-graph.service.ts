import { Injectable } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository, In } from 'typeorm';
import { KnowledgeNode, NodeType } from './entities/knowledge-node.entity';
import { KnowledgeEdge, EdgeType } from './entities/knowledge-edge.entity';

@Injectable()
export class KnowledgeGraphService {
    constructor(
        @InjectRepository(KnowledgeNode)
        private readonly nodeRepository: Repository<KnowledgeNode>,
        @InjectRepository(KnowledgeEdge)
        private readonly edgeRepository: Repository<KnowledgeEdge>,
    ) { }

    async findRelatedNodes(nodeId: string, depth = 1): Promise<KnowledgeNode[]> {
        const edges = await this.edgeRepository.find({
            where: [{ sourceNode: { id: nodeId } }, { targetNode: { id: nodeId } }],
            relations: ['sourceNode', 'targetNode'],
        });

        const relatedIds = edges.map(e =>
            e.sourceNode.id === nodeId ? e.targetNode.id : e.sourceNode.id
        );

        return this.nodeRepository.find({
            where: { id: In(relatedIds) },
        });
    }

    async semanticSearch(query: string): Promise<KnowledgeNode[]> {
        // This would typically call an embedding model (e.g., OpenAI text-embedding-ada-002)
        // and then use cosine similarity in the DB (PostGIS/PgVector)
        // For now, we fallback to keyword search on the name/description
        return this.nodeRepository.find({
            where: [
                { name: query },
                { description: query }
            ]
        });
    }

    async createRelationship(sourceId: string, targetId: string, type: EdgeType): Promise<KnowledgeEdge> {
        const sourceNode = await this.nodeRepository.findOneBy({ id: sourceId });
        const targetNode = await this.nodeRepository.findOneBy({ id: targetId });

        if (!sourceNode || !targetNode) throw new Error('Nodes not found');

        const edge = this.edgeRepository.create({
            sourceNode,
            targetNode,
            relation: type,
        });

        return this.edgeRepository.save(edge);
    }
}

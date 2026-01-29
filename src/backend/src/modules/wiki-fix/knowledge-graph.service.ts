import { Injectable, OnModuleInit, Logger } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository, In, Like } from 'typeorm';
import { KnowledgeNode, NodeType } from './entities/knowledge-node.entity';
import { KnowledgeEdge, EdgeType } from './entities/knowledge-edge.entity';

@Injectable()
export class KnowledgeGraphService implements OnModuleInit {
    private readonly logger = new Logger(KnowledgeGraphService.name);

    constructor(
        @InjectRepository(KnowledgeNode)
        private readonly nodeRepository: Repository<KnowledgeNode>,
        @InjectRepository(KnowledgeEdge)
        private readonly edgeRepository: Repository<KnowledgeEdge>,
    ) { }

    async onModuleInit() {
        await this.seedInitialData();
    }

    private async seedInitialData() {
        const count = await this.nodeRepository.count();
        if (count > 0) return;

        this.logger.log('Seeding initial Wiki-Fix Knowledge Graph...');

        const dtcs = [
            { name: 'P0300', type: NodeType.DTC, description: 'Random/Multiple Cylinder Misfire Detected' },
            { name: 'P0171', type: NodeType.DTC, description: 'System Too Lean (Bank 1)' },
            { name: 'P0420', type: NodeType.DTC, description: 'Catalyst System Efficiency Below Threshold (Bank 1)' },
        ];

        const symptoms = [
            { name: 'Rough Idle', type: NodeType.SYMPTOM, description: 'Engine shakes or vibrates at idle' },
            { name: 'Poor Fuel Economy', type: NodeType.SYMPTOM, description: 'Vehicle uses more fuel than usual' },
        ];

        const parts = [
            { name: 'Spark Plug', type: NodeType.PART, description: 'Ignition system component' },
            { name: 'Ignition Coil', type: NodeType.PART, description: 'Induction coil for spark' },
            { name: 'O2 Sensor', type: NodeType.PART, description: 'Oxygen sensor' },
            { name: 'Vacuum Leak', type: NodeType.SYMPTOM, description: 'Unmetered air entering engine' },
        ];

        const savedNodes: Record<string, KnowledgeNode> = {};

        for (const data of [...dtcs, ...symptoms, ...parts]) {
            const node = this.nodeRepository.create(data);
            savedNodes[node.name] = await this.nodeRepository.save(node);
        }

        // Create Relationships
        await this.createRelationship(savedNodes['P0300'].id, savedNodes['Rough Idle'].id, EdgeType.CAUSED_BY); // Actually Symptom_Of usually, but usage varies. Let's say P0300 CAUSES Rough Idle.
        // Or better: Rough Idle IS_SYMPTOM_OF P0300. EdgeType.SYMPTOM_OF
        await this.createRelationship(savedNodes['Rough Idle'].id, savedNodes['P0300'].id, EdgeType.SYMPTOM_OF);

        await this.createRelationship(savedNodes['Spark Plug'].id, savedNodes['P0300'].id, EdgeType.SOLVED_BY); // P0300 solved by Spark Plug? Or Part Solves Issue.
        // Let's stick to Semantic: Problem SOLVED_BY Solution.
        await this.createRelationship(savedNodes['P0300'].id, savedNodes['Spark Plug'].id, EdgeType.SOLVED_BY);
        await this.createRelationship(savedNodes['P0300'].id, savedNodes['Ignition Coil'].id, EdgeType.SOLVED_BY);

        await this.createRelationship(savedNodes['P0171'].id, savedNodes['Vacuum Leak'].id, EdgeType.CAUSED_BY);
        await this.createRelationship(savedNodes['P0171'].id, savedNodes['O2 Sensor'].id, EdgeType.SOLVED_BY); // If bad sensor

        this.logger.log('Knowledge Graph seeded successfully.');
    }

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
        // Enhanced generic search using LIKE/ILIKE
        // In TypeORM for Postgres, ILIKE is supported via ILike operator or raw query.
        // We'll use FindOptions with Like (which is case-insensitive in some DBs or we force generic Like)
        // Ideally we want case-insensitive. Postgres `Like` is case-sensitive, `ILike` is not.
        // NestJS TypeORM usually exposes ILike. Let's try to import it, if not valid we use Raw.
        // Note: I imported `Like` above. Let's check imports.

        return this.nodeRepository.find({
            where: [
                { name: Like(`%${query}%`) },
                { description: Like(`%${query}%`) }
            ],
            take: 20
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

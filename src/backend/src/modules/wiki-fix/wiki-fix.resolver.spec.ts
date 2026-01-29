import { Test, TestingModule } from '@nestjs/testing';
import { WikiFixResolver } from './wiki-fix.resolver';
import { KnowledgeGraphService } from './knowledge-graph.service';
import { KnowledgeNode, NodeType } from './entities/knowledge-node.entity';
import { KnowledgeEdge, EdgeType } from './entities/knowledge-edge.entity';

describe('WikiFixResolver', () => {
    let resolver: WikiFixResolver;
    let service: jest.Mocked<KnowledgeGraphService>;

    const mockNode: Partial<KnowledgeNode> = {
        id: 'node-1',
        name: 'P0420 fix',
        description: 'Clean the CAT',
        type: NodeType.REPAIR_STEP,
    };

    const mockEdge: Partial<KnowledgeEdge> = {
        id: 'edge-1',
        relation: EdgeType.SOLVED_BY,
    };

    beforeEach(async () => {
        const mockKnowledgeGraphService = {
            semanticSearch: jest.fn(),
            findRelatedNodes: jest.fn(),
            createRelationship: jest.fn(),
        };

        const module: TestingModule = await Test.createTestingModule({
            providers: [
                WikiFixResolver,
                {
                    provide: KnowledgeGraphService,
                    useValue: mockKnowledgeGraphService,
                },
            ],
        }).compile();

        resolver = module.get<WikiFixResolver>(WikiFixResolver);
        service = module.get(KnowledgeGraphService);
    });

    it('should be defined', () => {
        expect(resolver).toBeDefined();
    });

    describe('searchKnowledge', () => {
        it('should return search results', async () => {
            service.semanticSearch.mockResolvedValue([mockNode] as KnowledgeNode[]);
            const result = await resolver.searchKnowledge('P0420');
            expect(result).toEqual([mockNode]);
            expect(service.semanticSearch).toHaveBeenCalledWith('P0420');
        });
    });

    describe('relatedFixes', () => {
        it('should return related nodes', async () => {
            service.findRelatedNodes.mockResolvedValue([mockNode] as KnowledgeNode[]);
            const result = await resolver.relatedFixes('node-1');
            expect(result).toEqual([mockNode]);
            expect(service.findRelatedNodes).toHaveBeenCalledWith('node-1');
        });
    });

    describe('addKnowledgeRelation', () => {
        it('should create a relationship', async () => {
            service.createRelationship.mockResolvedValue(mockEdge as KnowledgeEdge);
            const result = await resolver.addKnowledgeRelation('node-1', 'node-2', EdgeType.SOLVED_BY);
            expect(result).toEqual(mockEdge);
            expect(service.createRelationship).toHaveBeenCalledWith('node-1', 'node-2', EdgeType.SOLVED_BY);
        });
    });
});

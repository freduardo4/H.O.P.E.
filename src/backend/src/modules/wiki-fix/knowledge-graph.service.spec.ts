
import { Test, TestingModule } from '@nestjs/testing';
import { getRepositoryToken } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { KnowledgeGraphService } from './knowledge-graph.service';
import { KnowledgeNode, NodeType } from './entities/knowledge-node.entity';
import { KnowledgeEdge } from './entities/knowledge-edge.entity';

describe('KnowledgeGraphService', () => {
    let service: KnowledgeGraphService;
    let nodeRepository: Repository<KnowledgeNode>;
    let edgeRepository: Repository<KnowledgeEdge>;

    const mockNodeRepository = {
        count: jest.fn(),
        create: jest.fn(),
        save: jest.fn(),
        find: jest.fn(),
        findOneBy: jest.fn(),
    };

    const mockEdgeRepository = {
        create: jest.fn(),
        save: jest.fn(),
        find: jest.fn(),
    };

    beforeEach(async () => {
        const module: TestingModule = await Test.createTestingModule({
            providers: [
                KnowledgeGraphService,
                {
                    provide: getRepositoryToken(KnowledgeNode),
                    useValue: mockNodeRepository,
                },
                {
                    provide: getRepositoryToken(KnowledgeEdge),
                    useValue: mockEdgeRepository,
                },
            ],
        }).compile();

        service = module.get<KnowledgeGraphService>(KnowledgeGraphService);
        nodeRepository = module.get<Repository<KnowledgeNode>>(getRepositoryToken(KnowledgeNode));
        edgeRepository = module.get<Repository<KnowledgeEdge>>(getRepositoryToken(KnowledgeEdge));
    });

    afterEach(() => {
        jest.clearAllMocks();
    });

    it('should be defined', () => {
        expect(service).toBeDefined();
    });

    describe('onModuleInit', () => {
        it('should seed data if graph is empty', async () => {
            mockNodeRepository.count.mockResolvedValue(0);
            mockNodeRepository.create.mockImplementation((dto) => dto);
            mockNodeRepository.save.mockImplementation((node) => Promise.resolve({ ...node, id: 'uuid-' + node.name }));
            mockNodeRepository.findOneBy.mockImplementation(({ id }) => Promise.resolve({ id })); // For createRelationship

            await service.onModuleInit();

            expect(mockNodeRepository.count).toHaveBeenCalled();
            // We have 3 DTCs, 2 Symptoms, 4 Parts = 9 nodes
            expect(mockNodeRepository.save).toHaveBeenCalledTimes(9);
            // Relationships: P0300(3), P0171(2) = 5 edges
            expect(mockEdgeRepository.save).toHaveBeenCalledTimes(7);
        });

        it('should NOT seed data if graph is not empty', async () => {
            mockNodeRepository.count.mockResolvedValue(5);

            await service.onModuleInit();

            expect(mockNodeRepository.count).toHaveBeenCalled();
            expect(mockNodeRepository.save).not.toHaveBeenCalled();
        });
    });

    describe('semanticSearch', () => {
        it('should find nodes by name or description using partial match', async () => {
            const mockResult = [{ id: '1', name: 'P0300' }];
            mockNodeRepository.find.mockResolvedValue(mockResult);

            const results = await service.semanticSearch('Misfire');

            expect(mockNodeRepository.find).toHaveBeenCalledWith(expect.objectContaining({
                take: 20
            }));
            expect(results).toEqual(mockResult);
        });
    });
});

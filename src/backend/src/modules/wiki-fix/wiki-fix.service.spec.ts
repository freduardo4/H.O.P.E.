import { Test, TestingModule } from '@nestjs/testing';
import { getRepositoryToken } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { NotFoundException } from '@nestjs/common';
import { WikiFixService } from './wiki-fix.service';
import { WikiPost } from './entities/wiki-post.entity';
import { RepairPattern } from './entities/repair-pattern.entity';
import { CreateWikiPostDto, SearchWikiFixDto } from './dto/wiki-fix.dto';

describe('WikiFixService', () => {
    let service: WikiFixService;
    let postRepository: jest.Mocked<Repository<WikiPost>>;
    let patternRepository: jest.Mocked<Repository<RepairPattern>>;

    const mockPost: Partial<WikiPost> = {
        id: 'post-uuid',
        title: 'Test Post',
        content: 'Test Content',
        authorId: 'user-uuid',
        upvotes: 0,
        createdAt: new Date(),
        patterns: [],
    };

    const mockQueryBuilder = {
        leftJoinAndSelect: jest.fn().mockReturnThis(),
        andWhere: jest.fn().mockReturnThis(),
        innerJoin: jest.fn().mockReturnThis(),
        skip: jest.fn().mockReturnThis(),
        take: jest.fn().mockReturnThis(),
        orderBy: jest.fn().mockReturnThis(),
        getManyAndCount: jest.fn(),
    };

    beforeEach(async () => {
        const module: TestingModule = await Test.createTestingModule({
            providers: [
                WikiFixService,
                {
                    provide: getRepositoryToken(WikiPost),
                    useValue: {
                        create: jest.fn(),
                        save: jest.fn(),
                        findOne: jest.fn(),
                        createQueryBuilder: jest.fn(() => mockQueryBuilder),
                    },
                },
                {
                    provide: getRepositoryToken(RepairPattern),
                    useValue: {
                        find: jest.fn(),
                    },
                },
            ],
        }).compile();

        service = module.get<WikiFixService>(WikiFixService);
        postRepository = module.get(getRepositoryToken(WikiPost));
        patternRepository = module.get(getRepositoryToken(RepairPattern));

        jest.clearAllMocks();
    });

    describe('createPost', () => {
        it('should create and save a new post', async () => {
            const dto: CreateWikiPostDto = {
                title: 'New Post',
                content: 'New Content',
            };

            const createdPost = { ...mockPost, ...dto };
            postRepository.create.mockReturnValue(createdPost as WikiPost);
            postRepository.save.mockResolvedValue(createdPost as WikiPost);

            const result = await service.createPost('user-uuid', dto);

            expect(postRepository.create).toHaveBeenCalledWith({
                ...dto,
                authorId: 'user-uuid',
            });
            expect(postRepository.save).toHaveBeenCalled();
            expect(result.title).toBe('New Post');
        });
    });

    describe('findAll', () => {
        it('should return paginated posts', async () => {
            const posts = [mockPost];
            mockQueryBuilder.getManyAndCount.mockResolvedValue([posts, 1]);

            const dto: SearchWikiFixDto = { page: 1, limit: 10 };
            const result = await service.findAll(dto);

            expect(result.items).toHaveLength(1);
            expect(result.total).toBe(1);
            expect(mockQueryBuilder.skip).toHaveBeenCalledWith(0);
            expect(mockQueryBuilder.take).toHaveBeenCalledWith(10);
        });

        it('should filter by query string', async () => {
            mockQueryBuilder.getManyAndCount.mockResolvedValue([[], 0]);

            await service.findAll({ query: 'misfire' });

            expect(mockQueryBuilder.andWhere).toHaveBeenCalledWith(
                expect.stringContaining('post.title ILIKE :q'),
                { q: '%misfire%' }
            );
        });

        it('should filter by DTC', async () => {
            mockQueryBuilder.getManyAndCount.mockResolvedValue([[], 0]);

            await service.findAll({ dtc: 'P0300' });

            expect(mockQueryBuilder.innerJoin).toHaveBeenCalledWith(
                'post.patterns',
                'p2',
                'p2.dtc = :dtc',
                { dtc: 'P0300' }
            );
        });
    });

    describe('findOne', () => {
        it('should return a post by id', async () => {
            postRepository.findOne.mockResolvedValue(mockPost as WikiPost);

            const result = await service.findOne('post-uuid');

            expect(postRepository.findOne).toHaveBeenCalledWith({
                where: { id: 'post-uuid' },
                relations: ['author', 'patterns'],
            });
            expect(result.id).toBe('post-uuid');
        });

        it('should throw NotFoundException if post not found', async () => {
            postRepository.findOne.mockResolvedValue(null);

            await expect(service.findOne('invalid-id')).rejects.toThrow(NotFoundException);
        });
    });

    describe('upvote', () => {
        it('should increment upvotes and save', async () => {
            const post = { ...mockPost, upvotes: 5 } as WikiPost;
            postRepository.findOne.mockResolvedValue(post);
            postRepository.save.mockResolvedValue({ ...post, upvotes: 6 } as WikiPost);

            const result = await service.upvote('post-uuid');

            expect(post.upvotes).toBe(6);
            expect(postRepository.save).toHaveBeenCalledWith(post);
            expect(result.upvotes).toBe(6);
        });
    });
});

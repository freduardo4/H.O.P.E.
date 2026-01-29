import { Test, TestingModule } from '@nestjs/testing';
import { WikiFixController } from './wiki-fix.controller';
import { WikiFixService } from './wiki-fix.service';
import { CreateWikiPostDto, SearchWikiFixDto } from './dto/wiki-fix.dto';
import { User } from '../auth/entities/user.entity';

describe('WikiFixController', () => {
    let controller: WikiFixController;
    let service: WikiFixService;

    const mockWikiFixService = {
        createPost: jest.fn(),
        findAll: jest.fn(),
        findOne: jest.fn(),
        upvote: jest.fn(),
    };

    const mockUser = { id: 'user-123' } as User;

    beforeEach(async () => {
        const module: TestingModule = await Test.createTestingModule({
            controllers: [WikiFixController],
            providers: [
                {
                    provide: WikiFixService,
                    useValue: mockWikiFixService,
                },
            ],
        }).compile();

        controller = module.get<WikiFixController>(WikiFixController);
        service = module.get<WikiFixService>(WikiFixService);
        jest.clearAllMocks();
    });

    it('should be defined', () => {
        expect(controller).toBeDefined();
    });

    describe('createPost', () => {
        it('should call service.createPost with user id and dto', async () => {
            const dto: CreateWikiPostDto = { title: 'Test', content: 'Content', tags: [] };
            await controller.createPost(mockUser, dto);
            expect(service.createPost).toHaveBeenCalledWith(mockUser.id, dto);
        });
    });

    describe('findAll', () => {
        it('should call service.findAll with query params', async () => {
            const dto: SearchWikiFixDto = { query: 'fix', page: 1 };
            await controller.findAll(dto);
            expect(service.findAll).toHaveBeenCalledWith(dto);
        });
    });

    describe('findOne', () => {
        it('should call service.findOne with id', async () => {
            const id = 'post-123';
            await controller.findOne(id);
            expect(service.findOne).toHaveBeenCalledWith(id);
        });
    });

    describe('upvote', () => {
        it('should call service.upvote with id', async () => {
            const id = 'post-123';
            await controller.upvote(id);
            expect(service.upvote).toHaveBeenCalledWith(id);
        });
    });
});

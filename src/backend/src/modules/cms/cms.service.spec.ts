import { Test, TestingModule } from '@nestjs/testing';
import { CmsService } from './cms.service';

describe('CmsService', () => {
    let service: CmsService;

    beforeEach(async () => {
        const module: TestingModule = await Test.createTestingModule({
            providers: [CmsService],
        }).compile();

        service = module.get<CmsService>(CmsService);
    });

    it('should be defined', () => {
        expect(service).toBeDefined();
    });

    it('should return mock content', async () => {
        const content = await service.getAllContent();
        expect(content).toBeDefined();
        expect(content.length).toBeGreaterThan(0);
        expect(content[0].title).toBe('Welcome to H.O.P.E. Central');
    });

    it('should filter content by category', async () => {
        const news = await service.getContentByCategory('news');
        expect(news.every(c => c.category === 'news')).toBeTruthy();
    });
});

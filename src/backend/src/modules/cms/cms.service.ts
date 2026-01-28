import { Injectable, Logger } from '@nestjs/common';

export interface CmsContent {
    id: string;
    title: string;
    body: string;
    category: string;
    publishedAt: Date;
}

@Injectable()
export class CmsService {
    private readonly logger = new Logger(CmsService.name);

    // Mock data for development
    private mockContent: CmsContent[] = [
        {
            id: '1',
            title: 'Welcome to H.O.P.E. Central',
            body: 'Your journey with Advanced Agentic Coding starts here. Explore our Wiki-Fix and Calibration Marketplace.',
            category: 'announcement',
            publishedAt: new Date(),
        },
        {
            id: '2',
            title: 'New Update: J2534 Stability Improvements',
            body: 'The latest desktop client update includes significant stability improvements for high-frequency data streaming.',
            category: 'news',
            publishedAt: new Date(),
        },
    ];

    async getAllContent(): Promise<CmsContent[]> {
        this.logger.log('Fetching all CMS content');
        // In a real scenario, this would fetch from Strapi/Contentful
        return this.mockContent;
    }

    async getContentByCategory(category: string): Promise<CmsContent[]> {
        this.logger.log(`Fetching CMS content for category: ${category}`);
        return this.mockContent.filter(c => c.category === category);
    }

    async getContentById(id: string): Promise<CmsContent | undefined> {
        return this.mockContent.find(c => c.id === id);
    }
}

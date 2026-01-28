import { Controller, Get, Param, Query, UseGuards, UseInterceptors } from '@nestjs/common';
import { CacheInterceptor } from '@nestjs/cache-manager';
import { CmsService, CmsContent } from './cms.service';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';

@Controller('cms')
@UseGuards(JwtAuthGuard)
@UseInterceptors(CacheInterceptor)
export class CmsController {
    constructor(private readonly cmsService: CmsService) { }

    @Get('content')
    async getAllContent(): Promise<CmsContent[]> {
        return this.cmsService.getAllContent();
    }

    @Get('content/category/:category')
    async getContentByCategory(@Param('category') category: string): Promise<CmsContent[]> {
        return this.cmsService.getContentByCategory(category);
    }

    @Get('content/:id')
    async getContentById(@Param('id') id: string): Promise<CmsContent> {
        return this.cmsService.getContentById(id);
    }
}

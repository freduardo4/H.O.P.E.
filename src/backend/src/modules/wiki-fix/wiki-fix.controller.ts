import { Controller, Get, Post, Body, Param, Query, UseGuards, ParseUUIDPipe } from '@nestjs/common';
import { WikiFixService } from './wiki-fix.service';
import { CreateWikiPostDto, SearchWikiFixDto } from './dto/wiki-fix.dto';
import { WikiPost } from './entities/wiki-post.entity';
import { JwtAuthGuard, CurrentUser } from '../auth';
import { User } from '../auth/entities/user.entity';

@Controller('wiki-fix')
@UseGuards(JwtAuthGuard)
export class WikiFixController {
    constructor(private readonly wikiFixService: WikiFixService) { }

    @Post('posts')
    async createPost(
        @CurrentUser() user: User,
        @Body() dto: CreateWikiPostDto,
    ): Promise<WikiPost> {
        return this.wikiFixService.createPost(user.id, dto);
    }

    @Get('posts')
    async findAll(@Query() dto: SearchWikiFixDto) {
        return this.wikiFixService.findAll(dto);
    }

    @Get('posts/:id')
    async findOne(@Param('id', ParseUUIDPipe) id: string): Promise<WikiPost> {
        return this.wikiFixService.findOne(id);
    }

    @Post('posts/:id/upvote')
    async upvote(@Param('id', ParseUUIDPipe) id: string): Promise<WikiPost> {
        return this.wikiFixService.upvote(id);
    }
}

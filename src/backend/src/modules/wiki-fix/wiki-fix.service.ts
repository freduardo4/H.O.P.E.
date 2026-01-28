import { Injectable, NotFoundException } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository, Like, In } from 'typeorm';
import { WikiPost } from './entities/wiki-post.entity';
import { RepairPattern } from './entities/repair-pattern.entity';
import { CreateWikiPostDto, SearchWikiFixDto } from './dto/wiki-fix.dto';

@Injectable()
export class WikiFixService {
    constructor(
        @InjectRepository(WikiPost)
        private readonly postRepository: Repository<WikiPost>,
        @InjectRepository(RepairPattern)
        private readonly patternRepository: Repository<RepairPattern>,
    ) { }

    async createPost(userId: string, dto: CreateWikiPostDto): Promise<WikiPost> {
        const post = this.postRepository.create({
            ...dto,
            authorId: userId,
        });
        return this.postRepository.save(post);
    }

    async findAll(dto: SearchWikiFixDto) {
        const { query, dtc, page = 1, limit = 10 } = dto;
        const skip = (page - 1) * limit;

        const queryBuilder = this.postRepository.createQueryBuilder('post')
            .leftJoinAndSelect('post.author', 'author')
            .leftJoinAndSelect('post.patterns', 'patterns');

        if (query) {
            queryBuilder.andWhere('(post.title ILIKE :q OR post.content ILIKE :q)', { q: `%${query}%` });
        }

        if (dtc) {
            queryBuilder.innerJoin('post.patterns', 'p2', 'p2.dtc = :dtc', { dtc });
        }

        const [items, total] = await queryBuilder
            .skip(skip)
            .take(limit)
            .orderBy('post.createdAt', 'DESC')
            .getManyAndCount();

        return {
            items,
            total,
            page,
            limit,
        };
    }

    async findOne(id: string): Promise<WikiPost> {
        const post = await this.postRepository.findOne({
            where: { id },
            relations: ['author', 'patterns'],
        });
        if (!post) throw new NotFoundException('Post not found');
        return post;
    }

    async upvote(id: string): Promise<WikiPost> {
        const post = await this.findOne(id);
        post.upvotes += 1;
        return this.postRepository.save(post);
    }
}

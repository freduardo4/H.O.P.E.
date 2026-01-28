import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { WikiFixService } from './wiki-fix.service';
import { WikiFixController } from './wiki-fix.controller';
import { WikiPost } from './entities/wiki-post.entity';
import { RepairPattern } from './entities/repair-pattern.entity';
import { KnowledgeNode } from './entities/knowledge-node.entity';
import { KnowledgeEdge } from './entities/knowledge-edge.entity';
import { WikiFixResolver } from './wiki-fix.resolver';
import { KnowledgeGraphService } from './knowledge-graph.service';

@Module({
    imports: [TypeOrmModule.forFeature([WikiPost, RepairPattern, KnowledgeNode, KnowledgeEdge])],
    controllers: [WikiFixController],
    providers: [WikiFixService, WikiFixResolver, KnowledgeGraphService],
    exports: [WikiFixService, KnowledgeGraphService],
})
export class WikiFixModule { }

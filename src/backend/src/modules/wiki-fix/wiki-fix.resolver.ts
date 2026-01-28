import { Resolver, Query, Mutation, Args, ID } from '@nestjs/graphql';
import { KnowledgeGraphService } from './knowledge-graph.service';
import { KnowledgeNode, NodeType } from './entities/knowledge-node.entity';
import { KnowledgeEdge, EdgeType } from './entities/knowledge-edge.entity';

@Resolver()
export class WikiFixResolver {
    constructor(private readonly graphService: KnowledgeGraphService) { }

    @Query(() => [KnowledgeNode])
    async searchKnowledge(@Args('query') query: string): Promise<KnowledgeNode[]> {
        return this.graphService.semanticSearch(query);
    }

    @Query(() => [KnowledgeNode])
    async relatedFixes(@Args('nodeId', { type: () => ID }) nodeId: string): Promise<KnowledgeNode[]> {
        return this.graphService.findRelatedNodes(nodeId);
    }

    @Mutation(() => KnowledgeEdge)
    async addKnowledgeRelation(
        @Args('sourceId', { type: () => ID }) sourceId: string,
        @Args('targetId', { type: () => ID }) targetId: string,
        @Args('relation', { type: () => EdgeType }) relation: EdgeType,
    ): Promise<KnowledgeEdge> {
        return this.graphService.createRelationship(sourceId, targetId, relation);
    }
}

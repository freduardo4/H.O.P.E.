import { Entity, PrimaryGeneratedColumn, Column, CreateDateColumn, UpdateDateColumn, ManyToOne, OneToMany, JoinColumn } from 'typeorm';
import { ObjectType, Field, ID, Int } from '@nestjs/graphql';
import { User } from '../../auth/entities/user.entity';
import { RepairPattern } from './repair-pattern.entity';

@ObjectType()
@Entity('wiki_posts')
export class WikiPost {
    @Field(() => ID)
    @PrimaryGeneratedColumn('uuid')
    id: string;

    @Field()
    @Column()
    title: string;

    @Field()
    @Column({ type: 'text' })
    content: string;

    @Field(() => [String])
    @Column({ type: 'simple-json', default: '[]' })
    tags: string[];

    @Field(() => Int)
    @Column({ default: 0 })
    upvotes: number;

    @Field(() => Int)
    @Column({ default: 0 })
    downvotes: number;

    @Field()
    @Column()
    authorId: string;

    @Field(() => User)
    @ManyToOne(() => User)
    @JoinColumn({ name: 'authorId' })
    author: User;

    @Field(() => [RepairPattern], { nullable: true })
    @OneToMany('RepairPattern', 'post')
    patterns: RepairPattern[];

    @Field()
    @CreateDateColumn()
    createdAt: Date;

    @Field()
    @UpdateDateColumn()
    updatedAt: Date;
}

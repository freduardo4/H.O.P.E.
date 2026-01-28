import { Entity, PrimaryGeneratedColumn, Column, CreateDateColumn, UpdateDateColumn, ManyToOne, OneToMany, JoinColumn } from 'typeorm';
import { User } from '../../auth/entities/user.entity';
import { RepairPattern } from './repair-pattern.entity';

@Entity('wiki_posts')
export class WikiPost {
    @PrimaryGeneratedColumn('uuid')
    id: string;

    @Column()
    title: string;

    @Column({ type: 'text' })
    content: string;

    @Column({ type: 'text', array: true, default: [] })
    tags: string[];

    @Column({ default: 0 })
    upvotes: number;

    @Column({ default: 0 })
    downvotes: number;

    @Column()
    authorId: string;

    @ManyToOne(() => User)
    @JoinColumn({ name: 'authorId' })
    author: User;

    @OneToMany('RepairPattern', 'post')
    patterns: RepairPattern[];

    @CreateDateColumn()
    createdAt: Date;

    @UpdateDateColumn()
    updatedAt: Date;
}

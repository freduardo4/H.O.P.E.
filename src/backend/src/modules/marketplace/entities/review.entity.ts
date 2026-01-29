import { Entity, PrimaryGeneratedColumn, Column, CreateDateColumn, UpdateDateColumn, ManyToOne } from 'typeorm';
import { ObjectType, Field, ID, Int } from '@nestjs/graphql';
import { CalibrationListing } from './calibration-listing.entity';
import { User } from '../../auth/entities/user.entity';

@ObjectType()
@Entity('calibration_reviews')
export class Review {
    @Field(() => ID)
    @PrimaryGeneratedColumn('uuid')
    id: string;

    @Field(() => Int)
    @Column({ type: 'int' })
    rating: number; // 1-5

    @Field({ nullable: true })
    @Column({ type: 'text', nullable: true })
    comment: string;

    @Field(() => CalibrationListing)
    @ManyToOne(() => CalibrationListing, (listing) => listing.reviews)
    listing: CalibrationListing;

    @Field(() => User)
    @ManyToOne(() => User)
    user: User;

    @Field()
    @CreateDateColumn()
    createdAt: Date;

    @Field()
    @UpdateDateColumn()
    updatedAt: Date;
}
